package handlers

// Anthropic → OpenAI Responses wire-format translator.
//
// The Responses handler's upstream dispatch goes through NormalizedRequest so
// any provider can speak whatever format it wants on the wire. The Responses
// client (Codex CLI, ChatGPT.app) on the other side only understands Responses
// SSE/JSON. This file bridges the gap when the chosen upstream is Anthropic —
// typically the minimax provider, which the `model_family_overrides` chain
// picks for Claude Code and any `model_overrides.gpt-*` entry routes to.
//
// Tool-use is deliberately not translated: text-only blocks pass through. A
// tool_use content_block from minimax is dropped silently and the stream still
// closes with `response.completed`, so codex sees a clean turn end. When a codex
// user needs tools through this path, swap in a Responses-native upstream or
// extend this translator.

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/routatic/proxy/internal/transformer"
	"github.com/routatic/proxy/pkg/types"
)

// proxyAnthropicToResponsesStream reads Anthropic SSE from upstream and writes
// Responses SSE to the client. Supports text content blocks only — tool calls
// pass through as a no-op so the stream still terminates cleanly.
func proxyAnthropicToResponsesStream(
	w http.ResponseWriter,
	body io.ReadCloser,
	requestedModel string,
	idleTimeout time.Duration,
	clientCtx context.Context,
	cancel context.CancelFunc,
	logger *slog.Logger,
) error {
	defer func() { _ = body.Close() }()
	defer cancel()

	flusher, ok := w.(http.Flusher)
	if !ok {
		return errors.New("streaming not supported by response writer")
	}

	st := &anthropicStreamState{
		responseID: "resp_" + uuid.NewString(),
		itemID:     "msg_" + uuid.NewString(),
		model:      requestedModel,
		emittedDone: false,
	}

	reader := bufio.NewReader(body)
	ping := transformer.StartIdleWatchdog(clientCtx, cancel, idleTimeout)

	for {
		select {
		case <-clientCtx.Done():
			if clientCtx.Err() == nil {
				return transformer.ErrStreamIdle
			}
			return transformer.ErrClientDisconnected
		default:
		}

		event, data, err := readSSEFrame(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return emitResponsesCompleted(w, flusher, st)
			}
			if transformer.IsIdleTimeout(err) {
				return transformer.ErrStreamIdle
			}
			if errors.Is(err, context.Canceled) || clientCtx.Err() != nil {
				return transformer.ErrClientDisconnected
			}
			return err
		}
		if event == "" || data == "" {
			continue
		}
		ping()

		if err := dispatchAnthropicEvent(w, flusher, st, event, data, logger); err != nil {
			return err
		}
		if st.emittedDone {
			return nil
		}
	}
}

type anthropicStreamState struct {
	responseID  string
	itemID      string
	model       string
	textStarted bool
	textAccum   string
	inputTokens int
	outputTokens int
	emittedDone bool
}

func dispatchAnthropicEvent(
	w http.ResponseWriter,
	flusher http.Flusher,
	st *anthropicStreamState,
	event, data string,
	logger *slog.Logger,
) error {
	switch event {
	case "message_start":
		var frame struct {
			Message struct {
				ID    string `json:"id"`
				Model string `json:"model"`
				Usage struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
				} `json:"usage"`
			} `json:"message"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			return nil
		}
		if frame.Message.ID != "" {
			st.responseID = "resp_" + strings.TrimPrefix(frame.Message.ID, "msg_")
		}
		if frame.Message.Usage.InputTokens > 0 {
			st.inputTokens = frame.Message.Usage.InputTokens
		}
		if frame.Message.Usage.OutputTokens > 0 {
			st.outputTokens = frame.Message.Usage.OutputTokens
		}
		resp := types.ResponsesResponse{
			ID:      st.responseID,
			Object:  "response",
			Model:   st.model,
			Output:  []types.ResponsesOutput{},
			Created: time.Now().Unix(),
		}
		if err := writeResponsesSSE(w, flusher, "response.created", types.ResponsesChunk{
			Type:     "response.created",
			Response: &resp,
		}); err != nil {
			return err
		}
		// Emit response.in_progress immediately — codex's parser expects this
		// event between `created` and the first output item, otherwise the
		// item never gets attached to the active response.
		return writeResponsesSSE(w, flusher, "response.in_progress", types.ResponsesChunk{
			Type:     "response.in_progress",
			Response: &resp,
		})

	case "content_block_start":
		var frame struct {
			Index        int `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content_block"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			return nil
		}
		if frame.ContentBlock.Type != "text" || st.textStarted {
			return nil // tool_use / non-text blocks: drop silently (known ceiling)
		}
		st.textStarted = true
		st.textAccum = frame.ContentBlock.Text
		outputIndex := 0
		// The Responses spec expects role:"assistant" + status:"in_progress"
		// inside the item so client parsers (codex) can attach incoming text
		// deltas to the right output item. Codex (Rust) uses
		// `#[serde(tag = "type")]` for ResponseItem, so the inner `type` MUST
		// be exactly `"message"` and content MUST be a list.
		//
		// Anthropic-format upstreams (minimax + any future minimax-like
		// provider) have no commentary/final-answer distinction — every
		// assistant text block is a final answer. Marking `phase` here is
		// what unblocks codex's renderer: without it the parser treats the
		// item as commentary and `last_agent_message` ends up null.
		//
		// Content lives on the `content_part.added` event, not on the item
		// itself (mirrors opencodex bridge.ts:999). Keeping content empty in
		// the item lets the parser handle stream-of-parts cleanly.
		if err := writeResponsesSSE(w, flusher, "response.output_item.added", map[string]any{
			"type":         "response.output_item.added",
			"output_index": outputIndex,
			"item_id":      st.itemID,
			"item": map[string]any{
				"id":      st.itemID,
				"type":    "message",
				"role":    "assistant",
				"status":  "in_progress",
				"phase":   "final_answer",
				"content": []any{},
			},
		}); err != nil {
			return err
		}
		return writeResponsesSSE(w, flusher, "response.content_part.added", map[string]any{
			"type":          "response.content_part.added",
			"item_id":       st.itemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        frame.ContentBlock.Text,
				"annotations": []any{},
			},
		})

	case "content_block_delta":
		if !st.textStarted {
			return nil
		}
		var frame struct {
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			return nil
		}
		if frame.Delta.Type != "text_delta" {
			return nil // input_json_delta (tool arg build): drop
		}
		st.textAccum += frame.Delta.Text
		return writeResponsesSSE(w, flusher, "response.output_text.delta", map[string]any{
			"type":          "response.output_text.delta",
			"item_id":       st.itemID,
			"output_index":  0,
			"content_index": 0,
			"delta":         frame.Delta.Text,
			"logprobs":      []any{},
		})

	case "content_block_stop":
		if !st.textStarted {
			return nil
		}
		if err := writeResponsesSSE(w, flusher, "response.output_text.done", map[string]any{
			"type":          "response.output_text.done",
			"item_id":       st.itemID,
			"output_index":  0,
			"content_index": 0,
			"text":          st.textAccum,
			"logprobs":      []any{},
		}); err != nil {
			return err
		}
		if err := writeResponsesSSE(w, flusher, "response.content_part.done", map[string]any{
			"type":          "response.content_part.done",
			"item_id":       st.itemID,
			"output_index":  0,
			"content_index": 0,
		}); err != nil {
			return err
		}
		// Final item snapshot — codex commits to the live item only after
		// seeing this with `status:"completed"` + same `phase` we tagged at
		// output_item.added time (opencodex bridge.ts:636 mirrors this).
		if err := writeResponsesSSE(w, flusher, "response.output_item.done", map[string]any{
			"type":        "response.output_item.done",
			"output_index": 0,
			"item_id":     st.itemID,
			"item": map[string]any{
				"id":      st.itemID,
				"type":    "message",
				"role":    "assistant",
				"status":  "completed",
				"phase":   "final_answer",
				"content": []any{},
			},
		}); err != nil {
			return err
		}
		st.textStarted = false

	case "message_delta":
		var frame struct {
			Usage struct {
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err == nil && frame.Usage.OutputTokens > 0 {
			st.outputTokens = frame.Usage.OutputTokens
		}
		return nil

	case "message_stop":
		return emitResponsesCompleted(w, flusher, st)

	case "ping", "error":
		// drop: ping is keepalive, error terminates upstream but message_stop
		// (or lack thereof) decides how we close.
		return nil
	}
	return nil
}

func emitResponsesCompleted(w http.ResponseWriter, flusher http.Flusher, st *anthropicStreamState) error {
	if st.emittedDone {
		return nil
	}
	resp := types.ResponsesResponse{
		ID:      st.responseID,
		Object:  "response",
		Model:   st.model,
		Created: time.Now().Unix(),
		Output: []types.ResponsesOutput{{
			Type: "message",
			ID:   st.itemID,
			Role: "assistant",
			Content: []types.ResponsesContent{{
				Type: "output_text",
				Text: st.textAccum,
			}},
		}},
		Usage: types.ResponsesUsage{
			InputTokens:  st.inputTokens,
			OutputTokens: st.outputTokens,
			TotalTokens:  st.inputTokens + st.outputTokens,
		},
	}
	if err := writeResponsesSSE(w, flusher, "response.completed", types.ResponsesChunk{
		Type:     "response.completed",
		Response: &resp,
	}); err != nil {
		return err
	}
	st.emittedDone = true
	return nil
}

// writeResponsesSSE marshals payload to JSON, writes a single Responses-style
// SSE frame (`event: <type>\ndata: <json>\n\n`) and flushes.
func writeResponsesSSE(w http.ResponseWriter, flusher http.Flusher, event string, payload any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, body); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

// readSSEFrame reads one Anthropic-style SSE frame: a sequence of `event:` and
// `data:` lines terminated by a blank line. Returns the event name and data
// string. On EOF with no frame, returns io.EOF.
func readSSEFrame(r *bufio.Reader) (event, data string, err error) {
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return "", "", err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if event != "" || data != "" {
				return event, data, nil
			}
			// blank line between frames — keep reading
			continue
		}
		switch {
		case strings.HasPrefix(line, "event: "):
			event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			data = strings.TrimPrefix(line, "data: ")
		case strings.HasPrefix(line, "data:"):
			data = strings.TrimPrefix(line, "data:")
		case strings.HasPrefix(line, ":"):
			// SSE comment / keepalive — ignore
		}
	}
}

// anthropicBodyToResponses converts an Anthropic non-streaming response body
// into a Responses-shaped JSON body. Used when the resolved upstream speaks
// Anthropic (e.g. minimax provider). Returns the original body if the upstream
// isn't actually Anthropic-shaped so callers don't accidentally rewrite good
// Responses bodies.
func anthropicBodyToResponses(body []byte, requestedModel string) []byte {
	var ar struct {
		ID         string `json:"id"`
		Type       string `json:"type"`
		Model      string `json:"model"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	// Anthropic bodies always carry `type:"message"`; Responses bodies do not.
	// If that field is missing we assume the upstream already speaks Responses
	// and pass the body through unchanged.
	if err := json.Unmarshal(body, &ar); err != nil || ar.Type != "message" || ar.ID == "" {
		return body
	}

	var text string
	for _, c := range ar.Content {
		if c.Type == "text" {
			text += c.Text
		}
	}
	// tool_use blocks are silently dropped — same ceiling as the streaming
	// translator. When codex sends a tool-use request to a non-Responses
	// upstream, this branch returns an empty output.

	resp := types.ResponsesResponse{
		ID:      "resp_" + strings.TrimPrefix(ar.ID, "msg_"),
		Object:  "response",
		Model:   requestedModel,
		Created: time.Now().Unix(),
		Output: []types.ResponsesOutput{{
			Type: "message",
			ID:   "msg_" + uuid.NewString(),
			Role: "assistant",
			Content: []types.ResponsesContent{{
				Type: "output_text",
				Text: text,
			}},
		}},
		Usage: types.ResponsesUsage{
			InputTokens:  ar.Usage.InputTokens,
			OutputTokens: ar.Usage.OutputTokens,
			TotalTokens:  ar.Usage.InputTokens + ar.Usage.OutputTokens,
		},
	}
	// Marshal into a wrapper map so we can include `status` which the
	// ResponsesResponse struct doesn't carry as a field (the SDK uses omitempty
	// on Output and a few others, so we attach status via a separate write).
	out, err := json.Marshal(resp)
	if err != nil {
		return body
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		return out
	}
	m["status"] = "completed"
	final, err := json.Marshal(m)
	if err != nil {
		return out
	}
	return final
}
