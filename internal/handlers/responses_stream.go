package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/routatic/proxy/internal/transformer"
)

// proxyResponsesPassthroughStream forwards raw OpenAI Responses SSE bytes
// directly from the upstream to the client. On the first `response.created`
// event, the nested `response.model` field is rewritten to requestedModel so
// Codex sees the user's requested model name and can resume sessions cleanly.
//
// Mirrors proxyAnthropicPassthroughStream (streaming.go) — same byte-level
// SSE framing (`event: TYPE\ndata: JSON\n\n`), same idle watchdog, same
// client-disconnect handling. The only difference is the model-rewrite
// target: `response.model` here vs `message.model` for Anthropic.
func proxyResponsesPassthroughStream(
	w http.ResponseWriter,
	body io.ReadCloser,
	requestedModel string,
	idleTimeout time.Duration,
	clientCtx context.Context,
	cancel context.CancelFunc,
) error {
	defer func() { _ = body.Close() }()
	defer cancel()

	const sseBoundary = "\n\n"
	const dataPrefix = "data: "
	const createEvent = "event: response.created"

	buf := make([]byte, 4096)
	pending := make([]byte, 0, 8192)
	rewritten := false
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
		n, rerr := body.Read(buf)
		if n > 0 {
			ping()
			pending = append(pending, buf[:n]...)
			if !rewritten && requestedModel != "" {
				if idx := bytes.Index(pending, []byte(sseBoundary)); idx >= 0 {
					frame := pending[:idx]
					pending = pending[idx+len(sseBoundary):]
					rewritten = true
					// Match the leading-event form (first frame) or the mid-stream
					// form (preceded by a leftover newline from a prior boundary).
					if bytes.HasPrefix(frame, []byte(createEvent+"\n")) ||
						bytes.Contains(frame, []byte("\n"+createEvent+"\n")) {
						fixed := rewriteResponsesCreatedModel(frame, requestedModel)
						if _, werr := w.Write(fixed); werr != nil {
							return transformer.ErrClientDisconnected
						}
					} else {
						if _, werr := w.Write(frame); werr != nil {
							return transformer.ErrClientDisconnected
						}
					}
					if _, werr := w.Write([]byte(sseBoundary)); werr != nil {
						return transformer.ErrClientDisconnected
					}
					if f, ok := w.(http.Flusher); ok {
						f.Flush()
					}
				} else if len(pending) > 65536 {
					// Upstream preamble exceeded 64 KB without a frame boundary —
					// flush to avoid head-of-line blocking, drop the buffered
					// partial frame, and mark as already-rewritten so the read
					// loop below drains the rest verbatim.
					if _, werr := w.Write(pending); werr != nil {
						return transformer.ErrClientDisconnected
					}
					pending = pending[:0]
					rewritten = true
				}
			}
			if rewritten && len(pending) > 0 {
				if _, werr := w.Write(pending); werr != nil {
					return transformer.ErrClientDisconnected
				}
				pending = pending[:0]
			}
			if rewritten {
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}
		if rerr == io.EOF {
			if len(pending) > 0 {
				_, _ = w.Write(pending)
			}
			return nil
		}
		if rerr != nil {
			if transformer.IsIdleTimeout(rerr) {
				return transformer.ErrStreamIdle
			}
			if errors.Is(rerr, context.Canceled) || errors.Is(rerr, transformer.ErrStreamReadCanceled) || clientCtx.Err() == context.Canceled {
				if clientCtx.Err() == nil {
					return transformer.ErrStreamIdle
				}
				return transformer.ErrClientDisconnected
			}
			return fmt.Errorf("failed to copy response: %w", rerr)
		}
	}
}

// rewriteResponsesCreatedModel scans a single SSE frame for the
// response.created data line and replaces the nested `response.model` value
// with requestedModel. Non-created frames are passed through unchanged.
func rewriteResponsesCreatedModel(frame []byte, requestedModel string) []byte {
	const eventLine = "event: response.created"
	const dataLinePrefix = "data: "

	var idx int
	if bytes.HasPrefix(frame, []byte(eventLine)) {
		idx = 0
	} else {
		rel := bytes.Index(frame, []byte("\n"+eventLine))
		if rel < 0 {
			return frame
		}
		idx = rel + 1
	}
	after := idx + len(eventLine)
	for after < len(frame) && frame[after] == '\n' {
		after++
	}
	dataRel := bytes.Index(frame[after:], []byte(dataLinePrefix))
	if dataRel < 0 {
		return frame
	}
	dataStart := after + dataRel + len(dataLinePrefix)
	dataEnd := dataStart
	for dataEnd < len(frame) && frame[dataEnd] != '\n' {
		dataEnd++
	}
	var parsed struct {
		Type     string `json:"type"`
		Response struct {
			Model string `json:"model"`
		} `json:"response"`
	}
	if err := json.Unmarshal(frame[dataStart:dataEnd], &parsed); err != nil {
		return frame
	}
	if parsed.Type != "response.created" {
		return frame
	}
	if parsed.Response.Model == requestedModel {
		return frame
	}
	rewrittenData, err := responsesRemarshalDataLine(frame[dataStart:dataEnd], requestedModel)
	if err != nil {
		return frame
	}
	out := make([]byte, 0, len(frame)+len(requestedModel))
	out = append(out, frame[:dataStart]...)
	out = append(out, rewrittenData...)
	out = append(out, frame[dataEnd:]...)
	return out
}

func responsesRemarshalDataLine(data []byte, requestedModel string) ([]byte, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	respRaw, ok := raw["response"]
	if !ok {
		return nil, errors.New("no response field")
	}
	var resp map[string]json.RawMessage
	if err := json.Unmarshal(respRaw, &resp); err != nil {
		return nil, err
	}
	encodedModel, err := json.Marshal(requestedModel)
	if err != nil {
		return nil, err
	}
	resp["model"] = encodedModel
	newRespRaw, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}
	raw["response"] = newRespRaw
	return json.Marshal(raw)
}
