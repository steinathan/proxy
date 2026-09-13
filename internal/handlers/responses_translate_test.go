package handlers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/routatic/proxy/internal/transformer"
)

func TestProxyAnthropicToResponsesStream_TextPath(t *testing.T) {
	upstream := strings.Join([]string{
		"event: message_start",
		`data: {"message":{"id":"msg_abc","model":"minimax-m3","usage":{"input_tokens":42,"output_tokens":0}}}`,
		"",
		"event: content_block_start",
		`data: {"index":0,"content_block":{"type":"text","text":""}}`,
		"",
		"event: content_block_delta",
		`data: {"index":0,"delta":{"type":"text_delta","text":"Hello "}}`,
		"",
		"event: content_block_delta",
		`data: {"index":0,"delta":{"type":"text_delta","text":"world"}}`,
		"",
		"event: content_block_stop",
		`data: {"index":0}`,
		"",
		"event: message_delta",
		`data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		"",
		"event: message_stop",
		`data: {}`,
		"",
	}, "\n")

	rr := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(upstream))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := proxyAnthropicToResponsesStream(rr, body, "gpt-5", time.Second, ctx, cancel, slogDefault())
	if err != nil {
		t.Fatalf("translator returned error: %v", err)
	}

	out := rr.Body.String()
	for _, want := range []string{
		`event: response.created`,
		`event: response.output_item.added`,
		`event: response.content_part.added`,
		`event: response.output_text.delta`,
		`"delta":"Hello "`,
		`"delta":"world"`,
		`event: response.output_text.done`,
		`event: response.content_part.done`,
		`event: response.output_item.done`,
		`event: response.completed`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\nfull output:\n%s", want, out)
		}
	}

	// Item must carry phase:"final_answer" on both added and done —
	// codex renders messages tagged this way as the agent's final answer
	// rather than as commentary. Without it, last_agent_message is null.
	if !strings.Contains(out, `"phase":"final_answer"`) {
		t.Errorf(`output_item.added and output_item.done must carry phase:"final_answer" so codex renders the message as the agent's final answer\nfull output:\n%s`, out)
	}
	// Item must have role:"assistant" — codex's #[serde(tag="type")] ResponseItem
	// parser rejects assistant messages with the wrong role.
	if !strings.Contains(out, `"role":"assistant"`) {
		t.Errorf(`output_item.added must carry role:"assistant"\nfull output:\n%s`, out)
	}

	// Final response should carry the requested model and the accumulated text.
	if !strings.Contains(out, `"model":"gpt-5"`) {
		t.Errorf("response.completed should echo client-requested model")
	}
	if !strings.Contains(out, `"text":"Hello world"`) {
		t.Errorf("accumulated text not present in response.completed output")
	}
}

func TestProxyAnthropicToResponsesStream_ToolUseDropped(t *testing.T) {
	upstream := strings.Join([]string{
		"event: message_start",
		`data: {"message":{"id":"msg_x","model":"minimax-m3","usage":{"input_tokens":5,"output_tokens":0}}}`,
		"",
		"event: content_block_start",
		`data: {"index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}`,
		"",
		"event: content_block_delta",
		`data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\""}}`,
		"",
		"event: content_block_delta",
		`data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"Kigali\"}"}}`,
		"",
		"event: content_block_stop",
		`data: {"index":0}`,
		"",
		"event: message_stop",
		`data: {}`,
		"",
	}, "\n")

	rr := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(upstream))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := proxyAnthropicToResponsesStream(rr, body, "gpt-5", time.Second, ctx, cancel, slogDefault())
	if err != nil {
		t.Fatalf("translator returned error: %v", err)
	}

	out := rr.Body.String()
	// tool_use path: response.created + response.completed, nothing in between.
	if !strings.Contains(out, "event: response.created") {
		t.Errorf("missing response.created")
	}
	if !strings.Contains(out, "event: response.completed") {
		t.Errorf("missing response.completed (stream must still terminate cleanly)")
	}
	if strings.Contains(out, "response.output_text.delta") {
		t.Errorf("should not have emitted text deltas for tool_use block")
	}
}

func TestAnthropicBodyToResponses_BasicText(t *testing.T) {
	body := []byte(`{"id":"msg_xyz","type":"message","role":"assistant","content":[{"type":"text","text":"hi there"}],"model":"minimax-m3","stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}`)

	out := anthropicBodyToResponses(body, "gpt-5")
	var rr struct {
		ID     string `json:"id"`
		Object string `json:"object"`
		Model  string `json:"model"`
		Status string `json:"status"`
		Output []struct {
			Type   string `json:"type"`
			Role   string `json:"role"`
			Status string `json:"status"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(out, &rr); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if rr.ID != "resp_xyz" {
		t.Errorf("id: got %q want %q", rr.ID, "resp_xyz")
	}
	if rr.Object != "response" {
		t.Errorf("object: got %q want response", rr.Object)
	}
	if rr.Model != "gpt-5" {
		t.Errorf("model: got %q want gpt-5", rr.Model)
	}
	if rr.Status != "completed" {
		t.Errorf("status: got %q want completed", rr.Status)
	}
	if len(rr.Output) != 1 || rr.Output[0].Role != "assistant" {
		t.Fatalf("expected one assistant output, got %+v", rr.Output)
	}
	if len(rr.Output[0].Content) != 1 || rr.Output[0].Content[0].Text != "hi there" {
		t.Errorf("content not translated: %+v", rr.Output[0].Content)
	}
	if rr.Usage.InputTokens != 3 || rr.Usage.OutputTokens != 2 || rr.Usage.TotalTokens != 5 {
		t.Errorf("usage: %+v", rr.Usage)
	}
}

func TestAnthropicBodyToResponses_PassesNonAnthropicThrough(t *testing.T) {
	body := []byte(`{"object":"response","id":"resp_native","output":[],"model":"gpt-5","usage":{}}`)
	out := anthropicBodyToResponses(body, "gpt-5")
	if !bytes.Equal(out, body) {
		t.Errorf("non-Anthropic body should pass through unchanged")
	}
}

// slogDefault returns a default slog logger for tests so the translator's
// signature stays consistent with production.
func slogDefault() *slog.Logger { return slog.Default() }

// touch imports so the linter doesn't complain about unused pulls when we
// drop tests.
var (
	_ = uuid.NewString
	_ = bufio.NewReader
	_ = transformer.StartIdleWatchdog
)
