package transformer

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/routatic/proxy/pkg/types"
)

// MiniMax M3 streams OpenAI-standard `reasoning` (not DeepSeek's
// `reasoning_content`) with content:null. It must be relayed to Claude Code
// as Anthropic thinking_delta, otherwise the stream is empty (output_tokens=0).
func TestMiniMaxM3ReasoningRelayed(t *testing.T) {
	chunk := types.ChatCompletionChunk{
		Choices: []types.Choice{{
			Delta: types.ChatMessage{
				Content:   json.RawMessage(`null`),
				Reasoning: strptr("thinking about the answer"),
			},
			FinishReason: "length",
		}},
	}
	data, _ := json.Marshal(chunk)

	rec := newMockResponseWriter()
	h := NewStreamHandler()
	ctx := context.Background()
	_ = h.ProxyStream(rec, sseLines(string(data)), "minimax/minimax-m3", ctx, 0, func() {})

	events := parseSSEEvents(t, rec.buf.String())
	var thinking string
	for _, ev := range events {
		if ev.Type == "content_block_delta" && ev.Delta != nil && ev.Delta.Type == "thinking_delta" {
			thinking += ev.Delta.Thinking
		}
	}
	if thinking != "thinking about the answer" {
		t.Fatalf("reasoning not relayed as thinking_delta. got %q", thinking)
	}
}

func strptr(s string) *string { return &s }

var _ = time.Second // keep time import if unused in future edits
