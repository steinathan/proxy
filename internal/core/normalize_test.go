package core

import (
	"encoding/json"
	"testing"

	"github.com/routatic/proxy/pkg/types"
)

func TestNormalizeRequestPreservesOrderedBlocksAndCacheDirectives(t *testing.T) {
	cache := &types.CacheControl{Type: "ephemeral"}
	req := &types.MessageRequest{
		Model: "test",
		Messages: []types.Message{{
			Role: "user",
			Content: json.RawMessage(`[
				{"type":"text","text":"before","cache_control":{"type":"ephemeral"}},
				{"type":"custom_provider_block","payload":{"value":42}},
				{"type":"text","text":"after"}
			]`),
		}},
		Tools: []types.Tool{{
			Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`), CacheControl: cache,
		}},
	}

	normalized := NormalizeRequest(req)
	if len(normalized.Messages) != 1 || len(normalized.Messages[0].Blocks) != 3 {
		t.Fatalf("ordered blocks were not retained: %+v", normalized.Messages)
	}
	if normalized.Messages[0].Blocks[0].CacheControl == nil ||
		normalized.Messages[0].Blocks[0].CacheControl.Type != "ephemeral" {
		t.Fatalf("text cache directive was lost: %+v", normalized.Messages[0].Blocks[0])
	}
	if got := string(normalized.Messages[0].Blocks[1].Raw); got == "" ||
		normalized.Messages[0].Blocks[1].Type != "custom_provider_block" {
		t.Fatalf("unknown block was not preserved: type=%q raw=%q",
			normalized.Messages[0].Blocks[1].Type, got)
	}
	if normalized.Tools[0].CacheControl == nil {
		t.Fatal("tool cache directive was lost")
	}
}

func TestNormalizeRequest_StripsAdaptiveThinkingType(t *testing.T) {
	// Claude Code / Codex send thinking.type = "adaptive" or "auto", which
	// OpenCode Go rejects (unknown variant). Normalization must drop the
	// effort so the request goes through without forcing a value.
	for _, tc := range []struct {
		thinkingType string
		wantEffort   string
	}{
		{"adaptive", ""},
		{"auto", ""},
		{"enabled", "enabled"},
		{"disabled", "disabled"},
	} {
		req := &types.MessageRequest{
			Model:    "claude-sonnet-5",
			Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
			Thinking: json.RawMessage(`{"type":"` + tc.thinkingType + `","budget_tokens":5000}`),
		}
		nr := NormalizeRequest(req)
		if nr.ReasoningEffort != tc.wantEffort {
			t.Errorf("thinking.type=%q: ReasoningEffort = %q, want %q", tc.thinkingType, nr.ReasoningEffort, tc.wantEffort)
		}
		if nr.ThinkingBudget != 5000 {
			t.Errorf("thinking.type=%q: ThinkingBudget = %d, want 5000", tc.thinkingType, nr.ThinkingBudget)
		}
	}
}

func TestNormalizeRequestPreservesLegacyToolResultOutput(t *testing.T) {
	req := &types.MessageRequest{
		Model: "test",
		Messages: []types.Message{{
			Role: "tool",
			Content: json.RawMessage(`[
				{"type":"tool_result","tool_use_id":"call_1","output":"legacy result"}
			]`),
		}},
	}

	normalized := NormalizeRequest(req)
	if got, want := normalized.Messages[0].ToolResultsList()[0].Content, "legacy result"; got != want {
		t.Fatalf("legacy tool result content = %q, want %q", got, want)
	}
	if got, want := string(normalized.Messages[0].Blocks[0].Content), `"legacy result"`; got != want {
		t.Fatalf("normalized tool result content = %s, want %s", got, want)
	}
}
