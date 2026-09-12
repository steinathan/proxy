package transformer

import (
	"encoding/json"
	"testing"

	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/pkg/types"
)

// decodeInputs unmarshals a ResponsesRequest Input (json.RawMessage) into a
// []ResponsesInput for tests that need to inspect items by field.
func decodeInputs(raw json.RawMessage) []types.ResponsesInput {
	var out []types.ResponsesInput
	if err := json.Unmarshal(raw, &out); err != nil {
		panic(err)
	}
	return out
}

func TestTransformToResponses_ToolResultInlinesAsText(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "muse-spark-1.2-contributor",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{"city":"Kigali"}}
				]`),
			},
			{
				Role: "user",
				Content: json.RawMessage(`[
					{"type":"tool_result","tool_use_id":"toolu_123","content":"sunny, 22C"}
				]`),
			},
		},
	}

	res, err := transformer.TransformToResponses(req, config.ModelConfig{ModelID: "muse-spark-1.2-contributor"})
	if err != nil {
		t.Fatalf("TransformToResponses() error = %v", err)
	}

	resInputs := decodeInputs(res.Input)

	for i, item := range resInputs {
		if item.Role == "tool" {
			t.Fatalf("input[%d]: role \"tool\" leaked into input array; tool_result must be function_call_output", i)
		}
	}

	if len(resInputs) < 3 {
		t.Fatalf("len(Input) = %d, want at least 3 (user, function_call, function_call_output)", len(resInputs))
	}

	// Find the function_call_output for toolu_123
	found := false
	for _, item := range resInputs {
		if item.Type == "function_call_output" && item.CallID == "toolu_123" && item.Output == "sunny, 22C" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("no function_call_output for toolu_123 with output sunny, 22C in %v", resInputs)
	}
}

func TestTransformToResponses_ToolResultWithoutTextStillEmits(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "muse-spark-1.2-contributor",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "user",
				Content: json.RawMessage(`[
					{"type":"tool_result","tool_use_id":"toolu_999","content":"done"}
				]`),
			},
		},
	}

	res, err := transformer.TransformToResponses(req, config.ModelConfig{ModelID: "muse-spark-1.2-contributor"})
	if err != nil {
		t.Fatalf("TransformToResponses() error = %v", err)
	}

	resInputs := decodeInputs(res.Input)

	for i, item := range resInputs {
		if item.Role == "tool" {
			t.Fatalf("input[%d]: role \"tool\" leaked into input array", i)
		}
	}

	if len(resInputs) != 1 {
		t.Fatalf("len(Input) = %d, want 1 (the function_call_output)", len(resInputs))
	}
	if resInputs[0].Type != "function_call_output" || resInputs[0].CallID != "toolu_999" || resInputs[0].Output != "done" {
		t.Fatalf("input[0] = %+v, want function_call_output with call_id toolu_999 and output done", resInputs[0])
	}
}
