package core

import (
	"encoding/json"
	"testing"

	"github.com/routatic/proxy/pkg/types"
)

// mkInput marshals a []types.ResponsesInput into json.RawMessage for test
// fixtures, mirroring the wire format the input handler decodes.
func mkInput(items []types.ResponsesInput) json.RawMessage {
	b, err := json.Marshal(items)
	if err != nil {
		panic(err)
	}
	return b
}

func TestNormalizeResponsesRequest_TextInput(t *testing.T) {
	req := &types.ResponsesRequest{
		Model:        "gpt-5",
		Instructions: "be helpful",
		Input: mkInput([]types.ResponsesInput{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		}),
	}
	nr := NormalizeResponsesRequest(req)
	if nr.Model != "gpt-5" {
		t.Fatalf("model = %q", nr.Model)
	}
	if nr.SystemPrompt != "be helpful" {
		t.Fatalf("system prompt = %q", nr.SystemPrompt)
	}
	if len(nr.Messages) != 1 {
		t.Fatalf("messages = %d", len(nr.Messages))
	}
	if nr.Messages[0].Role != "user" || nr.Messages[0].TextContent() != "hello" {
		t.Fatalf("user message wrong: %+v", nr.Messages[0])
	}
}

func TestNormalizeResponsesRequest_MultiPartContent(t *testing.T) {
	req := &types.ResponsesRequest{
		Model: "gpt-5",
		Input: mkInput([]types.ResponsesInput{{
			Role:    "user",
			Content: json.RawMessage(`[{"type":"input_text","text":"hi"},{"type":"input_text","text":"there"}]`),
		}}),
	}
	nr := NormalizeResponsesRequest(req)
	if got := nr.Messages[0].TextContent(); got != "hi\nthere" {
		t.Fatalf("joined content = %q, want %q", got, "hi\nthere")
	}
}

func TestNormalizeResponsesRequest_DeveloperRoleAppendsToSystem(t *testing.T) {
	req := &types.ResponsesRequest{
		Model:        "gpt-5",
		Instructions: "first",
		Input: mkInput([]types.ResponsesInput{
			{Role: "developer", Content: json.RawMessage(`"second"`)},
		}),
	}
	nr := NormalizeResponsesRequest(req)
	if nr.SystemPrompt != "first\nsecond" {
		t.Fatalf("system prompt = %q", nr.SystemPrompt)
	}
	if len(nr.Messages) != 0 {
		t.Fatalf("developer item should not produce a user/assistant message")
	}
}

func TestNormalizeResponsesRequest_FunctionCallAndOutput(t *testing.T) {
	req := &types.ResponsesRequest{
		Model: "gpt-5",
		Input: mkInput([]types.ResponsesInput{
			{Type: "function_call", CallID: "call_1", Name: "lookup", Arguments: `{"q":"x"}`},
			{Type: "function_call_output", CallID: "call_1", Output: "result"},
		}),
	}
	nr := NormalizeResponsesRequest(req)
	if len(nr.Messages) != 2 {
		t.Fatalf("messages = %d", len(nr.Messages))
	}
	if nr.Messages[0].Role != "assistant" {
		t.Fatalf("function_call role = %q, want assistant", nr.Messages[0].Role)
	}
	calls := nr.Messages[0].ToolCallsList()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "lookup" {
		t.Fatalf("tool calls wrong: %+v", calls)
	}
	if nr.Messages[1].Role != "user" {
		t.Fatalf("function_call_output role = %q", nr.Messages[1].Role)
	}
	if nr.Messages[1].HasToolCallID() != true {
		t.Fatalf("tool_result missing tool_use_id")
	}
}

func TestNormalizeResponsesRequest_ToolsMap(t *testing.T) {
	req := &types.ResponsesRequest{
		Model: "gpt-5",
		Tools: []types.ResponsesTool{{
			Type:        "function",
			Name:        "search",
			Description: "do a search",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		}},
	}
	nr := NormalizeResponsesRequest(req)
	if len(nr.Tools) != 1 {
		t.Fatalf("tools = %d", len(nr.Tools))
	}
	if nr.Tools[0].Name != "search" || nr.Tools[0].Description != "do a search" {
		t.Fatalf("tool meta wrong: %+v", nr.Tools[0])
	}
	if len(nr.Tools[0].InputSchema) == 0 {
		t.Fatalf("input schema not carried through")
	}
}

func TestNormalizeResponsesRequest_StreamAndNumericFields(t *testing.T) {
	temp := 0.7
	topp := 0.9
	req := &types.ResponsesRequest{
		Model:           "gpt-5",
		Stream:          true,
		Temperature:     &temp,
		TopP:            &topp,
		MaxOutputTokens: 512,
	}
	nr := NormalizeResponsesRequest(req)
	if !nr.Stream {
		t.Fatalf("stream not propagated")
	}
	if nr.MaxTokens != 512 {
		t.Fatalf("max tokens = %d", nr.MaxTokens)
	}
	if nr.Temperature == nil || *nr.Temperature != 0.7 {
		t.Fatalf("temperature not propagated")
	}
	if nr.TopP == nil || *nr.TopP != 0.9 {
		t.Fatalf("top_p not propagated")
	}
}

func TestNormalizeResponsesRequest_UnknownItemTypeDropped(t *testing.T) {
	req := &types.ResponsesRequest{
		Model: "gpt-5",
		Input: mkInput([]types.ResponsesInput{
			{Role: "user", Content: json.RawMessage(`"hi"`)},
			{Type: "image", /* unsupported in MVP */ },
		}),
	}
	nr := NormalizeResponsesRequest(req)
	if len(nr.Messages) != 1 {
		t.Fatalf("unknown item type should be silently dropped; got %d messages", len(nr.Messages))
	}
}

func TestNormalizeResponsesRequest_EmptyArgumentsDefaultsToEmptyObject(t *testing.T) {
	req := &types.ResponsesRequest{
		Model: "gpt-5",
		Input: mkInput([]types.ResponsesInput{
			{Type: "function_call", CallID: "call_1", Name: "noop", Arguments: ""},
		}),
	}
	nr := NormalizeResponsesRequest(req)
	calls := nr.Messages[0].ToolCallsList()
	if len(calls) != 1 || string(calls[0].Arguments) != "{}" {
		t.Fatalf("empty arguments should default to {}; got %+v", calls)
	}
}

func TestNormalizeResponsesRequest_NilSafe(t *testing.T) {
	nr := NormalizeResponsesRequest(nil)
	if nr == nil {
		t.Fatalf("nil request should produce empty normalized, not nil")
	}
	if nr.Model != "" || len(nr.Messages) != 0 {
		t.Fatalf("unexpected content: %+v", nr)
	}
}

func TestNormalizeResponsesRequest_BareStringInput(t *testing.T) {
	req := &types.ResponsesRequest{
		Model: "gpt-5",
		Input: json.RawMessage(`"hi"`),
	}
	nr := NormalizeResponsesRequest(req)
	if len(nr.Messages) != 1 {
		t.Fatalf("bare string should yield one user message; got %d", len(nr.Messages))
	}
	if nr.Messages[0].Role != "user" || nr.Messages[0].TextContent() != "hi" {
		t.Fatalf("message wrong: %+v", nr.Messages[0])
	}
}
