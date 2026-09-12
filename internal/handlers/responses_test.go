package handlers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/routatic/proxy/pkg/types"
)

// mkInput mirrors the core test helper: marshal []ResponsesInput to
// json.RawMessage for fixtures.
func mkInput(items []types.ResponsesInput) json.RawMessage {
	b, err := json.Marshal(items)
	if err != nil {
		panic(err)
	}
	return b
}

func jsonRaw(s string) json.RawMessage { return json.RawMessage(s) }

func responsesBoolPtr(b bool) *bool { return &b }


func TestValidateResponsesRequest(t *testing.T) {
	cases := []struct {
		name    string
		req     *types.ResponsesRequest
		wantErr string
	}{
		{
			name:    "empty model",
			req:     &types.ResponsesRequest{},
			wantErr: "model is required",
		},
		{
			name: "valid minimal",
			req: &types.ResponsesRequest{
				Model: "gpt-5",
				Input: mkInput([]types.ResponsesInput{{Role: "user", Content: jsonRaw(`"hi"`)}}),
			},
			wantErr: "",
		},
		{
			name: "previous_response_id rejected",
			req: &types.ResponsesRequest{
				Model:               "gpt-5",
				Input: mkInput([]types.ResponsesInput{{Role: "user", Content: jsonRaw(`"hi"`)}}),
				PreviousResponseID:  "resp_xyz",
			},
			wantErr: "previous_response_id",
		},
		{
			name: "background rejected",
			req: &types.ResponsesRequest{
				Model:      "gpt-5",
				Input: mkInput([]types.ResponsesInput{{Role: "user", Content: jsonRaw(`"hi"`)}}),
				Background: responsesBoolPtr(true),
			},
			wantErr: "background",
		},
		{
			name: "text.format rejected",
			req: &types.ResponsesRequest{
				Model: "gpt-5",
				Input: mkInput([]types.ResponsesInput{{Role: "user", Content: jsonRaw(`"hi"`)}}),
				Text:  &types.ResponsesTextFormat{Format: jsonRaw(`{"type":"json_schema"}`)},
			},
			wantErr: "text.format",
		},
		{
			name: "reasoning.effort rejected",
			req: &types.ResponsesRequest{
				Model: "gpt-5",
				Input: mkInput([]types.ResponsesInput{{Role: "user", Content: jsonRaw(`"hi"`)}}),
				Reasoning: &types.ResponsesReasoning{Effort: "high"},
			},
			wantErr: "reasoning",
		},
		{
			name: "truncation=auto rejected",
			req: &types.ResponsesRequest{
				Model:     "gpt-5",
				Input: mkInput([]types.ResponsesInput{{Role: "user", Content: jsonRaw(`"hi"`)}}),
				Truncation: "auto",
			},
			wantErr: "truncation",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := validateResponsesRequest(c.req)
			if c.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), c.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, c.wantErr)
			}
		})
	}
}
