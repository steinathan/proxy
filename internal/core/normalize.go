package core

import (
	"encoding/json"

	"github.com/routatic/proxy/pkg/types"
)

// thinkingConfig mirrors the Anthropic thinking field structure so we can
// decode it without coupling to a specific json.RawMessage layout.
type thinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// NormalizeRequest converts an Anthropic MessageRequest to a NormalizedRequest.
// This is a lossless extraction: all data from the Anthropic format survives.
func NormalizeRequest(anthropicReq *types.MessageRequest) *NormalizedRequest {
	nr := &NormalizedRequest{
		Model:        anthropicReq.Model,
		MaxTokens:    anthropicReq.MaxTokens,
		Stream:       anthropicReq.Stream != nil && *anthropicReq.Stream,
		CacheControl: anthropicReq.CacheControl,
	}

	// Extract system prompt (string or array of content blocks).
	nr.SystemPrompt = anthropicReq.SystemText()
	nr.SystemBlocks = normalizeSystemBlocks(anthropicReq.System)

	// Set temperature if provided.
	if anthropicReq.Temperature != nil {
		nr.Temperature = anthropicReq.Temperature
	}

	// Extract reasoning effort and thinking budget.
	if len(anthropicReq.Thinking) > 0 {
		var tc thinkingConfig
		if err := json.Unmarshal(anthropicReq.Thinking, &tc); err == nil {
			nr.ReasoningEffort = tc.Type
			nr.ThinkingBudget = tc.BudgetTokens
		}
	}

	// Convert messages.
	for _, msg := range anthropicReq.Messages {
		nm := NormalizedMessage{
			Role: msg.Role,
		}

		blocks := msg.ContentBlocks()
		for _, block := range blocks {
			nm.Blocks = append(nm.Blocks, normalizeContentBlock(block))
		}

		nr.Messages = append(nr.Messages, nm)
	}

	// Convert tools.
	for _, tool := range anthropicReq.Tools {
		nt := NormalizedToolDef{
			Name:         tool.Name,
			Description:  tool.Description,
			InputSchema:  tool.InputSchema,
			CacheControl: tool.CacheControl,
		}
		nr.Tools = append(nr.Tools, nt)
	}

	return nr
}

func normalizeSystemBlocks(raw json.RawMessage) []NormalizedContentBlock {
	if len(raw) == 0 {
		return nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []NormalizedContentBlock{{Type: "text", Text: text}}
	}
	var blocks []types.ContentBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return nil
	}
	out := make([]NormalizedContentBlock, 0, len(blocks))
	for _, block := range blocks {
		out = append(out, normalizeContentBlock(block))
	}
	return out
}

func normalizeContentBlock(block types.ContentBlock) NormalizedContentBlock {
	content := block.Content
	if len(content) == 0 && len(block.Output) > 0 {
		content = block.Output
	}

	return NormalizedContentBlock{
		Type:      block.Type,
		Text:      block.Text,
		ID:        block.ID,
		ToolUseID: block.ToolUseID,
		Name:      block.Name,
		Input:     append(json.RawMessage(nil), block.Input...),
		Content:   append(json.RawMessage(nil), content...),
		IsError:   block.IsError,
		Thinking:  block.Thinking,
		Signature: block.Signature,
		Image: func() *NormalizedImage {
			if block.Source == nil {
				return nil
			}
			return &NormalizedImage{
				MediaType: block.Source.MediaType,
				Data:      block.Source.Data,
			}
		}(),
		CacheControl: block.CacheControl,
		Raw:          append(json.RawMessage(nil), block.Raw...),
	}
}

// NormalizeResponsesRequest converts an OpenAI ResponsesRequest to a NormalizedRequest.
// Items with unsupported types are silently dropped (the input-side handler
// rejects well-known unsupported top-level fields like previous_response_id
// before calling this). Instructions and developer-role items fold into the
// system prompt. Function-call items become assistant tool_use blocks;
// function_call_output items become user tool_result blocks.
func NormalizeResponsesRequest(req *types.ResponsesRequest) *NormalizedRequest {
	if req == nil {
		return &NormalizedRequest{}
	}
	nr := &NormalizedRequest{
		Model:        req.Model,
		Stream:       req.Stream,
		MaxTokens:    req.MaxOutputTokens,
		Temperature:  req.Temperature,
		TopP:         req.TopP,
	}
	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		nr.ReasoningEffort = req.Reasoning.Effort
	}

	// Instructions and developer-role items fold into SystemPrompt. If a
	// multi-block instruction array ever shows up, the wire shape becomes
	// `{type:"input_text", text:"..."}`; we join by newline.
	if req.Instructions != "" {
		nr.SystemPrompt = req.Instructions
	}

	for _, item := range DecodeInputItems(req.Input) {
		switch item.Type {
		case "", "message":
			role := item.Role
			if role == "developer" {
				// developer role → system; if we already have an Instructions
				// system prompt, append with a newline separator.
				if text := extractInputText(item.Content); text != "" {
					if nr.SystemPrompt == "" {
						nr.SystemPrompt = text
					} else {
						nr.SystemPrompt += "\n" + text
					}
				}
				continue
			}
			if role == "" {
				role = "user" // bare input string → user message
			}
			text := extractInputText(item.Content)
			if text != "" {
				nr.Messages = append(nr.Messages, NormalizedMessage{
					Role:   role,
					Blocks: []NormalizedContentBlock{{Type: "text", Text: text}},
				})
			}

		case "function_call":
			args := item.Arguments
			if args == "" {
				args = "{}"
			}
			nr.Messages = append(nr.Messages, NormalizedMessage{
				Role: "assistant",
				Blocks: []NormalizedContentBlock{{
					Type:  "tool_use",
					ID:    item.CallID,
					Name:  item.Name,
					Input: json.RawMessage(args),
				}},
			})

		case "function_call_output":
			nr.Messages = append(nr.Messages, NormalizedMessage{
				Role: "user",
				Blocks: []NormalizedContentBlock{{
					Type:      "tool_result",
					ToolUseID: item.CallID,
					Content:   json.RawMessage(item.Output),
				}},
			})

		default:
			// Silently skip unknown item types — image, file, reasoning, etc.
			// The handler rejects the well-known unsupported fields upstream.
		}
	}

	for _, tool := range req.Tools {
		nr.Tools = append(nr.Tools, NormalizedToolDef{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: append([]byte(nil), tool.Parameters...),
		})
	}

	return nr
}

// DecodeInputItems normalizes the polymorphic `input` field into a uniform
// []ResponsesInput. The wire format permits either a bare string shorthand
// (`"hi"` → single user message) or the full array form. Returns nil for an
// empty/missing input. Exported so handlers can re-decode the Input field
// without going through NormalizeResponsesRequest.
func DecodeInputItems(raw json.RawMessage) []types.ResponsesInput {
	if len(raw) == 0 {
		return nil
	}
	// Trim whitespace; if the first non-space byte is `"`, it's a JSON string.
	for _, b := range raw {
		switch b {
		case ' ', '\t', '\n', '\r':
			continue
		case '"':
			var s string
			if err := json.Unmarshal(raw, &s); err == nil {
				encoded, _ := json.Marshal(s)
				return []types.ResponsesInput{{
					Role:    "user",
					Content: encoded,
				}}
			}
			return nil
		default:
			var items []types.ResponsesInput
			if err := json.Unmarshal(raw, &items); err != nil {
				return nil
			}
			return items
		}
	}
	return nil
}

// extractInputText decodes an OpenAI Responses `content` field. The wire format
// allows either a plain string ("hi") or an array of content parts
// ([{"type":"input_text","text":"..."}]); both forms return the joined text.
func extractInputText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	var out string
	for _, p := range parts {
		if p.Type == "input_text" || p.Type == "text" || p.Type == "output_text" {
			if out != "" && p.Text != "" {
				out += "\n"
			}
			out += p.Text
		}
	}
	return out
}

// DenormalizeResponse converts a NormalizedResponse to an Anthropic MessageResponse.
func DenormalizeResponse(nr *NormalizedResponse) *types.MessageResponse {
	resp := &types.MessageResponse{
		ID:    nr.ID,
		Type:  "message",
		Model: nr.Model,
		Usage: types.Usage{
			InputTokens:              nr.Usage.InputTokens,
			OutputTokens:             nr.Usage.OutputTokens,
			CacheCreationInputTokens: nr.Usage.CacheCreationTokens,
			CacheReadInputTokens:     nr.Usage.CacheReadTokens,
		},
	}

	// Build content blocks from messages.
	for _, msg := range nr.Messages {
		switch msg.Role {
		case "assistant":
			resp.Role = "assistant"
			for _, block := range msg.Blocks {
				resp.Content = append(resp.Content, types.ContentBlock{
					Type: block.Type, Text: block.Text, ID: block.ID,
					ToolUseID: block.ToolUseID, Name: block.Name,
					Input: block.Input, Content: block.Content,
					IsError: block.IsError, Thinking: block.Thinking,
					Signature: block.Signature, CacheControl: block.CacheControl,
					Raw: block.Raw,
				})
			}
		}

		// Determine stop reason.
		switch nr.StopReason {
		case "end_turn":
			resp.StopReason = "end_turn"
		case "max_tokens":
			resp.StopReason = "max_tokens"
		case "tool_use":
			resp.StopReason = "tool_use"
		default:
			resp.StopReason = "end_turn"
		}
	}

	return resp
}
