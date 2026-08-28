package transformer

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/pkg/types"
)

// TransformToResponses converts an Anthropic MessageRequest to OpenAI ResponsesRequest.
func (t *RequestTransformer) TransformToResponses(
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
) (*types.ResponsesRequest, error) {
	var input []types.ResponsesInput

	// Add system message if present
	systemText := anthropicReq.SystemText()
	if systemText != "" {
		content, _ := json.Marshal(systemText)
		input = append(input, types.ResponsesInput{
			Role:    "developer",
			Content: content,
		})
	}

	// Transform messages
	for _, msg := range anthropicReq.Messages {
		blocks := msg.ContentBlocks()
		var textParts []string

		for _, block := range blocks {
			switch block.Type {
			case "text":
				textParts = append(textParts, block.Text)
			case "image":
				textParts = append(textParts, "[Image]")
			case "tool_use":
				textParts = append(textParts, fmt.Sprintf("[Tool: %s(%s)]", block.Name, string(block.Input)))
			case "tool_result":
				// Responses API does not support role:tool as separate input with current struct;
				// inline tool results as text to avoid `input[4] did not match any supported type`.
				// TODO: map to proper function_call_output with call_id when ResponsesInput supports it.
				// ponytail: inline as text, proper function_call_output if ResponsesInput gains Type/CallID
				toolContent := block.TextContent()
				if toolContent != "" {
					textParts = append(textParts, fmt.Sprintf("[Tool Result %s: %s]", block.ToolUseID, toolContent))
				}
			case "thinking":
				// Preserve thinking as text for Responses
				if block.Thinking != "" {
					textParts = append(textParts, block.Thinking)
				}
			}
		}

		if len(textParts) > 0 {
			var sb strings.Builder
			for _, p := range textParts {
				sb.WriteString(p)
			}
			text := sb.String()
			content, _ := json.Marshal(text)
			input = append(input, types.ResponsesInput{
				Role:    msg.Role,
				Content: content,
			})
		}
	}

	req := &types.ResponsesRequest{
		Model:  model.ModelID,
		Input:  input,
		Stream: anthropicReq.Stream != nil && *anthropicReq.Stream,
	}

	// Transform tools if present
	if len(anthropicReq.Tools) > 0 {
		req.Tools = t.transformToolsForResponses(anthropicReq.Tools)
	}

	// Add reasoning if model supports it
	if model.ReasoningEffort != "" {
		req.Reasoning = &types.ResponsesReasoning{
			Effort: model.ReasoningEffort,
		}
	}

	return req, nil
}

// transformToolsForResponses converts Anthropic tools to Responses tool format.
func (t *RequestTransformer) transformToolsForResponses(tools []types.Tool) []types.ResponsesTool {
	var result []types.ResponsesTool

	for _, tool := range tools {
		schema := tool.InputSchema
		if len(schema) == 0 {
			schema = []byte(`{"type":"object","properties":{}}`)
		}

		result = append(result, types.ResponsesTool{
			Type:        "function",
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  json.RawMessage(schema),
		})
	}

	return result
}
