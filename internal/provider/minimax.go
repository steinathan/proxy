package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/routatic/proxy/internal/client"
	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/internal/core"
	"github.com/routatic/proxy/internal/transformer"
)

// MinimaxProvider implements core.Provider for MiniMax's Anthropic-compatible
// endpoint. All models speak native Anthropic wire format, so the stream
// relays to Claude Code untouched.
type MinimaxProvider struct {
	baseProvider
}

// NewMinimaxProvider creates a new MinimaxProvider.
func NewMinimaxProvider(atomic *config.AtomicConfig) *MinimaxProvider {
	return &MinimaxProvider{baseProvider: newBaseProvider(atomic)}
}

// Name returns the provider identifier.
func (p *MinimaxProvider) Name() string { return "minimax" }

// Capabilities returns provider-level capabilities.
func (p *MinimaxProvider) Capabilities() core.ProviderCapabilities {
	return core.ProviderCapabilities{
		SupportsStreaming:  true,
		SupportsTools:      true,
		SupportsThinking:   true,
		SupportsImageInput: true,
		MaxContextLength:   1_000_000,
		DefaultMaxTokens:   8192,
	}
}

// ModelCapabilities returns per-model capabilities.
func (p *MinimaxProvider) ModelCapabilities(modelID string) (core.ProviderCapabilities, bool) {
	return p.Capabilities(), true
}

// WireFormat returns the wire format; MiniMax is always Anthropic.
func (p *MinimaxProvider) WireFormat(model config.ModelConfig) core.WireFormat {
	return core.WireFormatAnthropic
}

// RoundTripName returns the model ID to use in the upstream request.
func (p *MinimaxProvider) RoundTripName(model config.ModelConfig) string {
	return model.ModelID
}

// StreamIdleTimeout returns the maximum gap between bytes on an active stream.
func (p *MinimaxProvider) StreamIdleTimeout(model config.ModelConfig) time.Duration {
	const fallback = 5 * time.Minute
	cfg := p.atomic.Get()
	ms := cfg.Minimax.StreamTimeoutMs
	if ms <= 0 {
		ms = cfg.Minimax.TimeoutMs
	}
	if ms <= 0 {
		return fallback
	}
	return time.Duration(ms) * time.Millisecond
}

// Execute sends a non-streaming Anthropic request and returns the response.
func (p *MinimaxProvider) Execute(ctx context.Context, req *core.NormalizedRequest, model config.ModelConfig) (*core.ExecuteResult, error) {
	cfg := p.atomic.Get()
	endpoint := cfg.Minimax.BaseURL
	apiKey := p.nextAPIKey(cfg.Minimax.EffectiveAPIKeys(), "")

	anthropicReq := transformer.NormalizedToAnthropic(req, model)
	rawBody, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal anthropic request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(rawBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("x-api-key", apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	start := time.Now()
	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= http.StatusBadRequest {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, &client.APIError{StatusCode: resp.StatusCode, Body: string(bodyBytes)}
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return &core.ExecuteResult{
		Body:    body,
		ModelID: model.ModelID,
		Latency: time.Since(start),
	}, nil
}

// Stream sends a streaming Anthropic request and returns an io.ReadCloser for
// SSE events. The stream emits raw Anthropic SSE bytes, which the handler
// relays to Claude Code unchanged.
func (p *MinimaxProvider) Stream(ctx context.Context, req *core.NormalizedRequest, model config.ModelConfig) (io.ReadCloser, error) {
	cfg := p.atomic.Get()
	endpoint := cfg.Minimax.BaseURL
	apiKey := p.nextAPIKey(cfg.Minimax.EffectiveAPIKeys(), "")

	anthropicReq := transformer.NormalizedToAnthropic(req, model)
	rawBody, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal anthropic request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(rawBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("x-api-key", apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode >= http.StatusBadRequest {
		bodyBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, &client.APIError{StatusCode: resp.StatusCode, Body: string(bodyBytes)}
	}

	return resp.Body, nil
}
