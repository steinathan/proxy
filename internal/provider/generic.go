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
	"github.com/routatic/proxy/pkg/types"
)

// GenericProvider implements core.Provider for any OpenAI-compatible
// upstream endpoint. One provider instance per configured generic provider
// in config.json's `generic_providers` list (groq, xai, mistral, kiro, ...).
//
// Requests are translated to OpenAI Chat Completions using the same
// transformer that powers the OpenRouter provider — same wire format, same
// request/response mapping, same SSE streaming. No new transformer code
// required; the catalog already classifies these as OpenAI-chat endpoints.
type GenericProvider struct {
	baseProvider
	cfg config.GenericProviderConfig // value, not pointer — config is hot-reloaded
}

// NewGenericProvider constructs a GenericProvider for one configured entry.
func NewGenericProvider(atomic *config.AtomicConfig, cfg config.GenericProviderConfig) *GenericProvider {
	return &GenericProvider{
		baseProvider: newBaseProvider(atomic),
		cfg:          cfg,
	}
}

// Name returns the provider identifier (matches GenericProviderConfig.Name).
func (p *GenericProvider) Name() string { return p.cfg.Name }

// Capabilities — generic OpenAI-compatible providers support the same
// surface as OpenRouter. If a specific provider needs a different cap,
// add it as a GenericProviderConfig field.
func (p *GenericProvider) Capabilities() core.ProviderCapabilities {
	return core.ProviderCapabilities{
		SupportsStreaming:  true,
		SupportsTools:      true,
		SupportsThinking:   false,
		SupportsImageInput: true,
		MaxContextLength:   128_000,
		DefaultMaxTokens:   8192,
	}
}

func (p *GenericProvider) ModelCapabilities(_ string) (core.ProviderCapabilities, bool) {
	return p.Capabilities(), true
}

func (p *GenericProvider) WireFormat(_ config.ModelConfig) core.WireFormat {
	return core.WireFormatOpenAIChat
}

func (p *GenericProvider) RoundTripName(model config.ModelConfig) string {
	return model.ModelID
}

func (p *GenericProvider) StreamIdleTimeout(model config.ModelConfig) time.Duration {
	// Generic providers don't expose a per-provider stream timeout in v1;
	// fall back to the same default the other providers use.
	const fallback = 5 * time.Minute
	return fallback
}

// affinityFromContext reads the user ID (X-User-ID) that the messages
// handler stamps into the context after the multi-tenancy patch extracts
// it. Empty when running with the pre-patch binary or when no auth header
// was sent (the common single-user case).
func affinityFromContext(ctx context.Context) string {
	if v := ctx.Value(genericCtxKey{}); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// genericCtxKey is the unexported context-key type used to thread the
// user ID through to nextAPIKey without leaking the type across packages.
type genericCtxKey struct{}

// WithUserID returns a derived context carrying the user ID for
// sticky-key selection. The caller is expected to set this from the
// extracted X-User-ID header in internal/handlers/messages.go.
func WithUserID(ctx context.Context, userID string) context.Context {
	if userID == "" {
		return ctx
	}
	return context.WithValue(ctx, genericCtxKey{}, userID)
}

func (p *GenericProvider) Execute(ctx context.Context, req *core.NormalizedRequest, model config.ModelConfig) (*core.ExecuteResult, error) {
	cfg := p.atomic.Get()
	// Look up our config slice entry by name on every call — cheap, and
	// means hot-reloading config.json picks up new base_url/api_keys.
	var entry *config.GenericProviderConfig
	for i := range cfg.GenericProviders {
		if cfg.GenericProviders[i].Name == p.cfg.Name {
			entry = &cfg.GenericProviders[i]
			break
		}
	}
	if entry == nil || entry.BaseURL == "" {
		return nil, fmt.Errorf("generic provider %q not configured", p.cfg.Name)
	}
	apiKey := p.nextAPIKey(entry.EffectiveAPIKeys(), affinityFromContext(ctx))

	openaiReq := transformer.TransformRequestFromNormalized(req, model)
	streamFalse := false
	openaiReq.Stream = &streamFalse

	start := time.Now()
	resp, err := p.doRequest(ctx, entry.BaseURL, apiKey, openaiReq, false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var chatResp types.ChatCompletionResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	normResp := transformer.OpenAIResponseToNormalized(&chatResp, model.ModelID)
	anthropicResp := core.DenormalizeResponse(normResp)
	resultBody, err := json.Marshal(anthropicResp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal response: %w", err)
	}

	return &core.ExecuteResult{
		Body:    resultBody,
		ModelID: model.ModelID,
		Latency: time.Since(start),
	}, nil
}

func (p *GenericProvider) Stream(ctx context.Context, req *core.NormalizedRequest, model config.ModelConfig) (io.ReadCloser, error) {
	cfg := p.atomic.Get()
	var entry *config.GenericProviderConfig
	for i := range cfg.GenericProviders {
		if cfg.GenericProviders[i].Name == p.cfg.Name {
			entry = &cfg.GenericProviders[i]
			break
		}
	}
	if entry == nil || entry.BaseURL == "" {
		return nil, fmt.Errorf("generic provider %q not configured", p.cfg.Name)
	}
	apiKey := p.nextAPIKey(entry.EffectiveAPIKeys(), affinityFromContext(ctx))

	openaiReq := transformer.TransformRequestFromNormalized(req, model)
	streamTrue := true
	openaiReq.Stream = &streamTrue

	resp, err := p.doRequest(ctx, entry.BaseURL, apiKey, openaiReq, true)
	if err != nil {
		return nil, err
	}

	return resp.Body, nil
}

func (p *GenericProvider) doRequest(ctx context.Context, endpoint, apiKey string, req any, stream bool) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode >= http.StatusBadRequest {
		bodyBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, &client.APIError{StatusCode: resp.StatusCode, Body: string(bodyBytes)}
	}

	return resp, nil
}
