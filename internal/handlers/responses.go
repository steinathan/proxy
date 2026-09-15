// Package handlers contains HTTP request handlers for API endpoints.
package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/routatic/proxy/internal/client"
	"github.com/routatic/proxy/internal/config"
	"github.com/routatic/proxy/internal/core"
	"github.com/routatic/proxy/internal/debug"
	"github.com/routatic/proxy/internal/history"
	"github.com/routatic/proxy/internal/metrics"
	"github.com/routatic/proxy/internal/middleware"
	"github.com/routatic/proxy/internal/provider"
	"github.com/routatic/proxy/internal/router"
	"github.com/routatic/proxy/internal/token"
	"github.com/routatic/proxy/internal/transformer"
	"github.com/routatic/proxy/pkg/types"
)

// ResponsesHandler handles /v1/responses requests (OpenAI Responses API).
//
// It mirrors MessagesHandler's shape: validate → capture → normalize →
// route → dispatch. The wire format on the way in is OpenAI Responses;
// we convert to NormalizedRequest, run through the existing model-routing /
// fallback / provider-dispatch pipeline, then emit Responses wire format
// on the way out (byte-passthrough SSE for streaming, raw JSON for
// non-streaming — providers filtered to ResponsesWireFormat return bodies
// in the right shape already).
type ResponsesHandler struct {
	providerRegistry *core.ProviderRegistry
	modelRouter      *router.ModelRouter
	fallbackHandler  *router.FallbackHandler
	tokenCounter     *token.Counter
	logger           *slog.Logger
	rateLimiter      *middleware.RateLimiter
	requestIDGen     *middleware.RequestIDGenerator
	metrics          *metrics.Metrics
	captureLogger    *debug.CaptureLogger
	history          *history.History
	storage          StorageWriter
	atomic           *config.AtomicConfig

	// openCodeClient is only used for timeout lookups (legacy client kept for
	// compatibility with MessagesHandler's timeout helpers).
	openCodeClient *client.OpenCodeClient

	sessionCache sync.Map
}

// NewResponsesHandler creates a new Responses handler.
func NewResponsesHandler(
	openCodeClient *client.OpenCodeClient,
	providerRegistry *core.ProviderRegistry,
	modelRouter *router.ModelRouter,
	fallbackHandler *router.FallbackHandler,
	tokenCounter *token.Counter,
	metrics *metrics.Metrics,
	captureLogger *debug.CaptureLogger,
	hist *history.History,
	storage StorageWriter,
	atomic *config.AtomicConfig,
) *ResponsesHandler {
	return &ResponsesHandler{
		openCodeClient:   openCodeClient,
		providerRegistry: providerRegistry,
		modelRouter:      modelRouter,
		fallbackHandler:  fallbackHandler,
		tokenCounter:     tokenCounter,
		logger:           slog.Default(),
		rateLimiter:      middleware.NewRateLimiter(100, time.Minute),
		requestIDGen:     middleware.NewRequestIDGenerator(),
		metrics:          metrics,
		captureLogger:    captureLogger,
		history:          hist,
		storage:          storage,
		atomic:           atomic,
	}
}

// HandleResponses handles POST /v1/responses.
func (h *ResponsesHandler) HandleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeResponsesError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	requestID := r.Header.Get("X-Request-ID")
	if len(requestID) > 256 {
		requestID = requestID[:256]
	}
	if requestID == "" {
		requestID = h.requestIDGen.Generate()
	}
	w.Header().Set("X-Request-ID", requestID)

	userID := extractLoopbackUserID(h.atomic, r)
	r = r.WithContext(provider.WithUserID(r.Context(), userID))

	sessionID := strings.TrimSpace(r.Header.Get("X-Claude-Code-Session-Id"))
	if sessionID == "" {
		sessionID = strings.TrimSpace(r.Header.Get("X-Opencode-Session"))
	}
	if sessionID == "" {
		sessionID = h.stableSessionID(userID)
	}
	r = r.WithContext(provider.WithSessionID(r.Context(), sessionID))

	clientIP := middleware.GetClientIP(r)
	if !h.rateLimiter.Allow(clientIP) {
		h.metrics.RecordRateLimited()
		h.logger.Warn("rate limited", "client", clientIP, "request_id", requestID)
		writeResponsesError(w, http.StatusTooManyRequests, "rate limited", "rate_limit_error")
		return
	}

	const maxBodySize = 104857600 // 100 MB
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
	parseStart := time.Now()
	var rawBody json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&rawBody); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeResponsesError(w, http.StatusRequestEntityTooLarge, "request body too large", "invalid_request_error")
			return
		}
		writeResponsesError(w, http.StatusBadRequest, "invalid request body", "invalid_request_error")
		return
	}

	if h.captureLogger != nil {
		h.captureLogger.CaptureOriginal(requestID, rawBody)
	}

	var req types.ResponsesRequest
	if err := json.Unmarshal(rawBody, &req); err != nil {
		writeResponsesError(w, http.StatusBadRequest, "invalid request body: "+err.Error(), "invalid_request_error")
		return
	}
	h.metrics.RecordStage(metrics.StageRequestParse, time.Since(parseStart))

	if err := validateResponsesRequest(&req); err != nil {
		writeResponsesError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	isStreaming := req.Stream
	h.metrics.RecordRequest(isStreaming)

	h.logger.Info("received responses request",
		"model", req.Model,
		"streaming", isStreaming,
		"input_items", len(req.Input),
		"tools", len(req.Tools),
	)

	routerMessages, tokenMessages := responsesRouterMessages(&req)
	needsTools := len(req.Tools) > 0
	needsVision := false // image input is deferred in MVP

	tokenStart := time.Now()
	tokenCount, err := h.tokenCounter.CountMessages(req.Instructions, tokenMessages)
	if err != nil {
		h.logger.Warn("failed to count tokens", "error", err)
		tokenCount = 0
	}
	h.metrics.RecordStage(metrics.StageTokenCount, time.Since(tokenStart))

	routeStart := time.Now()
	modelChain, routeResult, err := h.buildModelChain(req.Model, routerMessages, tokenCount, isStreaming, req.MaxOutputTokens, needsVision, needsTools)
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, router.ErrUnknownProvider) {
			status = http.StatusBadRequest
		}
		writeResponsesError(w, status, err.Error(), "invalid_request_error")
		return
	}
	h.metrics.RecordStage(metrics.StageRouting, time.Since(routeStart))

	h.logger.Info("routing responses request",
		"scenario", routeResult.Scenario,
		"model", routeResult.Primary.ModelID,
		"provider", routeResult.Primary.Provider,
		"tokens", tokenCount,
	)

	normalizeStart := time.Now()
	normalizedReq := core.NormalizeResponsesRequest(&req)
	h.metrics.RecordStage(metrics.StageNormalization, time.Since(normalizeStart))

	// The chain keeps Anthropic-format upstreams too — the streaming and
	// non-streaming dispatch translate Anthropic wire output back to Responses
	// on the way out (responses_translate.go). This lets minimax serve Codex
	// the same way it serves Claude Code.
	if len(modelChain) == 0 {
		writeResponsesError(w, http.StatusBadRequest, "no compatible upstream available", "invalid_request_error")
		return
	}

	if h.captureLogger != nil && len(modelChain) > 0 {
		data, _ := json.Marshal(normalizedReq)
		h.captureLogger.CaptureNormalized(requestID, modelChain[0].Provider, data)
	}

	if isStreaming {
		h.handleStreaming(w, r, normalizedReq, modelChain, routeResult.Scenario, requestID, userID, req.Model)
	} else {
		h.handleNonStreaming(w, r, normalizedReq, modelChain, routeResult.Scenario, requestID, userID, req.Model)
	}
}

// validateResponsesRequest rejects top-level fields that are deferred in
// MVP. Items inside Input[] are validated by the normalizer (silently dropped
// if unknown).
func validateResponsesRequest(req *types.ResponsesRequest) error {
	if req.Model == "" {
		return fmt.Errorf("model is required")
	}
	if req.PreviousResponseID != "" {
		return fmt.Errorf("previous_response_id is not supported")
	}
	if req.Truncation != "" && req.Truncation != "disabled" {
		return fmt.Errorf("truncation=%q is not supported", req.Truncation)
	}
	if req.Background != nil && *req.Background {
		return fmt.Errorf("background mode is not supported")
	}
	if req.Text != nil && len(req.Text.Format) > 0 {
		return fmt.Errorf("text.format is not supported")
	}
	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		switch req.Reasoning.Effort {
		case "none", "minimal", "low", "medium", "high", "xhigh", "max":
		case "adaptive", "auto":
			// Adaptive/auto: client delegates pick to upstream. OpenCode Go
			// supports neither; strip the field so the request goes through
			// without us forcing a value the upstream will 400 on.
			req.Reasoning = nil
		default:
			return fmt.Errorf("reasoning.effort %q is not supported", req.Reasoning.Effort)
		}
	}
	return nil
}

// responsesRouterMessages flattens ResponsesInput items into router-friendly
// text content for scenario detection. Tool calls/result placeholders keep
// the scenario detector informed that the conversation involves tools.
func responsesRouterMessages(req *types.ResponsesRequest) ([]router.MessageContent, []token.MessageContent) {
	var routerMessages []router.MessageContent
	var tokenMessages []token.MessageContent

	for _, item := range core.DecodeInputItems(req.Input) {
		switch item.Type {
		case "", "message":
			text := responsesInputText(item.Content)
			role := item.Role
			if role == "developer" {
				continue
			}
			if role == "" {
				role = "user"
			}
			routerMessages = append(routerMessages, router.MessageContent{
				Role:    role,
				Content: text,
			})
			tokenMessages = append(tokenMessages, token.MessageContent{
				Role:    role,
				Content: text,
			})
		case "function_call":
			routerMessages = append(routerMessages, router.MessageContent{
				Role:    "assistant",
				Content: fmt.Sprintf("[Tool Use: %s]", item.Name),
			})
		case "function_call_output":
			routerMessages = append(routerMessages, router.MessageContent{
				Role:    "user",
				Content: item.Output,
			})
		}
	}
	return routerMessages, tokenMessages
}

// responsesInputText decodes an OpenAI Responses `content` field. The wire
// format allows either a plain string ("hi") or an array of content parts
// ([{"type":"input_text","text":"..."}]); both forms return the joined text.
// Mirrors core.extractInputText but kept local so the core package remains
// the single source of truth for canonical normalization.
func responsesInputText(raw json.RawMessage) string {
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

func (h *ResponsesHandler) stableSessionID(key string) string {
	if v, ok := h.sessionCache.Load(key); ok {
		return v.(string)
	}
	id := uuid.NewString()
	actual, _ := h.sessionCache.LoadOrStore(key, id)
	return actual.(string)
}

// filterResponsesCompatible drops any model that the provider registry reports
// as a non-Responses wire format. We refuse to silently rewrite a Chat
// Completions or Anthropic upstream's body into Responses — that would lie to
// the client. If routing returned nothing compatible, the client sees a 400.
func (h *ResponsesHandler) filterResponsesCompatible(chain []config.ModelConfig) ([]config.ModelConfig, error) {
	if h.providerRegistry == nil {
		return chain, nil
	}
	out := make([]config.ModelConfig, 0, len(chain))
	for _, m := range chain {
		prov, ok := h.providerRegistry.Get(client.Provider(m))
		if !ok {
			continue
		}
		if prov.WireFormat(m) != core.WireFormatOpenAIResponses {
			h.logger.Info("dropping non-Responses model from responses chain",
				"model", m.ModelID, "provider", m.Provider,
				"wire_format", prov.WireFormat(m).String())
			continue
		}
		out = append(out, m)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no Responses-compatible upstream available for model %q", chain[0].ModelID)
	}
	return out, nil
}

// buildModelChain mirrors MessagesHandler.buildModelChain, sharing the same
// model router and scenario detector. Extracted rather than shared because
// the input shape (requestedModel + router messages) differs enough that
// duplicating ~30 lines is cheaper than parameterising the existing one.
func (h *ResponsesHandler) buildModelChain(
	requestedModel string,
	routerMessages []router.MessageContent,
	tokenCount int,
	isStreaming bool,
	requestedMaxTokens int,
	needsVision bool,
	needsTools bool,
) ([]config.ModelConfig, router.RouteResult, error) {
	var chain []config.ModelConfig
	var result router.RouteResult

	if requestedModel != "" {
		if overrideResult, ok := h.modelRouter.RouteWithOverride(requestedModel); ok {
			scenarioResult, err := h.routeOnce(routerMessages, tokenCount, "", isStreaming)
			if err != nil {
				return overrideResult.GetModelChain(), overrideResult, err
			}
			chain = appendUniqueModels(overrideResult.GetModelChain(), scenarioResult.GetModelChain())
			result = overrideResult
		} else if overrideResult, ok := h.modelRouter.RouteWithFamilyOverride(requestedModel); ok {
			scenarioResult, err := h.routeOnce(routerMessages, tokenCount, "", isStreaming)
			if err != nil {
				return overrideResult.GetModelChain(), overrideResult, err
			}
			chain = appendUniqueModels(overrideResult.GetModelChain(), scenarioResult.GetModelChain())
			result = overrideResult
		}
	}

	if chain == nil {
		var err error
		result, err = h.routeOnce(routerMessages, tokenCount, requestedModel, isStreaming)
		if err != nil {
			return nil, result, err
		}
		chain = result.GetModelChain()
	}

	decision, err := router.FilterByCapacity(chain, tokenCount, requestedMaxTokens, needsVision, needsTools)
	if err != nil {
		return nil, result, err
	}
	for _, s := range decision.Skipped {
		h.logger.Info("model skipped by capacity filter",
			"model", s.ModelID, "reason", s.Reason)
	}
	return decision.Models, result, nil
}

func (h *ResponsesHandler) routeOnce(
	routerMessages []router.MessageContent,
	tokenCount int,
	requestedModel string,
	isStreaming bool,
) (router.RouteResult, error) {
	if isStreaming && !h.modelRouter.IsStreamingScenarioRoutingEnabled() {
		return h.modelRouter.RouteForStreaming(routerMessages, tokenCount, requestedModel)
	}
	return h.modelRouter.Route(routerMessages, tokenCount, requestedModel)
}

// handleStreaming forwards a streaming /v1/responses request to the first
// Responses-capable model that produces a stream.
func (h *ResponsesHandler) handleStreaming(
	w http.ResponseWriter,
	r *http.Request,
	normalizedReq *core.NormalizedRequest,
	modelChain []config.ModelConfig,
	scenario router.Scenario,
	requestID string,
	userID string,
	requestedModel string,
) {
	clientCtx := r.Context()
	requestStart := time.Now()
	streamStart := time.Now()

	rw := &responseWriter{ResponseWriter: w}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	rw.WriteHeader(http.StatusOK)
	rw.Flush()

	var heartbeatPaused int32
	stopHeartbeat := startKeepaliveHeartbeat(clientCtx, rw, &heartbeatPaused, defaultKeepaliveInterval, keepaliveWriteTimeout, h.logger)
	defer stopHeartbeat()

	for _, model := range modelChain {
		select {
		case <-clientCtx.Done():
			h.logger.Debug("client disconnected, stopping responses fallbacks")
			return
		default:
		}

		h.logger.Info("attempting responses stream", "model", model.ModelID, "provider", model.Provider)
		timeout := h.openCodeClient.StreamingTimeout(model)
		attemptCtx, cancelAttempt := context.WithTimeout(clientCtx, timeout)
		idleTimeout := h.openCodeClient.StreamIdleTimeout(model)

		prov, ok := h.providerRegistry.Get(client.Provider(model))
		if !ok {
			cancelAttempt()
			h.logger.Warn("provider not in registry, skipping model", "model", model.ModelID)
			continue
		}
		caps, ok := prov.ModelCapabilities(model.ModelID)
		if !ok || !caps.SupportsStreaming {
			cancelAttempt()
			h.logger.Warn("model does not support streaming", "model", model.ModelID)
			continue
		}

		streamBody, err := prov.Stream(attemptCtx, normalizedReq, model)
		if err != nil {
			cancelAttempt()
			if clientCtx.Err() != nil {
				return
			}
			h.logger.Warn("streaming request failed via provider",
				"model", model.ModelID, "provider", model.Provider, "error", err)
			continue
		}
		streamReader := transformer.NewCtxReadCloser(attemptCtx, streamBody)

		atomic.StoreInt32(&heartbeatPaused, 1)
		// Dispatch on upstream wire format: Responses upstreams byte-passthrough;
		// Anthropic upstreams (e.g. minimax) translate SSE on the way out so the
		// Responses client (codex, ChatGPT.app) sees a valid Responses stream.
		var errProxy error
		switch prov.WireFormat(model) {
		case core.WireFormatAnthropic:
			errProxy = proxyAnthropicToResponsesStream(rw, streamReader, requestedModel, idleTimeout, attemptCtx, cancelAttempt, h.logger)
		default:
			errProxy = proxyResponsesPassthroughStream(rw, streamReader, requestedModel, idleTimeout, attemptCtx, cancelAttempt)
		}
		atomic.StoreInt32(&heartbeatPaused, 0)

		if errProxy != nil {
			if errProxy == transformer.ErrClientDisconnected {
				if clientCtx.Err() != nil {
					h.logger.Debug("client disconnected during responses stream")
					return
				}
				errProxy = fmt.Errorf("streaming timeout (%v) exceeded", timeout)
			}
			if errProxy == transformer.ErrStreamIdle {
				h.logger.Warn("upstream responses stream idle, trying next model",
					"model", model.ModelID, "idle_timeout", idleTimeout)
				if rw.ssePayloadWritten {
					h.sendStreamError(rw, "responses stream idle after SSE payload started")
					return
				}
				continue
			}
			h.logger.Warn("responses streaming failed", "model", model.ModelID, "error", errProxy)
			if rw.ssePayloadWritten {
				h.sendStreamError(rw, "all upstream models failed after SSE payload started")
				return
			}
			continue
		}

		latency := time.Since(streamStart)

		// Empty-response guard: some upstreams (e.g. muse-spark on the
		// Responses path) complete the stream with zero tokens and no SSE
		// payload. Treat that as a failure so the fallback chain tries the
		// next model instead of returning nothing to the client.
		if !rw.ssePayloadWritten && !rw.hasContent() {
			h.logger.Warn("responses stream returned no output, triggering fallback",
				"model", model.ModelID, "provider", model.Provider)
			if rw.wroteHeader {
				h.sendStreamError(rw, "upstream returned empty response")
			}
			continue
		}

		h.metrics.RecordSuccess(metrics.ModelKey(model.Provider, model.ModelID), latency)
		h.metrics.RecordStage(metrics.StageUpstream, latency)
		if firstContentAt := rw.firstContentTime(); !firstContentAt.IsZero() {
			h.metrics.RecordTTFT(firstContentAt.Sub(requestStart))
		}
		h.logger.Info("responses streaming completed",
			"model", model.ModelID,
			"latency", latency,
			"input_tokens", rw.usage.inputTokens,
			"output_tokens", rw.usage.outputTokens,
		)
		rec := history.RequestRecord{
			ID:           requestID,
			Model:        model.ModelID,
			Provider:     model.Provider,
			Scenario:     string(scenario),
			StartTime:    streamStart,
			Duration:     latency,
			InputTokens:  rw.usage.inputTokens,
			OutputTokens: rw.usage.outputTokens,
			Streaming:    true,
			Success:      true,
			Attempt:      1,
			UserID:       userID,
		}
		if h.history != nil {
			h.history.Add(rec)
		}
		if h.storage != nil {
			h.storage.RecordCompletion(rec)
		}
		return
	}

	h.metrics.RecordFailure()
	if rw.ssePayloadWritten {
		return
	}
	if !rw.wroteHeader {
		writeResponsesError(w, http.StatusBadGateway, "all upstream models failed", "api_error")
	} else {
		h.sendStreamError(rw, "all upstream models failed")
	}
}

// handleNonStreaming executes a non-streaming /v1/responses request via the
// first Responses-capable upstream that returns a body.
func (h *ResponsesHandler) handleNonStreaming(
	w http.ResponseWriter,
	r *http.Request,
	normalizedReq *core.NormalizedRequest,
	modelChain []config.ModelConfig,
	scenario router.Scenario,
	requestID string,
	userID string,
	requestedModel string,
) {
	ctx := r.Context()
	startTime := time.Now()
	upstreamStart := time.Now()

	result, responseBody, err := h.fallbackHandler.ExecuteWithFallback(
		ctx,
		modelChain,
		func(ctx context.Context, model config.ModelConfig) ([]byte, error) {
			timeout := h.openCodeClient.RequestTimeout(model)
			attemptCtx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()

			prov, ok := h.providerRegistry.Get(client.Provider(model))
			if !ok {
				return nil, fmt.Errorf("provider %q not registered", model.Provider)
			}
			execResult, execErr := prov.Execute(attemptCtx, normalizedReq, model)
			if execErr != nil {
				return nil, execErr
			}
			// Anthropic-format bodies need conversion to Responses JSON so the
			// Responses client can parse them. Other wire formats pass through.
			if prov.WireFormat(model) == core.WireFormatAnthropic {
				execResult.Body = anthropicBodyToResponses(execResult.Body, model.ModelID)
			}
			return execResult.Body, nil
		},
	)

	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			h.logger.Info("responses request context canceled during fallback", "error", err)
			return
		}
		if result != nil {
			h.metrics.RecordFailureForModel(metrics.ModelKey(modelChain[0].Provider, result.ModelID))
		}
		h.metrics.RecordFailure()
		writeResponsesError(w, http.StatusBadGateway, err.Error(), "api_error")
		return
	}
	h.metrics.RecordStage(metrics.StageUpstream, time.Since(upstreamStart))

	latency := time.Since(startTime)
	h.metrics.RecordSuccess(metrics.ModelKey(modelChain[0].Provider, result.ModelID), latency)

	h.logger.Info("responses request completed",
		"model", result.ModelID,
		"attempts", result.Attempted,
		"latency", latency,
	)

	var provider string
	for _, m := range modelChain {
		if m.ModelID == result.ModelID {
			provider = m.Provider
			break
		}
	}

	var inputTokens, outputTokens int
	var rr types.ResponsesResponse
	if errUnmarshal := json.Unmarshal(responseBody, &rr); errUnmarshal == nil {
		inputTokens = rr.Usage.InputTokens
		outputTokens = rr.Usage.OutputTokens
	}

	rec := history.RequestRecord{
		ID:           requestID,
		Model:        result.ModelID,
		Provider:     provider,
		Scenario:     string(scenario),
		StartTime:    startTime,
		Duration:     latency,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		Streaming:    false,
		Success:      true,
		Attempt:      result.Attempted,
		UserID:       userID,
	}
	if h.history != nil {
		h.history.Add(rec)
	}
	if h.storage != nil {
		h.storage.RecordCompletion(rec)
	}

	// Rewrite the top-level `model` to the client's requested model so Codex
	// can resume sessions keyed off the model it sent.
	out := rewriteResponsesBodyModel(responseBody, requestedModel)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
}

// rewriteResponsesBodyModel sets the top-level `model` field on a Responses
// JSON response body to requestedModel. Cheap JSON rewrite — no need to
// round-trip through types.
func rewriteResponsesBodyModel(body []byte, requestedModel string) []byte {
	if requestedModel == "" {
		return body
	}
	var resp map[string]json.RawMessage
	if err := json.Unmarshal(body, &resp); err != nil {
		return body
	}
	if _, ok := resp["model"]; !ok {
		return body
	}
	resp["model"] = json.RawMessage(quoteJSONString(requestedModel))
	out, err := json.Marshal(resp)
	if err != nil {
		return body
	}
	return out
}

// writeResponsesError writes an OpenAI-style error body. Codex parses this as
// {error:{message, type, code?}} — keep the shape stable.
func writeResponsesError(w http.ResponseWriter, status int, message, errType string) {
	body := map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errType,
			"code":    nil,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// sendStreamError emits an `event: error` SSE frame in OpenAI Responses
// shape mid-stream so Codex can render a meaningful message. Must only be
// called after SSE headers are flushed.
func (h *ResponsesHandler) sendStreamError(rw *responseWriter, message string) {
	payload := map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "api_error",
			"code":    nil,
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_, _ = rw.Write([]byte("event: error\ndata: "))
	_, _ = rw.Write(data)
	_, _ = rw.Write([]byte("\n\n"))
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
