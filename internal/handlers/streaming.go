package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/routatic/proxy/internal/core"
	"github.com/routatic/proxy/internal/transformer"
)

// StreamProxy handles SSE stream forwarding from various upstream wire formats
// to Anthropic-format SSE events. It wraps transformer.StreamHandler and
// dispatches by WireFormat.
type StreamProxy struct {
	handler *transformer.StreamHandler
}

// NewStreamProxy creates a new StreamProxy.
func NewStreamProxy() *StreamProxy {
	return &StreamProxy{
		handler: transformer.NewStreamHandler(),
	}
}

// ProxyStream proxies an upstream SSE stream to the response writer, transforming
// events from the wire format to Anthropic SSE events.
func (sp *StreamProxy) ProxyStream(
	w http.ResponseWriter,
	body io.ReadCloser,
	wireFormat core.WireFormat,
	modelID string,
	clientCtx context.Context,
	idleTimeout time.Duration,
	cancel context.CancelFunc,
) error {
	switch wireFormat {
	case core.WireFormatAnthropic:
		return sp.proxyAnthropicPassthroughStream(w, body, modelID, idleTimeout, clientCtx, cancel)
	case core.WireFormatOpenAIResponses:
		return sp.handler.ProxyResponsesStream(w, body, modelID, clientCtx, idleTimeout, cancel)
	case core.WireFormatGemini:
		return sp.handler.ProxyGeminiStream(w, body, modelID, clientCtx, idleTimeout, cancel)
	default:
		return sp.proxyOpenAIStream(w, body, modelID, clientCtx, idleTimeout, cancel)
	}
}

// proxyOpenAIStream delegates to the transformer's ProxyStream.
func (sp *StreamProxy) proxyOpenAIStream(
	w http.ResponseWriter,
	body io.ReadCloser,
	modelID string,
	clientCtx context.Context,
	idleTimeout time.Duration,
	cancel context.CancelFunc,
) error {
	return sp.handler.ProxyStream(w, body, modelID, clientCtx, idleTimeout, cancel)
}

// proxyAnthropicPassthroughStream forwards raw Anthropic SSE bytes directly to
// the client, with an idle watchdog. If requestedModel is non-empty, the first
// `message_start` event's `model` field is rewritten to it so Claude Code
// records the user's requested model name on resume rather than the upstream's.
// After the message_start event is rewritten (or skipped if no model change is
// needed), the rest of the stream is forwarded verbatim.
func (sp *StreamProxy) proxyAnthropicPassthroughStream(
	w http.ResponseWriter,
	body io.ReadCloser,
	requestedModel string,
	idleTimeout time.Duration,
	clientCtx context.Context,
	cancel context.CancelFunc,
) error {
	defer func() { _ = body.Close() }()
	defer cancel()

	const sseBoundary = "\n\n"
	const dataPrefix = "data: "
	const startEvent = "event: message_start"

	buf := make([]byte, 4096)
	pending := make([]byte, 0, 8192)
	rewritten := false
	ping := transformer.StartIdleWatchdog(clientCtx, cancel, idleTimeout)
	for {
		select {
		case <-clientCtx.Done():
			if clientCtx.Err() == nil {
				return transformer.ErrStreamIdle
			}
			return transformer.ErrClientDisconnected
		default:
		}
		n, rerr := body.Read(buf)
		if n > 0 {
			ping()
			pending = append(pending, buf[:n]...)
			if !rewritten && requestedModel != "" {
				if idx := bytes.Index(pending, []byte(sseBoundary)); idx >= 0 {
					frame := pending[:idx]
					pending = pending[idx+len(sseBoundary):]
					rewritten = true
					// Match either the leading-event form (first frame) or the
					// mid-stream form (preceded by a newline from a prior frame).
					if bytes.HasPrefix(frame, []byte(startEvent+"\n")) ||
						bytes.Contains(frame, []byte("\n"+startEvent+"\n")) {
						fixed := rewriteMessageStartModel(frame, requestedModel)
						if _, werr := w.Write(fixed); werr != nil {
							return transformer.ErrClientDisconnected
						}
					} else {
						if _, werr := w.Write(frame); werr != nil {
							return transformer.ErrClientDisconnected
						}
					}
					if _, werr := w.Write([]byte(sseBoundary)); werr != nil {
						return transformer.ErrClientDisconnected
					}
					if f, ok := w.(http.Flusher); ok {
						f.Flush()
					}
					// Fall through and flush any pre-buffered bytes below.
				} else {
					// Wait for more bytes to complete the first frame; cap so
					// the upstream can't starve us by emitting very long preambles.
					if len(pending) > 65536 {
						if _, werr := w.Write(pending); werr != nil {
							return transformer.ErrClientDisconnected
						}
						pending = pending[:0]
					}
					if rerr != nil && rerr != io.EOF {
						// Stream error before we hit the first frame — drop the
						// buffer rather than forward a half-frame; the read loop
						// below surfaces the real error.
						pending = pending[:0]
					}
				}
			}
			if rewritten {
				if len(pending) > 0 {
					if _, werr := w.Write(pending); werr != nil {
						return transformer.ErrClientDisconnected
					}
					pending = pending[:0]
				}
			}
			if rewritten {
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
		}
		if rerr == io.EOF {
			if len(pending) > 0 {
				_, _ = w.Write(pending)
			}
			return nil
		}
		if rerr != nil {
			if transformer.IsIdleTimeout(rerr) {
				return transformer.ErrStreamIdle
			}
			if errors.Is(rerr, context.Canceled) || errors.Is(rerr, transformer.ErrStreamReadCanceled) || clientCtx.Err() == context.Canceled {
				if clientCtx.Err() == nil {
					return transformer.ErrStreamIdle
				}
				return transformer.ErrClientDisconnected
			}
			return fmt.Errorf("failed to copy response: %w", rerr)
		}
	}
}

// rewriteMessageStartModel scans a single SSE frame for the message_start
// data line and replaces the nested `message.model` value with requestedModel.
// Non-message_start frames and frames whose model already matches are passed
// through unchanged. The frame is expected to be one SSE event's worth — no
// trailing `\n\n` boundary.
func rewriteMessageStartModel(frame []byte, requestedModel string) []byte {
	const eventLine = "event: message_start"
	const dataLinePrefix = "data: "

	// Locate the message_start event header. Allow it at frame start (the
	// first SSE event) or preceded by a newline (mid-stream frames).
	var idx int
	if bytes.HasPrefix(frame, []byte(eventLine)) {
		idx = 0
	} else {
		rel := bytes.Index(frame, []byte("\n"+eventLine))
		if rel < 0 {
			return frame
		}
		idx = rel + 1
	}
	// Skip past the event header line and its trailing newline.
	after := idx + len(eventLine)
	for after < len(frame) && frame[after] == '\n' {
		after++
	}
	// Find the start of the data line.
	dataRel := bytes.Index(frame[after:], []byte(dataLinePrefix))
	if dataRel < 0 {
		return frame
	}
	dataStart := after + dataRel + len(dataLinePrefix)
	dataEnd := dataStart
	for dataEnd < len(frame) && frame[dataEnd] != '\n' {
		dataEnd++
	}
	var parsed struct {
		Type    string `json:"type"`
		Message struct {
			Model string `json:"model"`
		} `json:"message"`
	}
	if err := json.Unmarshal(frame[dataStart:dataEnd], &parsed); err != nil {
		return frame
	}
	if parsed.Type != "message_start" {
		return frame
	}
	if parsed.Message.Model == requestedModel {
		return frame
	}
	rewrittenData, err := messagesRemarshalDataLine(frame[dataStart:dataEnd], requestedModel)
	if err != nil {
		return frame
	}
	out := make([]byte, 0, len(frame)+len(requestedModel))
	out = append(out, frame[:dataStart]...)
	out = append(out, rewrittenData...)
	out = append(out, frame[dataEnd:]...)
	return out
}

func messagesRemarshalDataLine(data []byte, requestedModel string) ([]byte, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	msgRaw, ok := raw["message"]
	if !ok {
		return nil, errors.New("no message field")
	}
	var msg map[string]json.RawMessage
	if err := json.Unmarshal(msgRaw, &msg); err != nil {
		return nil, err
	}
	encodedModel, err := json.Marshal(requestedModel)
	if err != nil {
		return nil, err
	}
	msg["model"] = encodedModel
	newMsgRaw, err := json.Marshal(msg)
	if err != nil {
		return nil, err
	}
	raw["message"] = newMsgRaw
	return json.Marshal(raw)
}
