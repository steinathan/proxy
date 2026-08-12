// Package provider implements the core.Provider interface for all supported
// upstream LLM providers.
package provider

import (
	"net/http"
	"sync/atomic"
	"time"

	"github.com/routatic/proxy/internal/config"
)

// baseProvider holds shared HTTP transport and key rotation used by all
// provider implementations in this package.
type baseProvider struct {
	atomic     *config.AtomicConfig
	httpClient *http.Client
	keyCounter atomic.Uint64
}

// newBaseProvider creates a baseProvider with a shared HTTP transport tuned
// for high-concurrency upstream calls.
func newBaseProvider(atomic *config.AtomicConfig) baseProvider {
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
		MaxConnsPerHost:     50,
		DisableKeepAlives:   false,
		Proxy:               http.ProxyFromEnvironment,
	}
	return baseProvider{
		atomic: atomic,
		httpClient: &http.Client{
			Transport: transport,
		},
	}
}

// nextAPIKey returns the next API key from the given pool.
//
// Selection rule:
//   - When affinity is empty (the common case for the upstream-only
//     providers that don't know about user identity), we round-robin.
//   - When affinity is set, we pin to a deterministic slot in the pool
//     derived from a stable hash of the affinity string. This is the
//     "sticky key" pattern: requests from the same user always land on
//     the same upstream key, which maximises Anthropic prompt-prefix
//     cache hits (cache key is per upstream API key + model + prefix).
//
// If the pinned slot is unhealthy (the caller observes failure, not us —
// this helper has no knowledge of circuit-breaker state), the caller should
// call nextAPIKey with empty affinity to round-robin through the pool.
func (b *baseProvider) nextAPIKey(keys []string, affinity string) string {
	if len(keys) == 0 {
		return ""
	}
	n := uint64(len(keys))
	if affinity != "" {
		return keys[fnvHash(affinity)%n]
	}
	old := b.keyCounter.Add(1)
	return keys[(old-1)%n]
}

// fnvHash is FNV-1a 64-bit. Tiny, stdlib-free, stable across runs and
// platforms. Good enough for key-affinity slot selection — we don't need
// crypto-grade distribution, we need cheap and stable.
func fnvHash(s string) uint64 {
	const (
		offset uint64 = 14695981039346656037
		prime  uint64 = 1099511628211
	)
	h := offset
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime
	}
	return h
}
