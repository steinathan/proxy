package server

import (
	"crypto/subtle"
	"net/http"
	"os"
)

// sharedSecretMiddleware enforces the edge→engine trust boundary. In the
// single-machine topology the Go engine only ever saw loopback traffic; now
// the Next.js edge lives on Vercel and forwards to this proxy over the public
// internet, so an unauthenticated fly.dev URL would burn upstream keys and
// let anyone spoof X-User-ID. When ROUTATIC_PROXY_SHARED_SECRET is set, every
// request except /health and /statusline must carry X-Rec-Secret matching it.
// Unset env = no enforcement, so local dev against 127.0.0.1 is unchanged.
func sharedSecretMiddleware(next http.Handler) http.Handler {
	secret := os.Getenv("ROUTATIC_PROXY_SHARED_SECRET")
	if secret == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health", "/statusline":
			next.ServeHTTP(w, r)
			return
		}
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Rec-Secret")), []byte(secret)) != 1 {
			http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
