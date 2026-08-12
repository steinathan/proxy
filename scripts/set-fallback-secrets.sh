#!/usr/bin/env bash
# Apply Fly secrets for the reclaude-proxy fallback providers.
# Empty placeholders are safe — the engine falls through to the next provider
# when a key is unset, so you can set them one at a time.
#
# Usage:
#   1. Get a key from the listed provider site
#   2. Run:    ./scripts/set-fallback-secrets.sh <SECRET_NAME>=<value> [...]
#   3. Or:     eval "$(cat .env.secrets)" ./scripts/set-fallback-secrets.sh
#
# Signup links:
#   Tier 0 (no key): mimo_free, huggingchat
#   Tier 1 (free):   groq, cerebras, cloudflare, nvidia_nim, github_models, gemini_free
#   Tier 2 (cheap):  deepseek, glm, kimi, qwen, siliconflow, stepfun, doubao
#   Tier 3 (Aggregator credits): openrouter, together, fireworks, nebius, deepinfra, hyperbolic, sambanova
#   Tier 3b (kie.ai discount reseller): one KIE_API_KEY covers 13 chat models (Claude/GPT/Gemini/Grok)

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "No secrets supplied. Examples:"
  echo "  $0 ROUTATIC_PROXY_GROQ_API_KEY=gsk_... ROUTATIC_PROXY_CEREBRAS_API_KEY=csk-..."
  echo "Available secret names:"
  echo "  ROUTATIC_PROXY_GROQ_API_KEY            (console.groq.com)"
  echo "  ROUTATIC_PROXY_CEREBRAS_API_KEY        (cloud.cerebras.ai)"
  echo "  ROUTATIC_PROXY_CLOUDFLARE_API_KEY      (dash.cloudflare.com)"
  echo "  ROUTATIC_PROXY_CF_ACCOUNT_ID           (dash.cloudflare.com → AI)"
  echo "  ROUTATIC_PROXY_NVIDIA_API_KEY          (build.nvidia.com)"
  echo "  ROUTATIC_PROXY_GITHUB_MODELS_API_KEY   (github.com/settings/tokens)"
  echo "  ROUTATIC_PROXY_GEMINI_API_KEY          (aistudio.google.com)"
  echo "  ROUTATIC_PROXY_DEEPSEEK_API_KEY        (platform.deepseek.com)"
  echo "  ROUTATIC_PROXY_GLM_API_KEY             (bigmodel.cn / z.ai)"
  echo "  ROUTATIC_PROXY_KIMI_API_KEY            (platform.moonshot.cn)"
  echo "  ROUTATIC_PROXY_QWEN_API_KEY            (dashscope.aliyun.com)"
  echo "  ROUTATIC_PROXY_SILICONFLOW_API_KEY     (siliconflow.cn)"
  echo "  ROUTATIC_PROXY_STEPFUN_API_KEY         (platform.stepfun.com)"
  echo "  ROUTATIC_PROXY_DOUBAO_API_KEY          (volcengine.com)"
  echo "  ROUTATIC_PROXY_OPENROUTER_API_KEY      (openrouter.ai)"
  echo "  ROUTATIC_PROXY_TOGETHER_API_KEY        (api.together.xyz)"
  echo "  ROUTATIC_PROXY_FIREWORKS_API_KEY       (fireworks.ai)"
  echo "  ROUTATIC_PROXY_NEBIUS_API_KEY          (studio.nebius.ai)"
  echo "  ROUTATIC_PROXY_DEEPINFRA_API_KEY       (deepinfra.com)"
  echo "  ROUTATIC_PROXY_HYPERBOLIC_API_KEY      (app.hyperbolic.xyz)"
  echo "  ROUTATIC_PROXY_SAMBANOVA_API_KEY       (cloud.sambanova.ai)"
  echo "  ROUTATIC_PROXY_KIE_API_KEY             (kie.ai/api-key — covers 13 chat models)"
  exit 1
fi

cd "$(dirname "$0")/.."

echo "Setting $# secret(s) on reclaude-proxy..."
fly secrets set --app reclaude-proxy "$@"
echo "Done. The proxy reloads config automatically (hot_reload: true)."
