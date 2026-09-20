#!/bin/bash
# Initialize Elasticsearch templates, lifecycle policies, and read-only search alias.
# Usage: bash es-init.sh <ES_HOST>
# Example: bash es-init.sh http://elasticsearch:9200

set -euo pipefail

ES_HOST="${1:-http://elasticsearch:9200}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

put_json() {
  local url="$1"
  local file="$2"
  curl -fsS -X PUT "${ES_HOST}${url}" \
    -H "Content-Type: application/json" \
    -d @"${SCRIPT_DIR}/${file}" | python3 -m json.tool
}

echo "==> Creating ILM policy: xinference-logs-policy (retain 30 days)"
put_json "/_ilm/policy/xinference-logs-policy" "es-ilm-policy.json"

echo ""
echo "==> Creating ILM policy: xinference-model-request-policy (retain 7 days)"
put_json "/_ilm/policy/xinference-model-request-policy" "es-model-request-ilm-policy.json"

echo ""
echo "==> Creating index template: xinference-logs"
put_json "/_index_template/xinference-logs" "es-index-template.json"

echo ""
echo "==> Creating index template: xinference-model-request"
put_json "/_index_template/xinference-model-request" "es-model-request-index-template.json"

echo ""
echo "==> Creating index template: xinference-audit"
put_json "/_index_template/xinference-audit" "es-audit-index-template.json"

echo ""
echo "==> Adding existing application/request indices to read-only alias: xinference-log-search"
# Templates add the alias to future indices. Resolve existing indices first so
# an empty wildcard remains a no-op, then use the atomic aliases API. The PUT
# alias API does not accept allow_no_indices or ignore_unavailable.
alias_payload="$(
  curl -fsS \
    "${ES_HOST}/xinference-logs-*,xinference-model-request-*?allow_no_indices=true&ignore_unavailable=true&filter_path=*.settings.index.provided_name" | \
    python3 -c 'import json, sys; data = json.load(sys.stdin); json.dump({"actions": [{"add": {"index": name, "alias": "xinference-log-search"}} for name in sorted(data)]}, sys.stdout)'
)"
alias_action_count="$(
  printf '%s' "${alias_payload}" | \
    python3 -c 'import json, sys; print(len(json.load(sys.stdin)["actions"]))'
)"
if [[ "${alias_action_count}" -gt 0 ]]; then
  curl -fsS -X POST "${ES_HOST}/_aliases" \
    -H "Content-Type: application/json" \
    -d "${alias_payload}" | python3 -m json.tool
else
  echo "No existing application/request indices found; template aliases will apply to future indices."
fi

echo ""
echo "==> Done"
