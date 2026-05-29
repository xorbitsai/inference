#!/bin/bash
# Initialize ES index templates and ILM policy
# Usage: bash es-init.sh <ES_HOST>
# Example: bash es-init.sh http://elasticsearch:9200

set -euo pipefail

ES_HOST="${1:-http://elasticsearch:9200}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Creating ILM policy: xinference-logs-policy (retain 30 days)"
curl -fsS -X PUT "${ES_HOST}/_ilm/policy/xinference-logs-policy" \
  -H "Content-Type: application/json" \
  -d @"${SCRIPT_DIR}/es-ilm-policy.json" | python3 -m json.tool

echo ""
echo "==> Creating index template: xinference-logs"
curl -fsS -X PUT "${ES_HOST}/_index_template/xinference-logs" \
  -H "Content-Type: application/json" \
  -d @"${SCRIPT_DIR}/es-index-template.json" | python3 -m json.tool

echo ""
echo "==> Creating index template: xinference-audit"
curl -fsS -X PUT "${ES_HOST}/_index_template/xinference-audit" \
  -H "Content-Type: application/json" \
  -d @"${SCRIPT_DIR}/es-audit-index-template.json" | python3 -m json.tool

echo ""
echo "==> Done"
