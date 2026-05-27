#!/bin/bash
# 初始化 ES 索引模板和 ILM 策略
# 用法: bash es-init.sh <ES_HOST>
# 示例: bash es-init.sh http://elasticsearch:9200

ES_HOST="${1:-http://elasticsearch:9200}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> 创建 ILM 策略: xinference-logs-policy (保留30天)"
curl -s -X PUT "${ES_HOST}/_ilm/policy/xinference-logs-policy" \
  -H "Content-Type: application/json" \
  -d @"${SCRIPT_DIR}/es-ilm-policy.json" | python3 -m json.tool

echo ""
echo "==> 创建索引模板: xinference-logs"
curl -s -X PUT "${ES_HOST}/_index_template/xinference-logs" \
  -H "Content-Type: application/json" \
  -d @"${SCRIPT_DIR}/es-index-template.json" | python3 -m json.tool

echo ""
echo "==> 完成"
