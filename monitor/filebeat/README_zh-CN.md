# Filebeat 日志采集配置

本目录用于采集 Xinference 应用日志、审计日志、Token Router 日志以及模型请求日志，
支持直接写入 Elasticsearch 或输出到 Kafka。

## 文件说明

| 文件 | 说明 |
|---|---|
| `filebeat-to-es.yml` | 文本应用日志写入 Elasticsearch |
| `filebeat-to-kafka.yml` | 文本应用日志输出到 Kafka |
| `filebeat-json-to-es.yml` | JSON 应用日志写入 Elasticsearch |
| `filebeat-json-to-kafka.yml` | JSON 应用日志输出到 Kafka |
| `filebeat-text-to-es.yml` | 另一套文本日志布局写入 Elasticsearch |
| `filebeat-text-to-kafka.yml` | 另一套文本日志布局输出到 Kafka |
| `es-index-template.json` | 应用日志 mapping 和统一查询 alias |
| `es-model-request-index-template.json` | 模型请求日志 mapping 和统一查询 alias |
| `es-model-request-ilm-policy.json` | 模型请求日志 7 天保留策略 |
| `es-audit-index-template.json` | 审计日志 mapping；审计日志不加入统一 alias |
| `es-init.sh` | 初始化 ILM、template，并为历史索引补充查询 alias |
| `docker-compose.yml` | Docker 部署示例 |

## 模型请求日志

`model_request.log` 是 NDJSON，六份配置都使用独立的 `filestream` input 采集。
Filebeat 只补充 `cluster=xinference` 和 `log_type=model_request`；`role`、
`address`、`node`、`module`、`pid`、`request_id`、`api_protocol` 由应用写入。

当示例默认值与生产环境不同时，应设置 Filebeat 容器内可见路径：

```env
XINFERENCE_MODEL_REQUEST_LOG_PATH=/data/logs/model_request.log*
```

仓库示例同时存在 `/data/logs` 和 `/data/xinference_home/logs` 两种目录约定，
部署前必须核对生产 bind mount。不能把 Filebeat 容器的 hostname 当成 Xinference
日志的 `node`。

请求日志 input 的单行上限为 32 MiB。Elasticsearch 直写模式下，应用的请求正文
上限可保持默认 16 MiB。正文保存在 Elasticsearch `_source` 中，但正文内部的任意
字段不会建立索引。

### Elasticsearch 路由

Elasticsearch 直写配置按以下规则路由：

- 应用日志：`xinference-logs-<log_type>-YYYY.MM.DD`；
- 模型请求日志：`xinference-model-request-YYYY.MM.DD`；
- 审计日志：`xinference-audit-YYYY.MM.DD`。

模型请求日志不能落入 `xinference-logs-model_request-*`。

应用日志和模型请求日志通过只读查询 alias 统一查询：

```text
xinference-log-search
```

Xinference API 应配置为查询该 alias：

```env
XINFERENCE_ES_INDEX=xinference-log-search
```

该 alias 只能作为查询目标，不能作为 Filebeat 写入目标；审计索引不会加入 alias。
alias 本身不是字段级安全边界：普通日志接口强制排除请求正文，独立正文接口同时校验
`logs:list` 和 `model_requests:read_body`。浏览器不能直接访问 Elasticsearch。

启用采集前初始化 Elasticsearch：

```bash
bash monitor/filebeat/es-init.sh http://elasticsearch:9200
```

脚本会安装应用日志和请求日志的 ILM/template，并把已有的应用日志、请求日志索引加入
`xinference-log-search`；后续新索引通过 template 自动加入。

### Kafka 输出

Kafka 配置沿用动态 topic：

```text
xinference-logs-<log_type>
```

因此模型请求会进入独立的 `xinference-logs-model_request` topic。下游消费者必须将该
topic 写入 `xinference-model-request-*`，不能写入
`xinference-logs-model_request-*`。

示例 Kafka producer 的消息上限约为 1 MiB，因此 Kafka 部署推荐设置：

```env
XINFERENCE_MODEL_REQUEST_LOG_BODY_MAX_BYTES=786432
```

约 768 KiB，可为 JSON 和 Kafka 协议开销预留空间。若必须保留更大的正文，需要同步
调整 Filebeat producer、broker、topic、replica fetch、consumer fetch 以及
Logstash/Flink 等消费端，不能只提高 Filebeat 上限。

## 快速使用

```bash
# 1. 核对容器内日志路径和目标地址。
# 2. 直写 Elasticsearch 时先初始化模板。
bash monitor/filebeat/es-init.sh http://elasticsearch:9200
# 3. 启动 Filebeat。
docker compose up -d
```

切换到 Kafka 输出时，修改 `docker-compose.yml` 中挂载的 Filebeat 配置文件。

## 应用日志格式

典型文本日志：

```text
2026-05-08 00:00:01,678 xinference.core.worker 1199841 DEBUG    Enter get_model, args: ...
```

解析字段包括 `@timestamp`、`module`、`pid`、`level`、`request_id`、`role`、
`address`、`node`、`log_type`。Worker 日志量较大，生产环境可考虑过滤 DEBUG 或
提高日志级别。

## Token Router 日志

六份示例配置均将 Token Router 日志作为独立 JSON 数据源采集：

- `/data/inference/logs/router-agent/router.log*`，`log_type=router_agent`；
- `/data/inference/logs/router-runtime/*/router.log*`，`log_type=router_runtime`。

压缩归档默认排除，避免重复采集。请保持 Token Router 使用 JSON 日志
（`XINFERENCE_LOG_FORMAT=json`），以便身份字段可被结构化查询。
