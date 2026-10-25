# Filebeat Log Collection

The configurations in this directory collect Xinference application, audit,
Token Router, and model-request logs. They support direct Elasticsearch output
or Kafka output.

## Files

| File | Description |
|---|---|
| `filebeat-to-es.yml` | Text application logs to Elasticsearch |
| `filebeat-to-kafka.yml` | Text application logs to Kafka |
| `filebeat-json-to-es.yml` | JSON application logs to Elasticsearch |
| `filebeat-json-to-kafka.yml` | JSON application logs to Kafka |
| `filebeat-text-to-es.yml` | Alternate text layout to Elasticsearch |
| `filebeat-text-to-kafka.yml` | Alternate text layout to Kafka |
| `es-index-template.json` | Application-log mapping and search alias |
| `es-model-request-index-template.json` | Protected model-request mapping and search alias |
| `es-model-request-ilm-policy.json` | Seven-day model-request retention policy |
| `es-audit-index-template.json` | Audit-log mapping; audit logs are not in the shared alias |
| `es-init.sh` | Installs ILM policies/templates and backfills the search alias |
| `docker-compose.yml` | Docker deployment example |

## Model request logs

`model_request.log` is NDJSON and is collected by a dedicated `filestream`
input. Filebeat adds only `cluster=xinference` and
`log_type=model_request`; the application supplies `role`, `address`, `node`,
`module`, `pid`, `request_id`, and `api_protocol`.

Set the path visible **inside the Filebeat container** when it differs from the
example default:

```env
XINFERENCE_MODEL_REQUEST_LOG_PATH=/data/logs/model_request.log*
```

The repository examples use both `/data/logs` and `/data/xinference_home/logs`
layouts. Verify the production bind mount before deployment. Do not use the
Filebeat container hostname as the Xinference `node` value.

The dedicated input accepts a 32 MiB line. With direct Elasticsearch output,
the application request-body limit can remain at its default 16 MiB. The body
is retained in Elasticsearch `_source`, but its arbitrary inner fields are not
indexed.

### Elasticsearch routing

Direct Elasticsearch configurations route records as follows:

- application logs: `xinference-logs-<log_type>-YYYY.MM.DD`;
- model request logs: `xinference-model-request-YYYY.MM.DD`;
- audit logs: `xinference-audit-YYYY.MM.DD`.

Model request logs must not fall through to
`xinference-logs-model_request-*`.

Application and model-request indices share the read-only query alias:

```text
xinference-log-search
```

Configure the Xinference API to query the alias:

```env
XINFERENCE_ES_INDEX=xinference-log-search
```

The alias is a query target only; never configure it as a Filebeat write
target. Audit indices are intentionally excluded. Request bodies are not
protected by the alias itself: normal Xinference log APIs exclude body fields,
and the dedicated body endpoint enforces `logs:list` plus
`model_requests:read_body`. Browser clients must not connect directly to
Elasticsearch.

Initialize Elasticsearch before enabling ingestion:

```bash
bash monitor/filebeat/es-init.sh http://elasticsearch:9200
```

The script installs the application/request ILM policies and templates, then
adds existing application and request indices to `xinference-log-search`.
Future indices receive the alias from their templates.

### Kafka output

Kafka configurations use the existing dynamic topic pattern:

```text
xinference-logs-<log_type>
```

Therefore model requests go to the separate
`xinference-logs-model_request` topic. The downstream consumer must route that
topic to `xinference-model-request-*`, not to
`xinference-logs-model_request-*`.

The example Kafka producer limit is about 1 MiB. The recommended default for a
Kafka deployment is therefore:

```env
XINFERENCE_MODEL_REQUEST_LOG_BODY_MAX_BYTES=786432
```

This leaves room for JSON and Kafka protocol overhead. To retain larger bodies,
increase every part of the Kafka chain together: Filebeat producer, broker,
topic, replica fetch, consumer fetch, and any Logstash/Flink consumer. Do not
raise only Filebeat's limit.

## Quick start

```bash
# 1. Verify container-visible log paths and target addresses.
# 2. Install Elasticsearch templates when using direct ES output.
bash monitor/filebeat/es-init.sh http://elasticsearch:9200
# 3. Start Filebeat.
docker compose up -d
```

To switch to Kafka output, change the Filebeat config mount in
`docker-compose.yml`.

## Application log format

A text log entry is typically:

```text
2026-05-08 00:00:01,678 xinference.core.worker 1199841 DEBUG    Enter get_model, args: ...
```

Parsed fields include `@timestamp`, `module`, `pid`, `level`, `request_id`,
`role`, `address`, `node`, and `log_type`. Worker logs can be large; consider
filtering DEBUG records or raising the production log level.

## Token Router logs

All six example configurations collect Token Router logs as independent JSON
sources:

- `/data/inference/logs/router-agent/router.log*`, with `log_type=router_agent`;
- `/data/inference/logs/router-runtime/*/router.log*`, with
  `log_type=router_runtime`.

Compressed archives are excluded to avoid duplicate ingestion. Keep Token
Router JSON logging enabled (`XINFERENCE_LOG_FORMAT=json`) so identity fields
remain queryable.
