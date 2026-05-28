# Filebeat Log Collection

Collects xinference worker/supervisor node logs, with output to Elasticsearch or Kafka.

## Files

| File | Description |
|------|-------------|
| `filebeat-to-es.yml` | Output to Elasticsearch, for direct querying + Kibana visualization |
| `filebeat-to-kafka.yml` | Output to Kafka, for high-throughput or downstream consumption |
| `docker-compose.yml` | Docker deployment example |

## Quick Start

```bash
# 1. Update log paths and target addresses in the config
# 2. Start
docker compose up -d

# To switch to Kafka output: change the volumes mount in docker-compose.yml
```

## Log Format

```
2026-05-08 00:00:01,678 xinference.core.worker 1199841 DEBUG    Enter get_model, args: ...
```

Parsed fields:
- `xinference.timestamp` — timestamp
- `xinference.module` — module name
- `xinference.pid` — process ID
- `xinference.level` — log level
- `xinference.request_id` — request ID (in some logs)
- `xinference.node` — node name (extracted from filename)
- `log_type` — worker / supervisor

## Notes

- Worker logs can be large (~100MB/day); in production, consider filtering DEBUG or raising the xinference log level
- Multiline merging is configured to handle progress bars and Python warnings
- Log paths must be updated to match actual server paths
