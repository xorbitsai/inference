# Xinference Alert Rules

Prometheus alert rules for monitoring Xinference cluster health.

## Files

| File | Language |
|------|----------|
| `rules.yml` | English |
| `rules-zh-CN.yml` | 中文 |
| `rules-ja.yml` | 日本語 |
| `rules-ko.yml` | 한국어 |

All files share the same alert expressions and thresholds — only `summary` and `description` annotations are localized.

## Alert Rules

### GPUMemoryHigh

- **Severity**: critical
- **Condition**: GPU memory usage > 90% for 5 minutes
- **Expression**: `xinference:worker_gpu_memory_used_bytes / xinference:worker_gpu_memory_total_bytes > 0.9`

### ModelRequestErrorRateHigh

- **Severity**: warning
- **Condition**: Model request error rate > 5% for 3 minutes
- **Expression**: `rate(xinference:model_request_errors_total[5m]) / rate(xinference:model_request_total[5m]) > 0.05`

### TTFTHigh

- **Severity**: warning
- **Condition**: Time to first token > 10 seconds for 3 minutes
- **Expression**: `rate(xinference:time_to_first_token_seconds_sum[5m]) / rate(xinference:time_to_first_token_seconds_count[5m]) > 10`

### WorkerOffline

- **Severity**: critical
- **Condition**: Online worker count < 1 for 2 minutes
- **Expression**: `xinference:workers_total < 1`

### DiskSpaceLow

- **Severity**: critical
- **Condition**: Disk available space < 10% for 5 minutes
- **Expression**: `(node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1`

### RequestQueueBacklog

- **Severity**: warning
- **Condition**: Pending requests > 10 for 3 minutes
- **Expression**: `xinference:model_pending_requests > 10`

### ModelLoadSlow

- **Severity**: warning
- **Condition**: Model load time > 5 minutes
- **Expression**: `xinference:model_last_load_duration_seconds > 300`

## Usage

Add the rule file to your Prometheus configuration:

```yaml
# prometheus.yml
rule_files:
  - "/path/to/monitor/alert/rules.yml"
```

Reload Prometheus to apply:

```bash
curl -X POST http://localhost:9090/-/reload
```

## Alertmanager Integration

Configure Alertmanager to route alerts by severity:

```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  receiver: default
  routes:
    - match:
        severity: critical
      receiver: critical-channel
    - match:
        severity: warning
      receiver: warning-channel
```
