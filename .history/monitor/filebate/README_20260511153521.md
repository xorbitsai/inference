# Filebeat 日志采集配置

采集 xinference worker/supervisor 节点日志，支持输出到 Elasticsearch 或 Kafka。

## 文件说明

| 文件 | 说明 |
|------|------|
| `filebeat-to-es.yml` | 输出到 Elasticsearch，适合直接查询+Kibana可视化 |
| `filebeat-to-kafka.yml` | 输出到 Kafka，适合高吞吐、需二次消费的场景 |
| `docker-compose.yml` | Docker 部署示例 |

## 快速使用

```bash
# 1. 修改配置中的日志路径和目标地址
# 2. 启动
docker compose up -d

# 切换到 Kafka 输出: 修改 docker-compose.yml 中的 volumes 挂载
```

## 日志格式

```
2026-05-08 00:00:01,678 xinference.core.worker 1199841 DEBUG    Enter get_model, args: ...
```

解析后字段:
- `xinference.timestamp` — 时间戳
- `xinference.module` — 模块名
- `xinference.pid` — 进程ID
- `xinference.level` — 日志级别
- `xinference.request_id` — 请求ID (部分日志)
- `xinference.node` — 节点名 (从文件名提取)
- `log_type` — worker / supervisor

## 注意事项

- Worker 日志量大(~100MB/天)，生产环境建议过滤 DEBUG 或调高 xinference 日志级别
- 多行合并已配置，可处理进度条和 Python warning
- 日志路径需改为实际服务器路径
