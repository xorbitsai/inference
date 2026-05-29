# PR-F #4953 审查修复方案（第五轮）

## 审查来源

- GitHub Code Review Agent (gemini-code-assist[bot])，2026-05-27
- 维护者 qinxuye 手动审查，2026-05-28 ~ 2026-05-29

---

## 问题列表

### 1. PromQL 正则 bug —— 中文 Dashboard（High）

**文件**：`monitor/dashboard/xinference-grafana-dashboard-zh-CN.json`

**问题**：`mountpoint !~ "/boot|"` 尾部 `|` 匹配空字符串，过滤掉所有磁盘指标。

**修复**：所有出现位置改为 `mountpoint !~ "/boot"`。

---

### 2. ES 索引模板 dynamic strict 导致文档丢失（High）

**文件**：`monitor/filebeat/es-index-template.json`

**问题**：`"dynamic": "strict"` 拒绝含未定义字段的文档，审计日志和新增字段的文档将被丢弃。

**修复**：改为 `"dynamic": false`。

---

### 3. Filebeat ILM 覆盖自定义索引路由（High）

**文件**：
- `monitor/filebeat/filebeat-json-to-es.yml`
- `monitor/filebeat/filebeat-text-to-es.yml`

**问题**：`setup.ilm.enabled: true` 使 Filebeat 忽略 `indices` 路由，所有事件混入单一 alias。

**修复**：两个文件中 `setup.ilm.enabled: true` 改为 `setup.ilm.enabled: false`。

---

### 4. es-init.sh curl 错误处理不足（Medium → 维护者二次要求）

**文件**：`monitor/filebeat/es-init.sh`

**问题**：
- 第一轮已加 `set -euo pipefail`，但维护者指出 `curl -s` 对 HTTP 4xx/5xx 仍返回 0，`python3 -m json.tool` 只要 body 是合法 JSON 也返回 0。
- 需要用 `curl -fsS` 或显式检查响应中 `acknowledged` / `error` 字段。

**修复**：使用 `curl -fsS`（HTTP 错误时 curl 自身失败），配合 `set -euo pipefail` 即可中断脚本。

---

### 5. es-init.sh 缺少审计索引模板安装（维护者要求）

**文件**：`monitor/filebeat/es-init.sh`

**问题**：脚本只装了 `xinference-logs` 模板，缺少 `xinference-audit` 模板。

**修复**：添加 `PUT /_index_template/xinference-audit`，引用 `es-audit-index-template.json`。

---

### 6. 共享脚本/配置中文改英文（维护者要求）

**文件**：
- `monitor/filebeat/es-init.sh` — 中文注释和 echo 输出
- `monitor/filebeat/filebeat-json-to-es.yml` — 中文 YAML 注释
- `monitor/filebeat/filebeat-text-to-es.yml` — 中文 YAML 注释
- `monitor/filebeat/es-index-template.json` — `_meta.description` 字段
- `monitor/filebeat/es-audit-index-template.json` — `_meta.description` 字段

**问题**：维护者明确要求非本地化的共享脚本和配置使用英文。`README_zh-CN.md` 和 zh-CN dashboard 可保留中文。

**修复**：将上述文件中所有中文注释、echo 输出、`_meta.description` 翻译为英文。

---

### 7. 英文 Dashboard config_info 查询类型错误（维护者要求）

**文件**：`monitor/dashboard/xinference-grafana-dashboard-en.json`

**问题**：`config_info` 从 instant 改成 range query，静态元数据用 range 会导致表格行重复。

**修复**：所有 `config_info` 相关 target 恢复为 `"instant": true` / `"range": false`。同时检查中文和日文 dashboard 是否有相同问题并一并修复。

---

## 修复文件清单

| 文件 | 修复内容 |
|---|---|
| `monitor/dashboard/xinference-grafana-dashboard-zh-CN.json` | PromQL `"/boot\|"` → `"/boot"` |
| `monitor/dashboard/xinference-grafana-dashboard-en.json` | config_info target 恢复 instant 查询 |
| `monitor/dashboard/xinference-grafana-dashboard-zh-CN.json` | config_info target 检查并修复（如有） |
| `monitor/dashboard/xinference-grafana-dashboard-ja.json` | config_info target 检查并修复（如有） |
| `monitor/filebeat/es-index-template.json` | `"dynamic": "strict"` → `"dynamic": false`；`_meta.description` 改英文 |
| `monitor/filebeat/es-audit-index-template.json` | `_meta.description` 改英文 |
| `monitor/filebeat/filebeat-json-to-es.yml` | `setup.ilm.enabled: false`；注释改英文 |
| `monitor/filebeat/filebeat-text-to-es.yml` | `setup.ilm.enabled: false`；注释改英文 |
| `monitor/filebeat/es-init.sh` | `set -euo pipefail` + `curl -fsS`；添加审计模板安装；注释和输出改英文 |

## 执行计划

1. 切换到 `feat/monitor-dashboard` 分支
2. 逐项修复上述 7 个问题
3. 运行 `pre-commit` 验证格式
4. 新增 commit：`fix(monitor): address review feedback — PromQL regex, ES dynamic mode, ILM routing, curl error handling, audit template, i18n`
5. Push 到远端，PR #4953 自动更新
