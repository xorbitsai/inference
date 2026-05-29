# PR-G #4954 审查修复方案（第四轮）

## 审查来源

- GitHub Code Review Agent (gemini-code-assist[bot])，2026-05-27
- 维护者 qinxuye 手动审查，2026-05-28 ~ 2026-05-29

---

## 问题列表

### 1. fetchWrapper 中 refreshed 条件判断简化（High - gemini）

**文件**：`xinference/ui/web/ui/src/components/fetchWrapper.js`

**问题**：`tryRefreshToken` 现在始终返回 boolean promise，条件判断应简化为直接 `if (refreshed)`，无需额外真值检查。

**修复**：将条件简化为 `if (refreshed)`。

---

### 2. 队列中 401 请求刷新后不会重试（High - 维护者 qinxuye）

**文件**：`xinference/ui/web/ui/src/components/fetchWrapper.js`

**问题**：`processQueue(true)` 调用 `prom.resolve()` 时未传值，队列中的请求收到 `undefined`。由于 `fetchWithRetry` 检查 `if (refreshed)`，`undefined` 为 falsy，导致只有第一个请求重试，其余队列请求继续原始 401 响应走错误流程。

**修复**：在 `processQueue` 中 resolve 时传递 success 参数，即 `prom.resolve(success)`，确保队列中所有请求都能根据刷新结果正确重试。

---

### 3. editingUser null 防护（Medium - gemini）

**文件**：`xinference/ui/web/ui/src/scenes/user_management/index.js`

**问题**：`handleSavePermissions` 被调用时 `editingUser` 可能为 null，访问 `editingUser.id` 会抛出 `TypeError`。

**修复**：在函数开头添加 `if (!editingUser) return` 守卫子句。

---

### 4. 审计中心菜单项权限不足的用户可见（维护者 qinxuye）

**文件**：`xinference/ui/web/ui/src/components/MenuSide.js:164`

**问题**：审计中心菜单项对所有 `authAdvanced` 用户显示，但后端 `/v1/audit/search` 路由注册的是 `admin` scope。非 admin 的 advanced-auth 用户看到菜单后点击立即 403。

**修复**：将审计中心菜单项的显示条件从 `authAdvanced` 改为同时检查用户是否具有 admin 角色（例如 `isAdmin && authAdvanced`）。

---

## 修复文件清单

| 文件 | 修复内容 |
|---|---|
| `xinference/ui/web/ui/src/components/fetchWrapper.js` | 简化 `refreshed` 条件 + `processQueue` resolve 传 boolean |
| `xinference/ui/web/ui/src/scenes/user_management/index.js` | 添加 `editingUser` null 防护 |
| `xinference/ui/web/ui/src/components/MenuSide.js` | 审计中心菜单项限制为 admin 用户可见 |

## 执行计划

1. 切换到 `feat/i18n-ui` 分支（PR #4954 所在分支）
2. 备份原始文件到 `dist/pr/pr-g-ui/pr-g-ui-04_py_bak/before/`
3. 逐项修复上述 4 个问题
4. 运行 `npx eslint` 检查
5. 备份修改后文件到 `dist/pr/pr-g-ui/pr-g-ui-04_py_bak/after/`
6. 新增 commit：`fix(ui): fix token refresh queue, null guard, and audit menu permission`
7. Push 到 myfork，PR #4954 自动更新
