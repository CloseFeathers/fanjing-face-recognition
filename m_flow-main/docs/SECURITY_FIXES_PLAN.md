# Code Scanning 安全修复计划 — 135 个 findings

## 总览

| 严重性 | 数量 | 可通过代码修复 | 需要 GitHub 设置 |
|--------|------|-------------|----------------|
| Critical | 1 | 1 | 0 |
| High | 15 | 10 | 5 |
| Medium | 118 | 118 | 0 |
| Low | 1 | 0 | 1 |
| **总计** | **135** | **129** | **6** |

---

## 阶段 1: Critical — 危险工作流 (1 个)

### DangerousWorkflow: `label-core-team.yml:34`

**问题**: 使用 `pull_request_target` 触发器。OpenSSF Scorecard 认为这允许 fork PR 在拥有仓库写权限的上下文中执行代码。

**实际风险分析**:

这个 workflow 的设计是**安全的**：
1. `checkout` 检出的是 **base branch**（第 36 行 `ref: ${{ github.event.pull_request.base.ref }}`），不是 fork 的代码
2. `run:` 步骤只读取 `.github/core-team.txt` 文件，不执行任何外部代码
3. `actions/github-script@v7` 只添加标签，不执行 PR 中的代码
4. `permissions` 已限制为 `contents: read` + `pull-requests: write`

**但 Scorecard 仍然标记它**，因为 `pull_request_target` + `checkout` 组合是一个已知的攻击向量模式。

**修复方案**: 保留当前实现（它是安全的），但添加注释解释安全性。或者，如果不需要给 fork PR 打标签，可以改为 `pull_request` 触发器（但这样就没有权限添加标签了）。

**推荐**: 不改代码。在 Scorecard 配置中添加忽略规则，或接受此 finding 作为 false positive。

---

## 阶段 2: High — Token Permissions (10 个)

### 问题

10 个 workflow 文件缺少顶层 `permissions` 声明，默认拥有 `GITHUB_TOKEN` 的全部权限（包括写权限）。

### 受影响的文件

| 文件 | 是否需要写权限 | 修复 |
|------|-------------|------|
| `backend_docker_build_test.yml` | 否 | 加 `permissions: {contents: read}` |
| `db_examples_tests.yml` | 否 | 同上 |
| `docker_compose.yml` | 否 | 同上 |
| `notebooks_tests.yml` | 否 | 同上 |
| `relational_db_migration_tests.yml` | 否 | 同上 |
| `vector_db_tests.yml` | 否 | 同上 |
| `weighted_edges_tests.yml` | 否 | 同上 |
| `release.yml` | 是 (contents: write) | 已有，Scorecard 误报 |
| `update-contributors.yml` | 是 (contents: write) | 已有，Scorecard 误报 |
| `dockerhub.yml` | 是 (packages: write) | 已有，Scorecard 误报 |

**实际需要修复的**: 7 个文件（纯只读 workflow 缺少 permissions）。

**修复方式**: 在每个文件顶部 `on:` 之后添加：
```yaml
permissions: {contents: read}
```

**安全性**: 零风险。只是限制了默认 token 权限，不影响功能。

---

## 阶段 3: Medium — Pinned Dependencies (116 个)

### 问题

GitHub Actions 和 Dockerfile 中的依赖使用标签（如 `@v4`）而非 SHA 哈希。标签可以被仓库所有者修改指向恶意代码。

### 涉及范围

| 类型 | 数量 | 示例 |
|------|------|------|
| `actions/checkout@v4` | ~70 | 所有 workflow |
| `astral-sh/setup-uv@v7` | ~15 | setup action |
| `docker/*@v3/v5/v6` | ~10 | Docker 相关 |
| `Dockerfile FROM` | ~10 | 基础镜像 |
| 其他 actions | ~11 | ruff, github-script 等 |

### 修复方式

将 `@v4` 替换为 `@<full-sha>`。例如：
```yaml
# 修改前
uses: actions/checkout@v4
# 修改后
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
```

### 建议分步执行

1. **第一批**: `actions/checkout` — 影响面最大（70 处），一次性全替换
2. **第二批**: `astral-sh/setup-uv` + `actions/setup-python` — 构建工具
3. **第三批**: `docker/*` actions — Docker 发布链
4. **第四批**: Dockerfile `FROM` — 固定基础镜像版本+SHA

### 风险

- **正面**: 防止供应链攻击
- **代价**: 升级 action 版本时需手动更新 SHA，增加维护负担
- **建议**: 至少 pin `actions/checkout` 和 `docker/*` 这些高权限 action

---

## 阶段 4: High — GitHub 设置 (5 个, 非代码)

这些需要在 GitHub 仓库 Settings 中配置，不能通过代码修复：

| 问题 | 位置 | 操作 |
|------|------|------|
| **BranchProtection** | Settings → Branches | 为 `main` 分支启用保护规则（require PR review, require status checks） |
| **CodeReview** | Settings → Branches | 要求至少 1 人 approve 才能合并 |
| **DependencyUpdate** | Settings → Security | 启用 Dependabot（或创建 `.github/dependabot.yml`） |
| **Vulnerabilities** | Settings → Security | 启用 Dependabot security alerts |
| **CIIBestPractices** | 外部 | 注册 OpenSSF Best Practices Badge |

### Dependabot 可以通过代码启用

创建 `.github/dependabot.yml`：
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "monthly"
```

---

## 执行优先级

| 阶段 | 内容 | 影响 | 复杂度 | 建议时间 |
|------|------|------|--------|---------|
| **1** | Critical workflow 评估 | 1 finding | 无改动 | 立即 |
| **2** | 7 个 workflow 加 permissions | 10 findings | 7 行代码 | 立即 |
| **3a** | Dependabot 配置文件 | 2 findings | 1 个新文件 | 立即 |
| **3b** | GitHub 分支保护设置 | 3 findings | 设置操作 | 立即 |
| **4a** | Pin actions/checkout SHA | ~70 findings | 批量替换 | 本周 |
| **4b** | Pin 其他 actions SHA | ~20 findings | 逐个查 SHA | 本周 |
| **4c** | Pin Dockerfile 基础镜像 | ~10 findings | 查询 digest | 本周 |

## 安全保证

- 阶段 2 (permissions): 只限制权限，不影响功能
- 阶段 3 (dependabot): 只增加自动化，不改现有代码
- 阶段 4 (pin SHA): 功能完全不变，只是锁定版本
- Critical workflow: 当前实现已安全，不需要修改
