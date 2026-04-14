# M-flow 0.3.4 发布计划 — 三渠道同步

## 背景

0.3.3 发布后修复了 **20+ 个生产 bug**，但修复只在 GitHub main 分支，未发布到任何用户可获取的渠道。当前用户在所有渠道获取的都是有 bug 的 0.3.3 版本。

## 影响用户的关键 bug（0.3.3 中存在）

| Bug | 影响 | 严重性 |
|-----|------|--------|
| `max_tokens` 与 gpt-5 不兼容 | 所有 LLM 调用失败 | **Critical** |
| 会话历史 `compress_text(str)` 类型错误 | 使用 session_id 搜索必崩 | **Critical** |
| 音频处理 `create_transcript` 方法不存在 | 音频文件无法处理 | **High** |
| 图像处理 `transcribe_image` 方法不存在 | 图像文件无法处理 | **High** |
| UUID 序列化缺失 | 搜索结果持久化随机崩溃 | **High** |
| `_hash_sensitive` 死代码 | 遥测数据全部损坏 | **Medium** |
| CLI argparse dest 冲突 | CLI 工具无法启动 | **Medium** |
| JPEG 扩展名带多余点号 | JPEG 图片无法识别 | **Medium** |
| Retry 装饰器重试 400/401 | 客户端错误浪费 120 秒 | **Medium** |
| 去重 tenant_id 不规范 | 去重失效产生重复数据 | **Medium** |

---

## 发布步骤

### 第 1 步：版本号 + 锁文件 + 变更日志

```bash
# 1. 更新版本号
# pyproject.toml: version = "0.3.3" → "0.3.4"

# 2. 重新生成 uv.lock
uv lock

# 3. 更新 CHANGELOG.md
# 添加 [0.3.4] 条目（含 0.3.3 遗漏的条目）

# 4. 提交
git add pyproject.toml uv.lock CHANGELOG.md
git commit -m "release: bump version to 0.3.4"
git push origin main
```

### 第 2 步：PyPI + DockerHub（统一 release workflow）

在 GitHub Actions 页面手动触发 `release | Publish M-flow`：
- **flavour**: `main`（推 latest 标签）
- 该 workflow 自动：
  1. 读取 `pyproject.toml` 版本号（0.3.4）
  2. 创建 `v0.3.4` git tag
  3. 创建 GitHub Release
  4. `uv build` + `uv publish` 到 PyPI
  5. 构建并推送 `m_flow/m_flow:0.3.4` + `m_flow/m_flow:latest`

**前提条件**：
- `PYPI_TOKEN` secret 已配置
- `DOCKER_USERNAME` + `DOCKER_PASSWORD` secrets 已配置

### 第 3 步：MCP Docker 镜像

GitHub Release 创建后会自动触发 `dockerhub-mcp.yml`，推送 `m_flow/m_flow-mcp` 镜像。

### 第 4 步：OpenClaw Skill 更新

**重要发现**：OpenClaw skill 引用的是 `flowelement/m_flow-mcp`，而 CI 推送的是 `m_flow/m_flow-mcp`。命名空间不一致。

需要手动更新：
1. `openclaw-skill/mflow-memory/scripts/setup.sh` — 更新镜像 tag/digest
2. `openclaw-skill/mflow-memory/skills/mflow-memory/scripts/setup.sh` — 同上
3. 通过 ClawHub CLI 重新发布 skill

### 第 5 步：验证

```bash
# PyPI
pip install mflow-ai==0.3.4
python -c "import m_flow; print(m_flow.__version__)"

# Docker
docker pull m_flow/m_flow:0.3.4
docker run --rm m_flow/m_flow:0.3.4 python -c "import m_flow"

# OpenClaw
# 运行更新后的 setup.sh 验证 MCP 服务启动
```

---

## 需要验证的 Secrets

| Secret | 用途 | 是否已配置 |
|--------|------|-----------|
| `PYPI_TOKEN` | PyPI 发布 | **需确认** |
| `DOCKER_USERNAME` | DockerHub 登录 | **需确认** |
| `DOCKER_PASSWORD` | DockerHub 密码 | **需确认** |

---

## 风险与注意事项

1. **uv.lock 必须同步**：Dependabot 合并了多个依赖更新，`uv lock` 会生成新的锁文件。release workflow 使用 `--locked` 参数，锁文件不一致会构建失败。

2. **Docker 命名空间不一致**：CI 用 `m_flow/m_flow-mcp`，OpenClaw 用 `flowelement/m_flow-mcp`。发布后需要确认两个命名空间都推送了。

3. **CHANGELOG.md 缺少 0.3.3 条目**：当前最新条目是 `[0.3.2]`。建议同时补齐 0.3.3 和 0.3.4。

4. **Helm Chart 版本过旧**：`deployment/helm/Chart.yaml` 的 `appVersion` 还是 `0.3.1`，需要一并更新。
