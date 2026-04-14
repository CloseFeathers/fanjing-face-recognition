# 最后 4 个 CI 失败的深度根因分析

## 总览

| 失败 | 真正的错误 | 分类 | 严重性 |
|------|-----------|------|--------|
| Conversation FS | `InvalidSummaryInputsError: Expected ContentFragment, got str` | 预存在的接口不匹配 bug | **高** |
| Conversation Redis | 同上 | 同上 | **高** |
| Dedup PGVector | `Expected 1, found 2` — PG 清理不完整 | 预存在 bug + 测试环境 | 中 |
| Dataset DB handler | `Graph database not found` — 错误的环境变量名 | 预存在的测试 bug | 中 |

---

## 1. Conversation History (FS + Redis) — 接口不匹配

### 错误

```
InvalidSummaryInputsError: Expected a list of ContentFragment instances, got str.
```

### 根因

**不是 UUID 序列化问题**（之前的诊断是错的）。真正的问题是 retriever 调用了错误的函数。

调用链：

```
m_flow.search(session_id=...) 
  → UnifiedTripletSearch.get_completion()
    → compress_text(context_text)    ← context_text 是 str
```

`compress_text` 的签名是：

```python
# m_flow/knowledge/summarization/summarize_text.py:28
async def compress_text(data_chunks: list[ContentFragment], ...):
```

但 retriever 传入的是 **字符串**（由 `convert_retrieved_objects_to_context` 生成的纯文本）。

**3 个 retriever 都有同样的问题**：
- `unified_triplet_search.py:209` — `compress_text(context_text)`
- `episodic_retriever.py:289` — `compress_text(context_text)`
- `procedural_retriever.py:403` — `compress_text(context_text)`

### 分析

`compress_text` 在 memorize 管道中被正确使用（接收 `list[ContentFragment]`），但在 retrieval 端被**误用**。retriever 想要的是"压缩搜索上下文文本用于会话缓存"，但 `compress_text` 在重构后只接受结构化的 `ContentFragment` 列表。

**这是一个接口演化 bug**——`compress_text` 的签名被改了但调用方没有同步更新。

### 修复方向

两种方案：

**方案 A (推荐)**：在 retriever 中跳过 `compress_text`，直接使用原始 `context_text` 字符串

```python
# unified_triplet_search.py 原来的：
context_summary, completion = await asyncio.gather(
    compress_text(context_text),  # ← 这里错了
    LLMService.complete_text(...)
)

# 改为：
context_summary = context_text  # 或截断
completion = await LLMService.complete_text(...)
```

**方案 B**：创建一个 `summarize_context_text(text: str) -> str` 函数给 retriever 使用，不依赖 ContentFragment。

### 影响范围

- 3 个 retriever 文件需要修改
- 不影响 memorize 管道（它正确使用 `compress_text`）
- 修复后 Conversation FS + Redis 都会通过

---

## 2. Dedup PGVector — Postgres 清理不完整

### 错误

```
Expected 1 data entity, found 2. Text file deduplication failed
```

### 根因

**两层问题叠加：**

**(A) Postgres 的 `delete_database` 不调 `cache_clear()`**

SQLite 版本的 `delete_database` (SqlAlchemyAdapter.py:398) 会调用 `create_relational_engine.cache_clear()`，但 **Postgres 版本 (第 404-412 行) 不会**。这导致 `get_relational_config()` 的 LRU 缓存可能残留旧的配置。

**(B) 跨测试的 Postgres 状态残留**

去重测试在同一 CI job 中先跑 SQLite 版再跑 PG 版。Postgres 是外部持久化服务（Docker 容器），`reset_system()` 中的 `delete_database()` 通过 `DROP TABLE ... CASCADE` + `create_database()` 清理。但如果：

- 其他 E2E 测试先于去重测试运行并写入了同一 PG
- `public_staging` schema 不存在导致 reflect 异常被静默
- 或 pgvector 的 `prune()` 和关系 DB 的 `delete_database()` 对同一 PG 有竞态

任一情况都可能导致表未完全清理。

### 修复方向

1. 在 Postgres 版 `delete_database()` 中也调用 `create_relational_engine.cache_clear()`
2. 去重测试开头显式 `DROP TABLE IF EXISTS data CASCADE` 确保干净状态
3. 或改用独立的 Postgres database name 避免跨测试冲突

---

## 3. Dataset DB handler — 环境变量名错误

### 错误

```
Graph database not found at .../test_dataset_database_handler/databases/{user_id}/test.kuzu
```

### 根因

**测试使用了错误的环境变量名**，导致自定义 handler 从未被启用。

测试设置 (test_dataset_database_handler.py:16-17)：
```python
os.environ["MFLOW_GRAPH_PARTITION_HANDLER"] = "custom_kuzu_handler"
os.environ["MFLOW_VECTOR_PARTITION_HANDLER"] = "custom_lancedb_handler"
```

但 `GraphConfig` 的字段名不是 `graph_partition_handler`，而是 `graph_dataset_database_handler`：

```python
# m_flow/adapters/graph/config.py:49
graph_dataset_database_handler: str = _PROVIDER_KUZU
```

`pydantic-settings` 加上 `env_prefix="MFLOW_"` 后，正确的环境变量应该是：
```
MFLOW_GRAPH_DATASET_DATABASE_HANDLER=custom_kuzu_handler
```

而不是 `MFLOW_GRAPH_PARTITION_HANDLER`。

**结果**：
1. 自定义 `CustomKuzuHandler` 从未被调用
2. 系统使用默认的 `KuzuDatasetStoreHandler`
3. 默认 handler 创建数据库名为 `{dataset_id}.pkl`（不是 `test.kuzu`）
4. 测试断言检查 `test.kuzu` → 文件不存在 → 失败

### 修复方向

修正环境变量名：
```python
os.environ["MFLOW_GRAPH_DATASET_DATABASE_HANDLER"] = "custom_kuzu_handler"
os.environ["MFLOW_VECTOR_DATASET_DATABASE_HANDLER"] = "custom_lancedb_handler"
```

---

## 安全性验证

### 修复 #1: Conversation — 移除 compress_text 调用

**验证结论: 完全安全**

- `compress_text(context_text)` 有**两重错误**:
  1. 传入 `str` 但签名要求 `list[ContentFragment]` → 直接崩溃
  2. 即使不崩溃，返回 `list[FragmentDigest]`，但 `save_conversation_history` 要求 `context_summary: str`
- 修复: 直接用 `context_text`（本身已是 str）赋值给 `context_summary`
- **不改变任何生产行为** — `compress_text` 在此路径上从未成功执行过，所以移除它不会改变任何已有的成功路径
- 需改 3 个文件: `unified_triplet_search.py`, `episodic_retriever.py`, `procedural_retriever.py`

### 修复 #2: Dataset handler — 修正环境变量名

**验证结论: 完全安全**

- 仅改测试文件中的环境变量名
- `MFLOW_GRAPH_PARTITION_HANDLER` → `MFLOW_GRAPH_DATASET_DATABASE_HANDLER`
- `MFLOW_VECTOR_PARTITION_HANDLER` → `MFLOW_VECTOR_DATASET_DATABASE_HANDLER`
- 与 `GraphConfig.graph_dataset_database_handler` 和 `VectorConfig.vector_dataset_database_handler` 对齐
- **零生产代码改动**

### 修复 #3: Dedup PG — 在 delete_database 加 cache_clear

**验证结论: 安全**

- SQLite 版 `delete_database` (第 394-398 行) **已经调用** `create_relational_engine.cache_clear()` — 这是有意设计
- PG 版 (第 403-412 行) **遗漏了**同样的调用 — 这是 bug
- 添加 `cache_clear()` 效果: 下次获取引擎时重新创建，确保配置一致
- **唯一风险**: 如果有并发请求在 cache_clear 后、重建前访问引擎 → 会重新创建，不会崩溃
- 这是**补齐 SQLite 版已有的逻辑**，不是新增行为

## 修复优先级

| # | 修复 | 复杂度 | 影响 | 安全性 |
|---|------|--------|------|--------|
| 1 | Conversation: 移除 compress_text 调用 | 低 | 修复 2 job + 真实 bug | 零风险 |
| 2 | Dataset handler: 修正环境变量名 | 低 | 修复 1 job | 零风险 |
| 3 | Dedup: PG delete_database 加 cache_clear | 低 | 修复 1 job + 真实 bug | 极低风险 |

3 个修复全部安全，全部是修复预存在的 bug，不引入新行为。
