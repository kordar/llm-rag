# LLM RAG 骨架（Go + PGVector + vLLM Embedding + Rerank）

该仓库提供一套可直接编译运行的 RAG 企业级骨架：

- Embedding：接口化 Provider + 可配置的批处理/并发/重试
- 向量库：PostgreSQL + pgvector
- 精排：可选 rerank 服务（失败自动降级为仅向量召回）

## 目录结构

```
rag/
  main.go
  embedding/
    client.go
    vllm.go
  model/
    document.go
  pipeline/
    index.go
    search.go
  rerank/
    client.go
  vector/
    pgvector.go
```

## 前置条件

- Go 1.22+
- PostgreSQL（已安装 pgvector 扩展）
- Embedding 服务（兼容 OpenAI `/v1/embeddings` 协议，示例以 vLLM 为主）
- 可选：Rerank 服务（`/v1/rerank`，返回 `scores: []float32`）

## 数据库初始化

在你的 Postgres 中执行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1024),
  metadata JSONB
);

CREATE INDEX IF NOT EXISTS documents_embedding_ivfflat
  ON documents USING ivfflat (embedding vector_cosine_ops);
```

说明：
- `VECTOR(1024)` 的维度需要与你的 embedding 模型输出一致（例如 1024/768 等）
- `ivfflat` 在小数据量阶段可以先不建，或在写入足够数据后再 `REINDEX/ANALYZE`

## 配置（环境变量）

`rag/main.go` 支持如下环境变量（均有默认值）：

- `EMBEDDING_BASE_URL`：默认 `http://localhost:8000`
- `EMBEDDING_MODEL`：默认 `/models/bge-large-zh`
- `PG_DSN`：默认 `postgres://user:pass@localhost:5432/rag?sslmode=disable`
- `RERANK_BASE_URL`：默认 `http://localhost:8001`
- `RERANK_MODEL`：默认 `bge-reranker`

## 运行

在仓库根目录执行：

```bash
go run ./rag
```

程序会：
1. 调用 embedding 生成向量
2. 写入 Postgres（documents 表）
3. 以查询句进行向量召回（默认召回 20）
4. 若配置了 rerank，则对召回候选精排并输出前 5

## 扩展点

- Embedding Provider：实现 `rag/embedding.Provider` 即可接入任意 embedding 服务
- 向量库：实现 `rag/vector.Store` 可切换到其他向量数据库
- Rerank：`rag/rerank.Client` 为独立客户端，pipeline 中为可选组件

