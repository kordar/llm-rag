package main

import (
	"context"
	"fmt"
	"os"

	"github.com/kordar/llm-rag/embedding"
	"github.com/kordar/llm-rag/pipeline"
	"github.com/kordar/llm-rag/rerank"
	"github.com/kordar/llm-rag/vector"
)

func main() {
	ctx := context.Background()

	embeddingBaseURL := getenv("EMBEDDING_BASE_URL", "http://localhost:8000")
	embeddingModel := getenv("EMBEDDING_MODEL", "/models/bge-large-zh")
	pgDSN := getenv("PG_DSN", "postgres://user:pass@localhost:5432/rag?sslmode=disable")
	rerankBaseURL := getenv("RERANK_BASE_URL", "http://localhost:8001")
	rerankModel := getenv("RERANK_MODEL", "bge-reranker")
	token := getenv("OLLAMA_BEARER_TOKEN", "xxxxx")

	embedder := embedding.New(
		embedding.NewVLLMProvider(embeddingBaseURL, embedding.WithHeader("Authorization", "Bearer "+token)),
		embedding.WithModel(embeddingModel),
		embedding.WithBatchSize(32),
		embedding.WithConcurrency(4),
	)

	fmt.Println("pg_dsn", pgDSN)
	store, err := vector.NewPGVector(pgDSN)
	if err != nil {
		panic(err)
	}
	defer store.Close()

	reranker := rerank.New(rerankBaseURL, rerankModel, rerank.WithHeader("Authorization", "Bearer "+token))

	indexer := &pipeline.Indexer{
		Embedder: embedder,
		Store:    store,
	}
	searcher := &pipeline.Searcher{
		Embedder: embedder,
		Store:    store,
		Reranker: reranker,
	}

	if err := indexer.Index(ctx, []string{
		"报销流程是：提交申请 → 审批 → 财务打款",
		"请假流程是：提交申请 → 领导审批",
	}); err != nil {
		panic(err)
	}

	hits, err := searcher.Search(ctx, "报销怎么走流程")
	if err != nil {
		panic(err)
	}

	for _, h := range hits {
		if h.HasRerank {
			fmt.Printf("%s\t%f\n", h.Content, h.RerankScore)
			continue
		}
		fmt.Printf("%s\t%f\n", h.Content, h.Distance)
	}
}

func getenv(key, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}
