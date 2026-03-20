package pipeline

import (
	"context"
	"sort"

	"github.com/kordar/llm-rag/embedding"
	"github.com/kordar/llm-rag/rerank"
	"github.com/kordar/llm-rag/vector"
)

type Hit struct {
	ID          string
	Content     string
	Metadata    []byte
	Distance    float32
	RerankScore float32
	HasRerank   bool
}

type Searcher struct {
	Embedder  *embedding.Client
	Store     vector.Store
	Reranker  *rerank.Client
	RecallTop int
	FinalTop  int
}

func (s *Searcher) Search(ctx context.Context, query string) ([]Hit, error) {
	recallTop := s.RecallTop
	if recallTop <= 0 {
		recallTop = 20
	}
	finalTop := s.FinalTop
	if finalTop <= 0 {
		finalTop = 5
	}

	embs, err := s.Embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}

	candidates, err := s.Store.Search(ctx, embs[0], recallTop)
	if err != nil {
		return nil, err
	}

	hits := make([]Hit, 0, len(candidates))
	for _, c := range candidates {
		hits = append(hits, Hit{
			ID:       c.ID,
			Content:  c.Content,
			Metadata: c.Metadata,
			Distance: c.Distance,
		})
	}
	if len(hits) == 0 {
		return hits, nil
	}

	if s.Reranker != nil {
		texts := make([]string, 0, len(hits))
		for _, h := range hits {
			texts = append(texts, h.Content)
		}

		scores, err := s.Reranker.Rerank(ctx, query, texts)
		if err == nil && len(scores) == len(hits) {
			for i := range hits {
				hits[i].RerankScore = scores[i]
				hits[i].HasRerank = true
			}
			sort.Slice(hits, func(i, j int) bool {
				return hits[i].RerankScore > hits[j].RerankScore
			})
		}
	}

	if finalTop > len(hits) {
		finalTop = len(hits)
	}
	return hits[:finalTop], nil
}
