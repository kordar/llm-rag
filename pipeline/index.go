package pipeline

import (
	"context"

	"github.com/google/uuid"
	"github.com/kordar/llm-rag/embedding"
	"github.com/kordar/llm-rag/model"
	"github.com/kordar/llm-rag/vector"
)

type Indexer struct {
	Embedder *embedding.Client
	Store    vector.Store
}

func (i *Indexer) Index(ctx context.Context, texts []string) error {
	if len(texts) == 0 {
		return nil
	}
	embeddings, err := i.Embedder.Embed(ctx, texts)
	if err != nil {
		return err
	}

	for idx, text := range texts {
		doc := model.Document{
			ID:        uuid.NewString(),
			Content:   text,
			Embedding: embeddings[idx],
		}
		if err := i.Store.Insert(ctx, doc); err != nil {
			return err
		}
	}
	return nil
}
