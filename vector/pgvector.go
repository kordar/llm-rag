package vector

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/kordar/llm-rag/model"
	"github.com/pgvector/pgvector-go"

	_ "github.com/lib/pq"
)

type Store interface {
	Insert(ctx context.Context, doc model.Document) error
	Search(ctx context.Context, embedding []float32, topK int) ([]SearchHit, error)
	Close() error
}

type SearchHit struct {
	ID       string
	Content  string
	Metadata json.RawMessage
	Distance float32
}

type PGVector struct {
	db *sql.DB
}

func NewPGVector(dsn string) (*PGVector, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, err
	}
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return &PGVector{db: db}, nil
}

func (p *PGVector) Close() error {
	if p == nil || p.db == nil {
		return nil
	}
	return p.db.Close()
}

func (p *PGVector) Insert(ctx context.Context, doc model.Document) error {
	if p == nil || p.db == nil {
		return errors.New("vector: nil db")
	}
	query := `
INSERT INTO documents (id, content, embedding, metadata)
VALUES ($1, $2, $3, $4)
`
	_, err := p.db.ExecContext(
		ctx,
		query,
		doc.ID,
		doc.Content,
		pgvector.NewVector(doc.Embedding),
		nullableJSON(doc.Metadata),
	)
	return err
}

func (p *PGVector) Search(ctx context.Context, embedding []float32, topK int) ([]SearchHit, error) {
	if p == nil || p.db == nil {
		return nil, errors.New("vector: nil db")
	}
	if topK <= 0 {
		return []SearchHit{}, nil
	}

	query := fmt.Sprintf(`
SELECT id, content, metadata, embedding <-> $1 AS distance
FROM documents
ORDER BY embedding <-> $1
LIMIT %d
`, topK)

	rows, err := p.db.QueryContext(ctx, query, pgvector.NewVector(embedding))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var hits []SearchHit
	for rows.Next() {
		var h SearchHit
		var metadata []byte
		if err := rows.Scan(&h.ID, &h.Content, &metadata, &h.Distance); err != nil {
			return nil, err
		}
		if len(metadata) > 0 {
			h.Metadata = append(json.RawMessage(nil), metadata...)
		}
		hits = append(hits, h)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return hits, nil
}

func nullableJSON(b []byte) interface{} {
	if len(b) == 0 {
		return nil
	}
	return json.RawMessage(b)
}
