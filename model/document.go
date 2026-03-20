package model

import "encoding/json"

type Document struct {
	ID        string
	Content   string
	Embedding []float32
	Metadata  json.RawMessage
}

