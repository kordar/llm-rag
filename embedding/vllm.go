package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type VLLMProvider struct {
	baseURL string
	client  *http.Client
	headers map[string]string
}

type VLLMOption func(*VLLMProvider)

func WithHTTPClient(hc *http.Client) VLLMOption {
	return func(p *VLLMProvider) {
		if hc != nil {
			p.client = hc
		}
	}
}

func WithHeader(key, value string) VLLMOption {
	return func(p *VLLMProvider) {
		if p.headers == nil {
			p.headers = map[string]string{}
		}
		p.headers[key] = value
	}
}

func NewVLLMProvider(baseURL string, opts ...VLLMOption) *VLLMProvider {
	p := &VLLMProvider{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	return p
}

type vllmEmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"`
}

type vllmEmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (p *VLLMProvider) Embed(ctx context.Context, model string, texts []string) ([][]float32, error) {
	if p == nil {
		return nil, errors.New("embedding: nil vllm provider")
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if model == "" {
		return nil, errors.New("embedding: empty model")
	}

	req := vllmEmbeddingRequest{
		Model: model,
		Input: texts,
	}

	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(p.baseURL, "/") + "/v1/embeddings"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range p.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var body bytes.Buffer
		_, _ = body.ReadFrom(resp.Body)
		return nil, fmt.Errorf("embedding: vllm http %d: %s", resp.StatusCode, body.String())
	}

	var out vllmEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}

	embs := make([][]float32, 0, len(out.Data))
	for _, d := range out.Data {
		embs = append(embs, d.Embedding)
	}
	return embs, nil
}
