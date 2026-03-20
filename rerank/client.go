package rerank

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

type Client struct {
	baseURL string
	model   string
	client  *http.Client
	headers map[string]string
}

type Option func(*Client)

func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		if hc != nil {
			c.client = hc
		}
	}
}

func WithHeader(key, value string) Option {
	return func(c *Client) {
		if c.headers == nil {
			c.headers = map[string]string{}
		}
		c.headers[key] = value
	}
}

func New(baseURL, model string, opts ...Option) *Client {
	c := &Client{
		baseURL: baseURL,
		model:   model,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(c)
		}
	}
	return c
}

type request struct {
	Model string   `json:"model"`
	Query string   `json:"query"`
	Texts []string `json:"texts"`
}

type response struct {
	Scores []float32 `json:"scores"`
}

func (c *Client) Rerank(ctx context.Context, query string, texts []string) ([]float32, error) {
	if c == nil || c.client == nil {
		return nil, errors.New("rerank: nil client")
	}
	if c.model == "" {
		return nil, errors.New("rerank: empty model")
	}
	if len(texts) == 0 {
		return []float32{}, nil
	}

	req := request{
		Model: c.model,
		Query: query,
		Texts: texts,
	}
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(c.baseURL, "/") + "/v1/rerank"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var body bytes.Buffer
		_, _ = body.ReadFrom(resp.Body)
		return nil, fmt.Errorf("rerank: http %d: %s", resp.StatusCode, body.String())
	}

	var out response
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Scores, nil
}

