package embedding

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"
)

type Provider interface {
	Embed(ctx context.Context, model string, texts []string) ([][]float32, error)
}

type Options struct {
	Model       string
	BatchSize   int
	Concurrency int
	Retry       RetryOptions
}

type Option func(*Options)

func WithModel(model string) Option {
	return func(o *Options) {
		o.Model = model
	}
}

func WithBatchSize(n int) Option {
	return func(o *Options) {
		o.BatchSize = n
	}
}

func WithConcurrency(n int) Option {
	return func(o *Options) {
		o.Concurrency = n
	}
}

func WithRetry(opts RetryOptions) Option {
	return func(o *Options) {
		o.Retry = opts
	}
}

type RetryOptions struct {
	MaxAttempts int
	BaseDelay   time.Duration
	MaxDelay    time.Duration
	Retryable   func(error) bool
}

type Client struct {
	provider Provider
	opts     Options
}

type batch struct {
	start int
	texts []string
}

func New(provider Provider, opts ...Option) *Client {
	o := Options{
		BatchSize:   64,
		Concurrency: 4,
		Retry: RetryOptions{
			MaxAttempts: 3,
			BaseDelay:   200 * time.Millisecond,
			MaxDelay:    2 * time.Second,
		},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	if o.Concurrency <= 0 {
		o.Concurrency = 1
	}
	return &Client{provider: provider, opts: o}
}

func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return c.EmbedWithModel(ctx, c.opts.Model, texts)
}

func (c *Client) EmbedWithModel(ctx context.Context, model string, texts []string) ([][]float32, error) {
	if c == nil || c.provider == nil {
		return nil, errors.New("embedding: nil client/provider")
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	batches := splitBatches(texts, c.opts.BatchSize)
	sem := make(chan struct{}, c.opts.Concurrency)

	out := make([][]float32, len(texts))
	var firstErr error
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, b := range batches {
		b := b
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			embs, err := c.embedWithRetry(ctx, model, b.texts)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}
			if len(embs) != len(b.texts) {
				mu.Lock()
				if firstErr == nil {
					firstErr = errors.New("embedding: invalid response size")
				}
				mu.Unlock()
				return
			}
			for i := range embs {
				out[b.start+i] = embs[i]
			}
		}()
	}

	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

func (c *Client) embedWithRetry(ctx context.Context, model string, texts []string) ([][]float32, error) {
	opts := c.opts.Retry
	maxAttempts := opts.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 1
	}

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		embs, err := c.provider.Embed(ctx, model, texts)
		if err == nil {
			return embs, nil
		}
		lastErr = err

		retryable := true
		if opts.Retryable != nil {
			retryable = opts.Retryable(err)
		}
		if !retryable || attempt == maxAttempts {
			break
		}

		delay := backoffDelay(opts.BaseDelay, opts.MaxDelay, attempt-1)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
	return nil, lastErr
}

func splitBatches(texts []string, batchSize int) []batch {
	if batchSize <= 0 || batchSize >= len(texts) {
		return []batch{{start: 0, texts: texts}}
	}
	out := make([]batch, 0, int(math.Ceil(float64(len(texts))/float64(batchSize))))
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		out = append(out, batch{start: i, texts: texts[i:end]})
	}
	return out
}

func backoffDelay(base, max time.Duration, attempt int) time.Duration {
	if base <= 0 {
		base = 100 * time.Millisecond
	}
	if max <= 0 {
		max = 2 * time.Second
	}
	mult := 1 << attempt
	d := time.Duration(mult) * base
	if d > max {
		return max
	}
	return d
}
