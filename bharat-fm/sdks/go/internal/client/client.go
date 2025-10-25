package client

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"

    "github.com/go-resty/resty/v2"
    "github.com/sirupsen/logrus"

    "github.com/bharat-ai/bharat-fm/sdks/go/internal/models"
)

// Client is the main client for interacting with BFMF APIs
type Client struct {
    httpClient *resty.Client
    config     *Config
    logger     *logrus.Logger
}

// Config represents the client configuration
type Config struct {
    BaseURL     string
    APIKey      string
    Timeout     time.Duration
    MaxRetries  int
    RetryDelay  time.Duration
    Debug       bool
}

// NewConfig creates a new default configuration
func NewConfig() *Config {
    return &Config{
        BaseURL:     "http://localhost:8000",
        Timeout:     30 * time.Second,
        MaxRetries:  3,
        RetryDelay:  1 * time.Second,
        Debug:       false,
    }
}

// NewClient creates a new BFMF client
func NewClient(config *Config) *Client {
    if config == nil {
        config = NewConfig()
    }

    logger := logrus.New()
    if config.Debug {
        logger.SetLevel(logrus.DebugLevel)
    } else {
        logger.SetLevel(logrus.InfoLevel)
    }

    httpClient := resty.New().
        SetBaseURL(config.BaseURL).
        SetTimeout(config.Timeout).
        SetRetryCount(config.MaxRetries).
        SetRetryWaitTime(config.RetryDelay).
        AddRetryCondition(func(r *resty.Response, err error) bool {
            return shouldRetry(r, err)
        }).
        SetHeader("User-Agent", "Bharat-Go-SDK/1.0.0").
        SetHeader("Content-Type", "application/json")

    if config.APIKey != "" {
        httpClient.SetHeader("Authorization", "Bearer "+config.APIKey)
    }

    if config.Debug {
        httpClient.OnBeforeRequest(func(c *resty.Client, r *resty.Request) error {
            logger.Debugf("Request: %s %s", r.Method, r.URL)
            if r.Body != nil {
                if bodyBytes, err := json.Marshal(r.Body); err == nil {
                    logger.Debugf("Request body: %s", string(bodyBytes))
                }
            }
            return nil
        })

        httpClient.OnAfterResponse(func(c *resty.Client, r *resty.Response) error {
            logger.Debugf("Response: %d %s", r.StatusCode(), r.Status())
            if r.Body() != nil {
                logger.Debugf("Response body: %s", string(r.Body()))
            }
            return nil
        })
    }

    return &Client{
        httpClient: httpClient,
        config:     config,
        logger:     logger,
    }
}

// GenerateText generates text from a prompt
func (c *Client) GenerateText(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResponse, error) {
    var result models.GenerationResponse
    
    apiReq := map[string]interface{}{
        "prompt":      req.Prompt,
        "max_tokens":  req.MaxTokens,
        "temperature": req.Temperature,
        "top_p":       req.TopP,
        "top_k":       req.TopK,
        "num_beams":   req.NumBeams,
        "do_sample":   req.DoSample,
    }
    
    if req.Language != "" {
        apiReq["language"] = req.Language
    }

    resp, err := c.httpClient.R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/generate")
    
    if err != nil {
        return nil, c.handleError(err)
    }
    
    if resp.StatusCode() != http.StatusOK {
        return nil, c.handleAPIError(resp)
    }
    
    return &result, nil
}

// GenerateTextBatch generates text for multiple prompts
func (c *Client) GenerateTextBatch(ctx context.Context, req *models.BatchGenerationRequest) (*models.BatchGenerationResponse, error) {
    var result models.BatchGenerationResponse
    
    apiRequests := make([]map[string]interface{}, len(req.Requests))
    for i, r := range req.Requests {
        apiReq := map[string]interface{}{
            "prompt":      r.Prompt,
            "max_tokens":  r.MaxTokens,
            "temperature": r.Temperature,
            "top_p":       r.TopP,
            "top_k":       r.TopK,
            "num_beams":   r.NumBeams,
            "do_sample":   r.DoSample,
        }
        if r.Language != "" {
            apiReq["language"] = r.Language
        }
        apiRequests[i] = apiReq
    }

    resp, err := c.httpClient.R().
        SetContext(ctx).
        SetBody(apiRequests).
        SetResult(&result).
        Post("/batch_generate")
    
    if err != nil {
        return nil, c.handleError(err)
    }
    
    if resp.StatusCode() != http.StatusOK {
        return nil, c.handleAPIError(resp)
    }
    
    return &result, nil
}

// GetEmbeddings gets text embeddings
func (c *Client) GetEmbeddings(ctx context.Context, req *models.EmbeddingRequest) (*models.EmbeddingResponse, error) {
    var result models.EmbeddingResponse
    
    apiReq := map[string]interface{}{
        "text":      req.Text,
        "normalize": req.Normalize,
    }

    resp, err := c.httpClient.R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/embeddings")
    
    if err != nil {
        return nil, c.handleError(err)
    }
    
    if resp.StatusCode() != http.StatusOK {
        return nil, c.handleAPIError(resp)
    }
    
    return &result, nil
}

// GetModelInfo gets model information
func (c *Client) GetModelInfo(ctx context.Context) (*models.ModelInfo, error) {
    var result models.ModelInfo
    
    resp, err := c.httpClient.R().
        SetContext(ctx).
        SetResult(&result).
        Get("/model/info")
    
    if err != nil {
        return nil, c.handleError(err)
    }
    
    if resp.StatusCode() != http.StatusOK {
        return nil, c.handleAPIError(resp)
    }
    
    return &result, nil
}

// GetHealth checks API health
func (c *Client) GetHealth(ctx context.Context) (*models.HealthResponse, error) {
    var result models.HealthResponse
    
    resp, err := c.httpClient.R().
        SetContext(ctx).
        SetResult(&result).
        Get("/health")
    
    if err != nil {
        return nil, c.handleError(err)
    }
    
    if resp.StatusCode() != http.StatusOK {
        return nil, c.handleAPIError(resp)
    }
    
    return &result, nil
}

// GetSupportedLanguages gets supported languages
func (c *Client) GetSupportedLanguages(ctx context.Context) (*models.SupportedLanguagesResponse, error) {
    var result models.SupportedLanguagesResponse
    
    resp, err := c.httpClient.R().
        SetContext(ctx).
        SetResult(&result).
        Get("/languages")
    
    if err != nil {
        return nil, c.handleError(err)
    }
    
    if resp.StatusCode() != http.StatusOK {
        return nil, c.handleAPIError(resp)
    }
    
    return &result, nil
}

// GenerateTextStream streams text generation
func (c *Client) GenerateTextStream(ctx context.Context, req *models.GenerationRequest) (<-chan string, error) {
    ch := make(chan string)
    
    apiReq := map[string]interface{}{
        "prompt":      req.Prompt,
        "max_tokens":  req.MaxTokens,
        "temperature": req.Temperature,
        "top_p":       req.TopP,
        "top_k":       req.TopK,
        "num_beams":   req.NumBeams,
        "do_sample":   req.DoSample,
        "stream":      true,
    }
    
    if req.Language != "" {
        apiReq["language"] = req.Language
    }

    go func() {
        defer close(ch)
        
        reqBody, err := json.Marshal(apiReq)
        if err != nil {
            c.logger.Errorf("Failed to marshal request: %v", err)
            return
        }
        
        httpReq, err := http.NewRequestWithContext(ctx, "POST", c.config.BaseURL+"/generate", bytes.NewReader(reqBody))
        if err != nil {
            c.logger.Errorf("Failed to create request: %v", err)
            return
        }
        
        httpReq.Header.Set("Content-Type", "application/json")
        httpReq.Header.Set("User-Agent", "Bharat-Go-SDK/1.0.0")
        if c.config.APIKey != "" {
            httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)
        }
        
        resp, err := http.DefaultClient.Do(httpReq)
        if err != nil {
            c.logger.Errorf("Failed to send request: %v", err)
            return
        }
        defer resp.Body.Close()
        
        if resp.StatusCode != http.StatusOK {
            c.logger.Errorf("API returned status: %d", resp.StatusCode)
            return
        }
        
        decoder := json.NewDecoder(resp.Body)
        for {
            var line map[string]interface{}
            if err := decoder.Decode(&line); err != nil {
                if err == io.EOF {
                    break
                }
                c.logger.Errorf("Failed to decode line: %v", err)
                continue
            }
            
            if generatedText, ok := line["generated_text"].(string); ok {
                select {
                case ch <- generatedText:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()
    
    return ch, nil
}

// handleError handles client errors
func (c *Client) handleError(err error) error {
    c.logger.Errorf("Client error: %v", err)
    return fmt.Errorf("BFMF client error: %w", err)
}

// handleAPIError handles API errors
func (c *Client) handleAPIError(resp *resty.Response) error {
    var apiError struct {
        Message string `json:"message"`
        Code    string `json:"code"`
    }
    
    if err := json.Unmarshal(resp.Body(), &apiError); err == nil {
        c.logger.Errorf("API error: %s (code: %s)", apiError.Message, apiError.Code)
        return fmt.Errorf("BFMF API error: %s (code: %s, status: %d)", apiError.Message, apiError.Code, resp.StatusCode())
    }
    
    c.logger.Errorf("API error: status %d, body: %s", resp.StatusCode(), string(resp.Body()))
    return fmt.Errorf("BFMF API error: status %d", resp.StatusCode())
}

// shouldRetry determines if a request should be retried
func shouldRetry(resp *resty.Response, err error) bool {
    if err != nil {
        return true
    }
    
    statusCode := resp.StatusCode()
    return statusCode == http.StatusRequestTimeout ||
           statusCode == http.StatusTooManyRequests ||
           statusCode == http.StatusInternalServerError ||
           statusCode == http.StatusBadGateway ||
           statusCode == http.StatusServiceUnavailable ||
           statusCode == http.StatusGatewayTimeout
}

// Close closes the client and releases resources
func (c *Client) Close() error {
    // Resty client doesn't need explicit closing, but we can add cleanup here if needed
    return nil
}

// GetConfig returns the client configuration
func (c *Client) GetConfig() *Config {
    return c.config
}

// UpdateConfig updates the client configuration
func (c *Client) UpdateConfig(config *Config) {
    c.config = config
    c.httpClient.SetBaseURL(config.BaseURL)
    c.httpClient.SetTimeout(config.Timeout)
    c.httpClient.SetRetryCount(config.MaxRetries)
    c.httpClient.SetRetryWaitTime(config.RetryDelay)
    
    if config.APIKey != "" {
        c.httpClient.SetHeader("Authorization", "Bearer "+config.APIKey)
    } else {
        c.httpClient.RemoveHeader("Authorization")
    }
}