package domains

import (
    "context"
    "fmt"

    "github.com/bharat-ai/bharat-fm/sdks/go/internal/client"
    "github.com/bharat-ai/bharat-fm/sdks/go/internal/models"
)

// LanguageAIClient provides language-specific AI capabilities
type LanguageAIClient struct {
    client *client.Client
}

// NewLanguageAIClient creates a new language AI client
func NewLanguageAIClient(client *client.Client) *LanguageAIClient {
    return &LanguageAIClient{
        client: client,
    }
}

// TranslationRequest represents a translation request
type TranslationRequest struct {
    Text              string           `json:"text"`
    SourceLanguage    models.Language  `json:"source_language"`
    TargetLanguage    models.Language  `json:"target_language"`
    PreserveFormatting bool            `json:"preserve_formatting,omitempty"`
}

// TranslationResponse represents a translation response
type TranslationResponse struct {
    TranslatedText   string          `json:"translated_text"`
    OriginalText     string          `json:"original_text"`
    SourceLanguage   models.Language  `json:"source_language"`
    TargetLanguage   models.Language  `json:"target_language"`
    Confidence       float64         `json:"confidence,omitempty"`
    TranslationTime  float64         `json:"translation_time"`
}

// LanguageDetectionRequest represents a language detection request
type LanguageDetectionRequest struct {
    Text             string `json:"text"`
    IncludeConfidence bool   `json:"include_confidence,omitempty"`
}

// LanguageDetectionResponse represents a language detection response
type LanguageDetectionResponse struct {
    DetectedLanguage models.Language `json:"detected_language"`
    Confidence      float64             `json:"confidence,omitempty"`
    Probabilities   map[string]float64  `json:"probabilities,omitempty"`
    DetectionTime   float64             `json:"detection_time"`
}

// Translate translates text between languages
func (c *LanguageAIClient) Translate(ctx context.Context, req *TranslationRequest) (*TranslationResponse, error) {
    var result TranslationResponse
    
    apiReq := map[string]interface{}{
        "text":                req.Text,
        "source_language":     req.SourceLanguage,
        "target_language":     req.TargetLanguage,
        "preserve_formatting": req.PreserveFormatting,
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/language/translate")
    
    if err != nil {
        return nil, fmt.Errorf("translation request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("translation API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// DetectLanguage detects the language of given text
func (c *LanguageAIClient) DetectLanguage(ctx context.Context, req *LanguageDetectionRequest) (*LanguageDetectionResponse, error) {
    var result LanguageDetectionResponse
    
    apiReq := map[string]interface{}{
        "text":               req.Text,
        "include_confidence": req.IncludeConfidence,
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/language/detect-language")
    
    if err != nil {
        return nil, fmt.Errorf("language detection request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("language detection API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// GetSupportedLanguagePairs gets supported language pairs for translation
func (c *LanguageAIClient) GetSupportedLanguagePairs(ctx context.Context) ([]struct {
    Source   string `json:"source"`
    Target   string `json:"target"`
    Supported bool   `json:"supported"`
    Quality  string `json:"quality"`
}, error) {
    var result []struct {
        Source   string `json:"source"`
        Target   string `json:"target"`
        Supported bool   `json:"supported"`
        Quality  string `json:"quality"`
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetResult(&result).
        Get("/language/supported-pairs")
    
    if err != nil {
        return nil, fmt.Errorf("get supported pairs request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("get supported pairs API error: status %d", resp.StatusCode())
    }
    
    return result, nil
}

// SimpleTranslate provides a simple translation interface
func (c *LanguageAIClient) SimpleTranslate(ctx context.Context, text string, sourceLang, targetLang models.Language) (string, error) {
    req := &TranslationRequest{
        Text:           text,
        SourceLanguage: sourceLang,
        TargetLanguage: targetLang,
    }
    
    resp, err := c.Translate(ctx, req)
    if err != nil {
        return "", err
    }
    
    return resp.TranslatedText, nil
}

// SimpleDetectLanguage provides a simple language detection interface
func (c *LanguageAIClient) SimpleDetectLanguage(ctx context.Context, text string) (models.Language, error) {
    req := &LanguageDetectionRequest{
        Text:             text,
        IncludeConfidence: true,
    }
    
    resp, err := c.DetectLanguage(ctx, req)
    if err != nil {
        return "", err
    }
    
    return resp.DetectedLanguage, nil
}