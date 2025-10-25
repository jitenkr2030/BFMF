package models

// GenerationRequest represents a text generation request
type GenerationRequest struct {
    Prompt      string  `json:"prompt"`
    MaxTokens   int     `json:"max_tokens,omitempty"`
    Temperature float64 `json:"temperature,omitempty"`
    TopP        float64 `json:"top_p,omitempty"`
    TopK        int     `json:"top_k,omitempty"`
    NumBeams    int     `json:"num_beams,omitempty"`
    DoSample    bool    `json:"do_sample,omitempty"`
    Language    string  `json:"language,omitempty"`
}

// NewGenerationRequest creates a new generation request
func NewGenerationRequest(prompt string) *GenerationRequest {
    return &GenerationRequest{
        Prompt:      prompt,
        MaxTokens:   100,
        Temperature: 1.0,
        TopP:        1.0,
        TopK:        50,
        NumBeams:    1,
        DoSample:    true,
    }
}

// GenerationResponse represents a text generation response
type GenerationResponse struct {
    GeneratedText   string  `json:"generated_text"`
    Prompt          string  `json:"prompt"`
    TokensGenerated int     `json:"tokens_generated"`
    GenerationTime float64 `json:"generation_time"`
    LanguageDetected string `json:"language_detected,omitempty"`
}

// BatchGenerationRequest represents a batch text generation request
type BatchGenerationRequest struct {
    Requests []*GenerationRequest `json:"requests"`
}

// NewBatchGenerationRequest creates a new batch generation request
func NewBatchGenerationRequest(requests []*GenerationRequest) *BatchGenerationRequest {
    return &BatchGenerationRequest{
        Requests: requests,
    }
}

// BatchGenerationResponse represents a batch text generation response
type BatchGenerationResponse struct {
    Responses []*GenerationResponse `json:"responses"`
}

// EmbeddingRequest represents an embedding request
type EmbeddingRequest struct {
    Text     string `json:"text"`
    Normalize bool   `json:"normalize,omitempty"`
}

// NewEmbeddingRequest creates a new embedding request
func NewEmbeddingRequest(text string) *EmbeddingRequest {
    return &EmbeddingRequest{
        Text:     text,
        Normalize: true,
    }
}

// EmbeddingResponse represents an embedding response
type EmbeddingResponse struct {
    Embeddings  []float64 `json:"embeddings"`
    Text        string    `json:"text"`
    EmbeddingDim int       `json:"embedding_dim"`
}

// ModelInfo represents model information
type ModelInfo struct {
    ModelName         string   `json:"model_name"`
    ModelType         string   `json:"model_type"`
    ModelSize         string   `json:"model_size"`
    SupportedLanguages []string `json:"supported_languages"`
    MaxContextLength  int      `json:"max_context_length"`
    Version           string   `json:"version"`
}

// HealthResponse represents a health check response
type HealthResponse struct {
    Status     string `json:"status"`
    Timestamp string `json:"timestamp"`
    ModelLoaded bool   `json:"model_loaded"`
    Device     string `json:"device"`
}

// SupportedLanguagesResponse represents supported languages response
type SupportedLanguagesResponse struct {
    SupportedLanguages []string `json:"supported_languages"`
    MultilingualEnabled bool    `json:"multilingual_enabled"`
}

// Language represents supported languages
type Language string

const (
    LanguageHindi      Language = "hi"
    LanguageEnglish    Language = "en"
    LanguageBengali    Language = "bn"
    LanguageTamil      Language = "ta"
    LanguageTelugu     Language = "te"
    LanguageMarathi    Language = "mr"
    LanguageGujarati   Language = "gu"
    LanguageKannada    Language = "kn"
    LanguageMalayalam  Language = "ml"
    LanguagePunjabi    Language = "pa"
    LanguageOdia       Language = "or"
    LanguageAssamese   Language = "as"
    LanguageSanskrit   Language = "sa"
    LanguageUrdu       Language = "ur"
    LanguageKashmiri   Language = "ks"
    LanguageSindhi     Language = "sd"
    LanguageNepali     Language = "ne"
    LanguageManipuri   Language = "mni"
    LanguageKonkani    Language = "kok"
    LanguageMaithili   Language = "mai"
    LanguageSantali    Language = "sat"
    LanguageDogri      Language = "doi"
    LanguageBodo       Language = "brx"
)

// GetAllLanguages returns all supported languages
func GetAllLanguages() []Language {
    return []Language{
        LanguageHindi,
        LanguageEnglish,
        LanguageBengali,
        LanguageTamil,
        LanguageTelugu,
        LanguageMarathi,
        LanguageGujarati,
        LanguageKannada,
        LanguageMalayalam,
        LanguagePunjabi,
        LanguageOdia,
        LanguageAssamese,
        LanguageSanskrit,
        LanguageUrdu,
        LanguageKashmiri,
        LanguageSindhi,
        LanguageNepali,
        LanguageManipuri,
        LanguageKonkani,
        LanguageMaithili,
        LanguageSantali,
        LanguageDogri,
        LanguageBodo,
    }
}