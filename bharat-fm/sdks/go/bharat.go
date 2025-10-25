// Package bharat provides the Go SDK for Bharat Foundation Model Framework
package bharat

import (
    "github.com/bharat-ai/bharat-fm/sdks/go/internal/client"
    "github.com/bharat-ai/bharat-fm/sdks/go/internal/domains"
    "github.com/bharat-ai/bharat-fm/sdks/go/internal/models"
)

// Client is the main client for interacting with BFMF APIs
type Client = client.Client

// Config represents the client configuration
type Config = client.Config

// NewConfig creates a new default configuration
func NewConfig() *Config {
    return client.NewConfig()
}

// NewClient creates a new BFMF client
func NewClient(config *Config) *Client {
    return client.NewClient(config)
}

// Model types
type (
    GenerationRequest         = models.GenerationRequest
    GenerationResponse        = models.GenerationResponse
    BatchGenerationRequest   = models.BatchGenerationRequest
    BatchGenerationResponse  = models.BatchGenerationResponse
    EmbeddingRequest         = models.EmbeddingRequest
    EmbeddingResponse        = models.EmbeddingResponse
    ModelInfo                = models.ModelInfo
    HealthResponse           = models.HealthResponse
    SupportedLanguagesResponse = models.SupportedLanguagesResponse
    Language                 = models.Language
)

// Domain clients
type (
    LanguageAIClient   = domains.LanguageAIClient
    GovernanceAIClient = domains.GovernanceAIClient
    EducationAIClient  = domains.EducationAIClient
    FinanceAIClient    = domains.FinanceAIClient
)

// Domain request/response types
type (
    TranslationRequest         = domains.TranslationRequest
    TranslationResponse        = domains.TranslationResponse
    LanguageDetectionRequest  = domains.LanguageDetectionRequest
    LanguageDetectionResponse = domains.LanguageDetectionResponse
    RTIRequest               = domains.RTIRequest
    RTIResponse              = domains.RTIResponse
    PolicyAnalysisRequest     = domains.PolicyAnalysisRequest
    PolicyAnalysisResponse    = domains.PolicyAnalysisResponse
    TutoringRequest          = domains.TutoringRequest
    TutoringResponse         = domains.TutoringResponse
    ContentGenerationRequest  = domains.ContentGenerationRequest
    ContentGenerationResponse = domains.ContentGenerationResponse
    FinancialAnalysisRequest  = domains.FinancialAnalysisRequest
    FinancialAnalysisResponse = domains.FinancialAnalysisResponse
    TransactionAuditRequest   = domains.TransactionAuditRequest
    TransactionAuditResponse  = domains.TransactionAuditResponse
)

// NewGenerationRequest creates a new generation request
func NewGenerationRequest(prompt string) *GenerationRequest {
    return models.NewGenerationRequest(prompt)
}

// NewBatchGenerationRequest creates a new batch generation request
func NewBatchGenerationRequest(requests []*GenerationRequest) *BatchGenerationRequest {
    return models.NewBatchGenerationRequest(requests)
}

// NewEmbeddingRequest creates a new embedding request
func NewEmbeddingRequest(text string) *EmbeddingRequest {
    return models.NewEmbeddingRequest(text)
}

// NewLanguageAIClient creates a new language AI client
func NewLanguageAIClient(client *Client) *LanguageAIClient {
    return domains.NewLanguageAIClient(client)
}

// NewGovernanceAIClient creates a new governance AI client
func NewGovernanceAIClient(client *Client) *GovernanceAIClient {
    return domains.NewGovernanceAIClient(client)
}

// NewEducationAIClient creates a new education AI client
func NewEducationAIClient(client *Client) *EducationAIClient {
    return domains.NewEducationAIClient(client)
}

// NewFinanceAIClient creates a new finance AI client
func NewFinanceAIClient(client *Client) *FinanceAIClient {
    return domains.NewFinanceAIClient(client)
}

// GetAllLanguages returns all supported languages
func GetAllLanguages() []Language {
    return models.GetAllLanguages()
}

// Version returns the SDK version
func Version() string {
    return "1.0.0"
}