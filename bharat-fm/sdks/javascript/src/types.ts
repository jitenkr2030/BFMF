/**
 * Type definitions for Bharat Foundation Model Framework SDK
 */

/**
 * Request model for text generation
 */
export interface GenerationRequest {
  /** Input prompt for text generation */
  prompt: string;
  /** Maximum number of tokens to generate */
  maxTokens?: number;
  /** Sampling temperature */
  temperature?: number;
  /** Top-p sampling parameter */
  topP?: number;
  /** Top-k sampling parameter */
  topK?: number;
  /** Number of beams for beam search */
  numBeams?: number;
  /** Whether to use sampling */
  doSample?: boolean;
  /** Language hint for generation */
  language?: string;
}

/**
 * Response model for text generation
 */
export interface GenerationResponse {
  /** Generated text */
  generatedText: string;
  /** Original prompt */
  prompt: string;
  /** Number of tokens generated */
  tokensGenerated: number;
  /** Time taken for generation in seconds */
  generationTime: number;
  /** Detected language */
  languageDetected?: string;
}

/**
 * Request model for batch generation
 */
export interface BatchGenerationRequest {
  /** List of generation requests */
  requests: GenerationRequest[];
}

/**
 * Response model for batch generation
 */
export interface BatchGenerationResponse {
  /** List of generation responses */
  responses: GenerationResponse[];
}

/**
 * Request model for embeddings
 */
export interface EmbeddingRequest {
  /** Input text for embedding */
  text: string;
  /** Whether to normalize embeddings */
  normalize?: boolean;
}

/**
 * Response model for embeddings
 */
export interface EmbeddingResponse {
  /** Embedding vector */
  embeddings: number[];
  /** Original text */
  text: string;
  /** Embedding dimension */
  embeddingDim: number;
}

/**
 * Model information response
 */
export interface ModelInfo {
  /** Name of the model */
  modelName: string;
  /** Type of the model */
  modelType: string;
  /** Size of the model */
  modelSize: string;
  /** List of supported languages */
  supportedLanguages: string[];
  /** Maximum context length */
  maxContextLength: number;
  /** Model version */
  version: string;
}

/**
 * Health check response
 */
export interface HealthResponse {
  /** Health status */
  status: string;
  /** Current timestamp */
  timestamp: string;
  /** Whether model is loaded */
  modelLoaded: boolean;
  /** Device being used */
  device: string;
}

/**
 * Supported languages response
 */
export interface SupportedLanguagesResponse {
  /** List of supported languages */
  supportedLanguages: string[];
  /** Whether multilingual is enabled */
  multilingualEnabled: boolean;
}

/**
 * Configuration interface for the client
 */
export interface ClientConfig {
  /** Base URL for the API */
  baseURL?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Maximum number of retries */
  maxRetries?: number;
  /** Delay between retries in milliseconds */
  retryDelay?: number;
  /** Whether to enable debug logging */
  debug?: boolean;
}

/**
 * Domain-specific configuration
 */
export interface DomainConfig {
  /** Domain-specific model path */
  modelPath?: string;
  /** Domain-specific API endpoint */
  endpoint?: string;
  /** Domain-specific parameters */
  parameters?: Record<string, any>;
}