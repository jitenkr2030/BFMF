/**
 * API types for the Bharat AI React Native SDK
 */

// Base request types
export interface BaseRequest {
  /** Request ID for tracking */
  requestId?: string;
  /** Language code for the request */
  language?: string;
  /** Domain-specific parameters */
  domain?: string;
  /** Custom metadata */
  metadata?: Record<string, any>;
}

// Text Generation
export interface GenerationRequest extends BaseRequest {
  /** Input prompt for text generation */
  prompt: string;
  /** Maximum number of tokens to generate */
  maxTokens?: number;
  /** Temperature for sampling (0.0 to 1.0) */
  temperature?: number;
  /** Top-p sampling parameter */
  topP?: number;
  /** Top-k sampling parameter */
  topK?: number;
  /** Number of sequences to generate */
  numSequences?: number;
  /** Stop sequences */
  stopSequences?: string[];
  /** Enable streaming response */
  stream?: boolean;
  /** Include probability scores */
  includeScores?: boolean;
}

export interface GenerationResponse {
  /** Unique response ID */
  id: string;
  /** Generated text */
  text: string;
  /** Number of tokens generated */
  tokensUsed: number;
  /** Confidence score */
  confidence: number;
  /** Generation timestamp */
  timestamp: string;
  /** Model used for generation */
  model: string;
  /** Language detected */
  language: string;
  /** Probability scores (if requested) */
  scores?: number[];
  /** Alternative completions */
  alternatives?: string[];
  /** Processing time in milliseconds */
  processingTime: number;
}

// Batch Generation
export interface BatchGenerationRequest extends BaseRequest {
  /** Array of prompts to process */
  prompts: string[];
  /** Maximum tokens per response */
  maxTokens?: number;
  /** Temperature for sampling */
  temperature?: number;
  /** Process in parallel */
  parallel?: boolean;
}

export interface BatchGenerationResponse {
  /** Batch processing ID */
  batchId: string;
  /** Array of generation responses */
  responses: GenerationResponse[];
  /** Total processing time */
  processingTime: number;
  /** Success rate */
  successRate: number;
}

// Embeddings
export interface EmbeddingRequest extends BaseRequest {
  /** Text to embed */
  text: string;
  /** Embedding model to use */
  model?: string;
  /** Embedding dimensions */
  dimensions?: number;
  /** Normalize embeddings */
  normalize?: boolean;
}

export interface EmbeddingResponse {
  /** Unique embedding ID */
  id: string;
  /** Embedding vector */
  embedding: number[];
  /** Dimensions of the embedding */
  dimensions: number;
  /** Model used */
  model: string;
  /** Processing time */
  processingTime: number;
}

// Model Information
export interface ModelInfo {
  /** Model name */
  name: string;
  /** Model version */
  version: string;
  /** Model description */
  description: string;
  /** Supported languages */
  supportedLanguages: string[];
  /** Capabilities */
  capabilities: string[];
  /** Maximum context length */
  maxContextLength: number;
  /** Supported domains */
  domains: string[];
  /** Model size in GB */
  modelSize: number;
  /** Last updated timestamp */
  lastUpdated: string;
}

// Health Check
export interface HealthResponse {
  /** Service status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Service version */
  version: string;
  /** Uptime in seconds */
  uptime: number;
  /** Current load metrics */
  metrics: {
    cpu: number;
    memory: number;
    requestsPerSecond: number;
    averageResponseTime: number;
  };
  /** Available models */
  availableModels: string[];
  /** Service dependencies */
  dependencies: {
    [key: string]: 'healthy' | 'degraded' | 'unhealthy';
  };
  /** Last health check timestamp */
  timestamp: string;
}

// Supported Languages
export interface SupportedLanguagesResponse {
  /** Array of supported languages */
  languages: Array<{
    /** Language code */
    code: string;
    /** Language name */
    name: string;
    /** Native name */
    nativeName: string;
    /** Supported domains */
    domains: string[];
    /** Support level */
    supportLevel: 'full' | 'partial' | 'experimental';
  }>;
  /** Total number of supported languages */
  totalLanguages: number;
  /** Last updated timestamp */
  lastUpdated: string;
}

// Streaming Response
export interface StreamingResponse {
  /** Chunk ID */
  chunkId: string;
  /** Text chunk */
  text: string;
  /** Is this the final chunk */
  isFinal: boolean;
  /** Cumulative text */
  cumulativeText: string;
  /** Processing time for this chunk */
  chunkProcessingTime: number;
  /** Total processing time */
  totalProcessingTime: number;
}

// Error Response
export interface ErrorResponse {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Detailed error description */
  details?: string;
  /** HTTP status code */
  statusCode: number;
  /** Request ID */
  requestId?: string;
  /** Timestamp */
  timestamp: string;
  /** Retry information */
  retry?: {
    /** Whether to retry */
    shouldRetry: boolean;
    /** Suggested retry delay */
    retryAfter?: number;
    /** Maximum retries */
    maxRetries?: number;
  };
}