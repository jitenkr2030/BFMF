/**
 * Bharat Foundation Model Framework JavaScript/TypeScript SDK
 * 
 * This SDK provides a convenient way to interact with BFMF APIs
 * and integrate Bharat AI capabilities into JavaScript/TypeScript applications.
 */

export { BharatAIClient } from './client';
export { BharatAIError } from './errors';
export { 
  GenerationRequest, 
  GenerationResponse, 
  ModelInfo, 
  HealthResponse,
  EmbeddingRequest,
  EmbeddingResponse
} from './types';
export { 
  Language, 
  ModelType, 
  GenerationParameters 
} from './constants';

// Re-export domain-specific clients
export { LanguageAIClient } from './domains/language';
export { GovernanceAIClient } from './domains/governance';
export { EducationAIClient } from './domains/education';
export { FinanceAIClient } from './domains/finance';

// Version
export const SDK_VERSION = '1.0.0';

// Default configuration
export const DEFAULT_CONFIG = {
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
};