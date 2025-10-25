/**
 * Constants for Bharat Foundation Model Framework SDK
 */

/**
 * Supported languages
 */
export enum Language {
  HINDI = 'hi',
  ENGLISH = 'en',
  BENGALI = 'bn',
  TAMIL = 'ta',
  TELUGU = 'te',
  MARATHI = 'mr',
  GUJARATI = 'gu',
  KANNADA = 'kn',
  MALAYALAM = 'ml',
  PUNJABI = 'pa',
  ODIA = 'or',
  ASSAMESE = 'as',
  SANSKRIT = 'sa',
  URDU = 'ur',
  KASHMIRI = 'ks',
  SINDHI = 'sd',
  NEPALI = 'ne',
  MANIPURI = 'mni',
  KONKANI = 'kok',
  MAITHILI = 'mai',
  SANTALI = 'sat',
  DOGRI = 'doi',
  BODO = 'brx'
}

/**
 * Model types
 */
export enum ModelType {
  GLM = 'glm',
  LLAMA = 'llama',
  MOE = 'moe'
}

/**
 * Default generation parameters
 */
export interface GenerationParameters {
  /** Default maximum tokens */
  DEFAULT_MAX_TOKENS = 100;
  /** Default temperature */
  DEFAULT_TEMPERATURE = 1.0;
  /** Default top-p */
  DEFAULT_TOP_P = 1.0;
  /** Default top-k */
  DEFAULT_TOP_K = 50;
  /** Default number of beams */
  DEFAULT_NUM_BEAMS = 1;
  /** Default sampling flag */
  DEFAULT_DO_SAMPLE = true;
}

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  GENERATE: '/generate',
  BATCH_GENERATE: '/batch_generate',
  EMBEDDINGS: '/embeddings',
  MODEL_INFO: '/model/info',
  HEALTH: '/health',
  LANGUAGES: '/languages'
} as const;

/**
 * Error codes
 */
export enum ErrorCode {
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  AUTHENTICATION_ERROR = 'AUTHENTICATION_ERROR',
  RATE_LIMIT_ERROR = 'RATE_LIMIT_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  MODEL_NOT_LOADED = 'MODEL_NOT_LOADED',
  INTERNAL_ERROR = 'INTERNAL_ERROR'
}

/**
 * HTTP status codes
 */
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  TIMEOUT: 408,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503
} as const;