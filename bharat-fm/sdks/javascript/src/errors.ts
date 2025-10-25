/**
 * Error classes for Bharat Foundation Model Framework SDK
 */

/**
 * Base error class for BFMF SDK
 */
export class BharatAIError extends Error {
  public readonly code: string;
  public readonly statusCode?: number;
  public readonly details?: any;

  constructor(
    message: string,
    code: string,
    statusCode?: number,
    details?: any
  ) {
    super(message);
    this.name = 'BharatAIError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
    
    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, BharatAIError);
    }
  }

  /**
   * Create error from HTTP response
   */
  static fromResponse(response: any, message?: string): BharatAIError {
    const statusCode = response.status;
    const data = response.data || {};
    
    let code = 'UNKNOWN_ERROR';
    let errorMessage = message || `HTTP ${statusCode}: ${response.statusText}`;
    
    // Map HTTP status codes to error codes
    switch (statusCode) {
      case 400:
        code = 'VALIDATION_ERROR';
        errorMessage = data.message || 'Invalid request parameters';
        break;
      case 401:
        code = 'AUTHENTICATION_ERROR';
        errorMessage = 'Authentication failed';
        break;
      case 403:
        code = 'AUTHENTICATION_ERROR';
        errorMessage = 'Access forbidden';
        break;
      case 404:
        code = 'NOT_FOUND';
        errorMessage = 'Resource not found';
        break;
      case 408:
        code = 'TIMEOUT_ERROR';
        errorMessage = 'Request timeout';
        break;
      case 429:
        code = 'RATE_LIMIT_ERROR';
        errorMessage = 'Rate limit exceeded';
        break;
      case 500:
        code = 'INTERNAL_ERROR';
        errorMessage = 'Internal server error';
        break;
      case 503:
        code = 'MODEL_NOT_LOADED';
        errorMessage = 'Model not loaded or unavailable';
        break;
    }
    
    return new BharatAIError(errorMessage, code, statusCode, data);
  }

  /**
   * Create network error
   */
  static networkError(message: string, details?: any): BharatAIError {
    return new BharatAIError(message, 'NETWORK_ERROR', undefined, details);
  }

  /**
   * Create timeout error
   */
  static timeoutError(message: string = 'Request timeout'): BharatAIError {
    return new BharatAIError(message, 'TIMEOUT_ERROR');
  }

  /**
   * Create validation error
   */
  static validationError(message: string, details?: any): BharatAIError {
    return new BharatAIError(message, 'VALIDATION_ERROR', undefined, details);
  }

  /**
   * Create authentication error
   */
  static authenticationError(message: string = 'Authentication failed'): BharatAIError {
    return new BharatAIError(message, 'AUTHENTICATION_ERROR');
  }

  /**
   * Create rate limit error
   */
  static rateLimitError(message: string = 'Rate limit exceeded'): BharatAIError {
    return new BharatAIError(message, 'RATE_LIMIT_ERROR');
  }

  /**
   * Create model not loaded error
   */
  static modelNotLoadedError(message: string = 'Model not loaded'): BharatAIError {
    return new BharatAIError(message, 'MODEL_NOT_LOADED');
  }

  /**
   * Create internal error
   */
  static internalError(message: string, details?: any): BharatAIError {
    return new BharatAIError(message, 'INTERNAL_ERROR', undefined, details);
  }

  /**
   * Convert error to JSON
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
      details: this.details,
      stack: this.stack
    };
  }
}

/**
 * Validation error for invalid parameters
 */
export class ValidationError extends BharatAIError {
  constructor(message: string, details?: any) {
    super(message, 'VALIDATION_ERROR', undefined, details);
    this.name = 'ValidationError';
  }
}

/**
 * Authentication error for failed authentication
 */
export class AuthenticationError extends BharatAIError {
  constructor(message: string = 'Authentication failed') {
    super(message, 'AUTHENTICATION_ERROR', 401);
    this.name = 'AuthenticationError';
  }
}

/**
 * Rate limit error for too many requests
 */
export class RateLimitError extends BharatAIError {
  constructor(message: string = 'Rate limit exceeded') {
    super(message, 'RATE_LIMIT_ERROR', 429);
    this.name = 'RateLimitError';
  }
}

/**
 * Network error for connection issues
 */
export class NetworkError extends BharatAIError {
  constructor(message: string, details?: any) {
    super(message, 'NETWORK_ERROR', undefined, details);
    this.name = 'NetworkError';
  }
}

/**
 * Timeout error for request timeouts
 */
export class TimeoutError extends BharatAIError {
  constructor(message: string = 'Request timeout') {
    super(message, 'TIMEOUT_ERROR');
    this.name = 'TimeoutError';
  }
}

/**
 * Model not loaded error
 */
export class ModelNotLoadedError extends BharatAIError {
  constructor(message: string = 'Model not loaded') {
    super(message, 'MODEL_NOT_LOADED', 503);
    this.name = 'ModelNotLoadedError';
  }
}