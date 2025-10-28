/**
 * Custom error class for Bharat AI SDK
 */

export class BharatAIError extends Error {
  public readonly code: string;
  public readonly statusCode?: number;
  public readonly timestamp: string;
  public readonly retry?: {
    shouldRetry: boolean;
    retryAfter?: number;
    maxRetries?: number;
  };

  constructor(
    code: string,
    message: string,
    cause?: any,
    statusCode?: number,
    retry?: {
      shouldRetry: boolean;
      retryAfter?: number;
      maxRetries?: number;
    }
  ) {
    super(message);
    this.name = 'BharatAIError';
    this.code = code;
    this.statusCode = statusCode;
    this.timestamp = new Date().toISOString();
    this.retry = retry;
    
    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, BharatAIError);
    }
    
    // Set the cause if provided
    if (cause) {
      this.cause = cause;
    }
  }

  /**
   * Check if this error should be retried
   */
  public shouldRetry(): boolean {
    return this.retry?.shouldRetry || false;
  }

  /**
   * Get suggested retry delay
   */
  public getRetryDelay(): number {
    return this.retry?.retryAfter || 1000;
  }

  /**
   * Get maximum retry attempts
   */
  public getMaxRetries(): number {
    return this.retry?.maxRetries || 3;
  }

  /**
   * Convert error to JSON
   */
  public toJSON(): Record<string, any> {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      statusCode: this.statusCode,
      timestamp: this.timestamp,
      retry: this.retry,
      stack: this.stack,
    };
  }

  /**
   * Create network error
   */
  static networkError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'NETWORK_ERROR',
      message,
      cause,
      0,
      { shouldRetry: true, retryAfter: 1000, maxRetries: 3 }
    );
  }

  /**
   * Create timeout error
   */
  static timeoutError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'TIMEOUT_ERROR',
      message,
      cause,
      0,
      { shouldRetry: true, retryAfter: 2000, maxRetries: 2 }
    );
  }

  /**
   * Create authentication error
   */
  static authenticationError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'AUTHENTICATION_ERROR',
      message,
      cause,
      401,
      { shouldRetry: false }
    );
  }

  /**
   * Create authorization error
   */
  static authorizationError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'AUTHORIZATION_ERROR',
      message,
      cause,
      403,
      { shouldRetry: false }
    );
  }

  /**
   * Create rate limit error
   */
  static rateLimitError(message: string, retryAfter?: number, cause?: any): BharatAIError {
    return new BharatAIError(
      'RATE_LIMIT_ERROR',
      message,
      cause,
      429,
      { shouldRetry: true, retryAfter: retryAfter || 5000, maxRetries: 1 }
    );
  }

  /**
   * Create validation error
   */
  static validationError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'VALIDATION_ERROR',
      message,
      cause,
      400,
      { shouldRetry: false }
    );
  }

  /**
   * Create server error
   */
  static serverError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'SERVER_ERROR',
      message,
      cause,
      500,
      { shouldRetry: true, retryAfter: 3000, maxRetries: 2 }
    );
  }

  /**
   * Create not found error
   */
  static notFoundError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'NOT_FOUND_ERROR',
      message,
      cause,
      404,
      { shouldRetry: false }
    );
  }

  /**
   * Create initialization error
   */
  static initializationError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'INITIALIZATION_FAILED',
      message,
      cause,
      0,
      { shouldRetry: false }
    );
  }

  /**
   * Create cache error
   */
  static cacheError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'CACHE_ERROR',
      message,
      cause,
      0,
      { shouldRetry: false }
    );
  }

  /**
   * Create configuration error
   */
  static configurationError(message: string, cause?: any): BharatAIError {
    return new BharatAIError(
      'CONFIGURATION_ERROR',
      message,
      cause,
      0,
      { shouldRetry: false }
    );
  }
}