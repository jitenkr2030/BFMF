package com.bharatai.sdk.exceptions;

/**
 * Base exception class for Bharat AI SDK
 */
public class BharatAIException extends Exception {
    
    private final String code;
    private final int statusCode;
    
    /**
     * Create a new BharatAIException
     * 
     * @param message Error message
     * @param code Error code
     * @param statusCode HTTP status code
     */
    public BharatAIException(String message, String code, int statusCode) {
        super(message);
        this.code = code;
        this.statusCode = statusCode;
    }
    
    /**
     * Create a new BharatAIException
     * 
     * @param message Error message
     * @param code Error code
     * @param statusCode HTTP status code
     * @param cause Root cause
     */
    public BharatAIException(String message, String code, int statusCode, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.statusCode = statusCode;
    }
    
    /**
     * Get the error code
     * 
     * @return Error code
     */
    public String getCode() {
        return code;
    }
    
    /**
     * Get the HTTP status code
     * 
     * @return HTTP status code
     */
    public int getStatusCode() {
        return statusCode;
    }
    
    /**
     * Create exception from HTTP status code
     * 
     * @param statusCode HTTP status code
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException fromStatusCode(int statusCode, String message) {
        String code = "UNKNOWN_ERROR";
        
        switch (statusCode) {
            case 400:
                code = "VALIDATION_ERROR";
                break;
            case 401:
                code = "AUTHENTICATION_ERROR";
                break;
            case 403:
                code = "AUTHENTICATION_ERROR";
                break;
            case 404:
                code = "NOT_FOUND";
                break;
            case 408:
                code = "TIMEOUT_ERROR";
                break;
            case 429:
                code = "RATE_LIMIT_ERROR";
                break;
            case 500:
                code = "INTERNAL_ERROR";
                break;
            case 503:
                code = "MODEL_NOT_LOADED";
                break;
        }
        
        return new BharatAIException(message, code, statusCode);
    }
    
    /**
     * Create network error
     * 
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException networkError(String message) {
        return new BharatAIException(message, "NETWORK_ERROR", 0);
    }
    
    /**
     * Create timeout error
     * 
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException timeoutError(String message) {
        return new BharatAIException(message, "TIMEOUT_ERROR", 408);
    }
    
    /**
     * Create validation error
     * 
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException validationError(String message) {
        return new BharatAIException(message, "VALIDATION_ERROR", 400);
    }
    
    /**
     * Create authentication error
     * 
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException authenticationError(String message) {
        return new BharatAIException(message, "AUTHENTICATION_ERROR", 401);
    }
    
    /**
     * Create rate limit error
     * 
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException rateLimitError(String message) {
        return new BharatAIException(message, "RATE_LIMIT_ERROR", 429);
    }
    
    /**
     * Create model not loaded error
     * 
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException modelNotLoadedError(String message) {
        return new BharatAIException(message, "MODEL_NOT_LOADED", 503);
    }
    
    /**
     * Create internal error
     * 
     * @param message Error message
     * @return BharatAIException
     */
    public static BharatAIException internalError(String message) {
        return new BharatAIException(message, "INTERNAL_ERROR", 500);
    }
}