package com.bharatai.sdk.http;

import okhttp3.Interceptor;
import okhttp3.Request;
import okhttp3.Response;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Interceptor for retrying failed requests
 */
public class RetryInterceptor implements Interceptor {
    
    private final int maxRetries;
    private final long retryDelay;
    
    /**
     * Create a new retry interceptor
     * 
     * @param maxRetries Maximum number of retries
     * @param retryDelay Delay between retries in milliseconds
     */
    public RetryInterceptor(int maxRetries, long retryDelay) {
        this.maxRetries = maxRetries;
        this.retryDelay = retryDelay;
    }
    
    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();
        Response response = null;
        IOException lastException = null;
        
        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                response = chain.proceed(request);
                
                // Check if response is successful
                if (response.isSuccessful()) {
                    return response;
                }
                
                // Check if we should retry based on status code
                if (shouldRetry(response.code())) {
                    if (attempt < maxRetries) {
                        response.close();
                        Thread.sleep(retryDelay);
                        continue;
                    }
                }
                
                return response;
            } catch (IOException e) {
                lastException = e;
                
                if (attempt < maxRetries && shouldRetry(e)) {
                    try {
                        Thread.sleep(retryDelay);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new IOException("Retry interrupted", ie);
                    }
                    continue;
                }
                
                throw e;
            }
        }
        
        // This should not be reached, but just in case
        if (lastException != null) {
            throw lastException;
        }
        
        return response;
    }
    
    /**
     * Check if request should be retried based on status code
     * 
     * @param statusCode HTTP status code
     * @return True if should retry
     */
    private boolean shouldRetry(int statusCode) {
        return statusCode == 408 || // Request timeout
               statusCode == 429 || // Too many requests
               statusCode == 500 || // Internal server error
               statusCode == 502 || // Bad gateway
               statusCode == 503 || // Service unavailable
               statusCode == 504;   // Gateway timeout
    }
    
    /**
     * Check if request should be retried based on exception
     * 
     * @param exception IOException
     * @return True if should retry
     */
    private boolean shouldRetry(IOException exception) {
        return exception.getMessage() != null && 
               (exception.getMessage().contains("timeout") ||
                exception.getMessage().contains("Connection reset") ||
                exception.getMessage().contains("Connection refused") ||
                exception.getMessage().contains("Connection timed out"));
    }
}