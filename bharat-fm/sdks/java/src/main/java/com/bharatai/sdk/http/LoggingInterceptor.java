package com.bharatai.sdk.http;

import okhttp3.Interceptor;
import okhttp3.Request;
import okhttp3.Response;
import okio.Buffer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Interceptor for logging HTTP requests and responses
 */
public class LoggingInterceptor implements Interceptor {
    
    private static final Logger logger = LoggerFactory.getLogger(LoggingInterceptor.class);
    
    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();
        
        // Log request
        logger.debug("Sending request to: {}", request.url());
        logger.debug("Method: {}", request.method());
        logger.debug("Headers: {}", request.headers());
        
        // Log request body if present
        if (request.body() != null) {
            Buffer buffer = new Buffer();
            request.body().writeTo(buffer);
            String requestBody = buffer.readUtf8();
            logger.debug("Request body: {}", requestBody);
        }
        
        long startTime = System.currentTimeMillis();
        
        try {
            Response response = chain.proceed(request);
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            // Log response
            logger.debug("Received response in {} ms", duration);
            logger.debug("Response code: {}", response.code());
            logger.debug("Response headers: {}", response.headers());
            
            // Log response body if present
            if (response.body() != null) {
                String responseBody = response.peekBody(Long.MAX_VALUE).string();
                logger.debug("Response body: {}", responseBody);
            }
            
            return response;
        } catch (IOException e) {
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            logger.error("Request failed after {} ms: {}", duration, e.getMessage());
            throw e;
        }
    }
}