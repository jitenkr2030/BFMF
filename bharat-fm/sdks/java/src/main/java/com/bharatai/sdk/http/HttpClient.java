package com.bharatai.sdk.http;

import com.bharatai.sdk.config.ClientConfig;
import com.bharatai.sdk.exceptions.BharatAIException;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * HTTP client for making requests to BFMF API
 */
public class HttpClient {
    
    private final OkHttpClient client;
    private final ObjectMapper objectMapper;
    private final ClientConfig config;
    
    /**
     * Create a new HTTP client
     * 
     * @param config Client configuration
     */
    public HttpClient(ClientConfig config) {
        this.config = config;
        this.objectMapper = new ObjectMapper();
        
        OkHttpClient.Builder builder = new OkHttpClient.Builder()
            .connectTimeout(config.getTimeout(), TimeUnit.MILLISECONDS)
            .readTimeout(config.getTimeout(), TimeUnit.MILLISECONDS)
            .writeTimeout(config.getTimeout(), TimeUnit.MILLISECONDS)
            .addInterceptor(new RetryInterceptor(config.getMaxRetries(), config.getRetryDelay()));
        
        if (config.isDebug()) {
            builder.addInterceptor(new LoggingInterceptor());
        }
        
        this.client = builder.build();
    }
    
    /**
     * Make a GET request
     * 
     * @param endpoint API endpoint
     * @param responseType Response type class
     * @param <T> Response type
     * @return Response object
     * @throws BharatAIException if the request fails
     */
    public <T> T get(String endpoint, Class<T> responseType) throws BharatAIException {
        Request request = new Request.Builder()
            .url(config.getBaseURL() + endpoint)
            .addHeader("User-Agent", "Bharat-Java-SDK/1.0.0")
            .addHeader("Content-Type", "application/json")
            .get()
            .build();
        
        return executeRequest(request, responseType);
    }
    
    /**
     * Make a GET request with query parameters
     * 
     * @param endpoint API endpoint
     * @param queryParams Query parameters
     * @param responseType Response type class
     * @param <T> Response type
     * @return Response object
     * @throws BharatAIException if the request fails
     */
    public <T> T get(String endpoint, Map<String, String> queryParams, Class<T> responseType) throws BharatAIException {
        HttpUrl.Builder urlBuilder = HttpUrl.parse(config.getBaseURL() + endpoint).newBuilder();
        
        if (queryParams != null) {
            for (Map.Entry<String, String> entry : queryParams.entrySet()) {
                urlBuilder.addQueryParameter(entry.getKey(), entry.getValue());
            }
        }
        
        Request request = new Request.Builder()
            .url(urlBuilder.build())
            .addHeader("User-Agent", "Bharat-Java-SDK/1.0.0")
            .addHeader("Content-Type", "application/json")
            .get()
            .build();
        
        return executeRequest(request, responseType);
    }
    
    /**
     * Make a POST request
     * 
     * @param endpoint API endpoint
     * @param requestBody Request body object
     * @param responseType Response type class
     * @param <T> Response type
     * @return Response object
     * @throws BharatAIException if the request fails
     */
    public <T> T post(String endpoint, Object requestBody, Class<T> responseType) throws BharatAIException {
        try {
            String jsonBody = objectMapper.writeValueAsString(requestBody);
            
            RequestBody body = RequestBody.create(
                jsonBody,
                MediaType.parse("application/json")
            );
            
            Request.Builder requestBuilder = new Request.Builder()
                .url(config.getBaseURL() + endpoint)
                .addHeader("User-Agent", "Bharat-Java-SDK/1.0.0")
                .addHeader("Content-Type", "application/json")
                .post(body);
            
            if (config.getApiKey() != null) {
                requestBuilder.addHeader("Authorization", "Bearer " + config.getApiKey());
            }
            
            Request request = requestBuilder.build();
            
            return executeRequest(request, responseType);
        } catch (Exception e) {
            throw new BharatAIException("Failed to serialize request body", "SERIALIZATION_ERROR", 0, e);
        }
    }
    
    /**
     * Make a PUT request
     * 
     * @param endpoint API endpoint
     * @param requestBody Request body object
     * @param responseType Response type class
     * @param <T> Response type
     * @return Response object
     * @throws BharatAIException if the request fails
     */
    public <T> T put(String endpoint, Object requestBody, Class<T> responseType) throws BharatAIException {
        try {
            String jsonBody = objectMapper.writeValueAsString(requestBody);
            
            RequestBody body = RequestBody.create(
                jsonBody,
                MediaType.parse("application/json")
            );
            
            Request.Builder requestBuilder = new Request.Builder()
                .url(config.getBaseURL() + endpoint)
                .addHeader("User-Agent", "Bharat-Java-SDK/1.0.0")
                .addHeader("Content-Type", "application/json")
                .put(body);
            
            if (config.getApiKey() != null) {
                requestBuilder.addHeader("Authorization", "Bearer " + config.getApiKey());
            }
            
            Request request = requestBuilder.build();
            
            return executeRequest(request, responseType);
        } catch (Exception e) {
            throw new BharatAIException("Failed to serialize request body", "SERIALIZATION_ERROR", 0, e);
        }
    }
    
    /**
     * Make a DELETE request
     * 
     * @param endpoint API endpoint
     * @param responseType Response type class
     * @param <T> Response type
     * @return Response object
     * @throws BharatAIException if the request fails
     */
    public <T> T delete(String endpoint, Class<T> responseType) throws BharatAIException {
        Request.Builder requestBuilder = new Request.Builder()
            .url(config.getBaseURL() + endpoint)
            .addHeader("User-Agent", "Bharat-Java-SDK/1.0.0")
            .addHeader("Content-Type", "application/json")
            .delete();
        
        if (config.getApiKey() != null) {
            requestBuilder.addHeader("Authorization", "Bearer " + config.getApiKey());
        }
        
        Request request = requestBuilder.build();
        
        return executeRequest(request, responseType);
    }
    
    /**
     * Execute HTTP request and handle response
     * 
     * @param request HTTP request
     * @param responseType Response type class
     * @param <T> Response type
     * @return Response object
     * @throws BharatAIException if the request fails
     */
    private <T> T executeRequest(Request request, Class<T> responseType) throws BharatAIException {
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String errorMessage = response.body() != null ? response.body().string() : "Request failed";
                throw BharatAIException.fromStatusCode(response.code(), errorMessage);
            }
            
            if (response.body() == null) {
                throw new BharatAIException("Empty response body", "EMPTY_RESPONSE", response.code());
            }
            
            String responseBody = response.body().string();
            
            try {
                return objectMapper.readValue(responseBody, responseType);
            } catch (Exception e) {
                throw new BharatAIException("Failed to parse response", "PARSE_ERROR", response.code(), e);
            }
        } catch (IOException e) {
            if (e.getMessage() != null && e.getMessage().contains("timeout")) {
                throw BharatAIException.timeoutError("Request timeout: " + e.getMessage());
            }
            throw BharatAIException.networkError("Network error: " + e.getMessage());
        }
    }
    
    /**
     * Update client configuration
     * 
     * @param newConfig New configuration
     */
    public void updateConfig(ClientConfig newConfig) {
        // In a real implementation, you might want to recreate the client
        // or update the existing client's configuration
    }
    
    /**
     * Close the HTTP client
     */
    public void close() {
        client.dispatcher().executorService().shutdown();
        client.connectionPool().evictAll();
    }
    
    /**
     * Get the underlying OkHttp client
     * 
     * @return OkHttp client
     */
    OkHttpClient getOkHttpClient() {
        return client;
    }
}