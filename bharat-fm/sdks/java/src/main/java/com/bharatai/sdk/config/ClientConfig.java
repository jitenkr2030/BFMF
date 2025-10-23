package com.bharatai.sdk.config;

/**
 * Configuration class for Bharat AI client
 */
public class ClientConfig {
    
    private final String baseURL;
    private final String apiKey;
    private final int timeout;
    private final int maxRetries;
    private final long retryDelay;
    private final boolean debug;
    
    private ClientConfig(Builder builder) {
        this.baseURL = builder.baseURL;
        this.apiKey = builder.apiKey;
        this.timeout = builder.timeout;
        this.maxRetries = builder.maxRetries;
        this.retryDelay = builder.retryDelay;
        this.debug = builder.debug;
    }
    
    /**
     * Get the base URL for the API
     * 
     * @return Base URL
     */
    public String getBaseURL() {
        return baseURL;
    }
    
    /**
     * Get the API key for authentication
     * 
     * @return API key
     */
    public String getApiKey() {
        return apiKey;
    }
    
    /**
     * Get the request timeout in milliseconds
     * 
     * @return Timeout in milliseconds
     */
    public int getTimeout() {
        return timeout;
    }
    
    /**
     * Get the maximum number of retries
     * 
     * @return Maximum retries
     */
    public int getMaxRetries() {
        return maxRetries;
    }
    
    /**
     * Get the delay between retries in milliseconds
     * 
     * @return Retry delay in milliseconds
     */
    public long getRetryDelay() {
        return retryDelay;
    }
    
    /**
     * Get whether debug logging is enabled
     * 
     * @return Debug enabled flag
     */
    public boolean isDebug() {
        return debug;
    }
    
    /**
     * Update configuration with new values
     * 
     * @param newConfig New configuration values
     */
    public void update(ClientConfig newConfig) {
        // This method would update the configuration
        // In a real implementation, you might want to make this class mutable
        // or create a new instance
    }
    
    /**
     * Builder class for ClientConfig
     */
    public static class Builder {
        private String baseURL = "http://localhost:8000";
        private String apiKey;
        private int timeout = 30000;
        private int maxRetries = 3;
        private long retryDelay = 1000;
        private boolean debug = false;
        
        /**
         * Set the base URL for the API
         * 
         * @param baseURL Base URL
         * @return Builder instance
         */
        public Builder baseURL(String baseURL) {
            this.baseURL = baseURL;
            return this;
        }
        
        /**
         * Set the API key for authentication
         * 
         * @param apiKey API key
         * @return Builder instance
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }
        
        /**
         * Set the request timeout in milliseconds
         * 
         * @param timeout Timeout in milliseconds
         * @return Builder instance
         */
        public Builder timeout(int timeout) {
            this.timeout = timeout;
            return this;
        }
        
        /**
         * Set the maximum number of retries
         * 
         * @param maxRetries Maximum retries
         * @return Builder instance
         */
        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }
        
        /**
         * Set the delay between retries in milliseconds
         * 
         * @param retryDelay Retry delay in milliseconds
         * @return Builder instance
         */
        public Builder retryDelay(long retryDelay) {
            this.retryDelay = retryDelay;
            return this;
        }
        
        /**
         * Enable or disable debug logging
         * 
         * @param debug Debug enabled flag
         * @return Builder instance
         */
        public Builder debug(boolean debug) {
            this.debug = debug;
            return this;
        }
        
        /**
         * Build the ClientConfig instance
         * 
         * @return ClientConfig instance
         */
        public ClientConfig build() {
            return new ClientConfig(this);
        }
    }
}