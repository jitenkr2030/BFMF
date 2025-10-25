package com.bharatai.sdk.models;

/**
 * Request model for text generation
 */
public class GenerationRequest {
    
    private String prompt;
    private Integer maxTokens;
    private Double temperature;
    private Double topP;
    private Integer topK;
    private Integer numBeams;
    private Boolean doSample;
    private String language;
    
    /**
     * Create a new generation request
     */
    public GenerationRequest() {}
    
    /**
     * Create a new generation request with prompt
     * 
     * @param prompt Input prompt for text generation
     */
    public GenerationRequest(String prompt) {
        this.prompt = prompt;
    }
    
    /**
     * Get the input prompt
     * 
     * @return Prompt
     */
    public String getPrompt() {
        return prompt;
    }
    
    /**
     * Set the input prompt
     * 
     * @param prompt Input prompt
     */
    public void setPrompt(String prompt) {
        this.prompt = prompt;
    }
    
    /**
     * Get the maximum number of tokens to generate
     * 
     * @return Maximum tokens
     */
    public Integer getMaxTokens() {
        return maxTokens;
    }
    
    /**
     * Set the maximum number of tokens to generate
     * 
     * @param maxTokens Maximum tokens
     */
    public void setMaxTokens(Integer maxTokens) {
        this.maxTokens = maxTokens;
    }
    
    /**
     * Get the sampling temperature
     * 
     * @return Temperature
     */
    public Double getTemperature() {
        return temperature;
    }
    
    /**
     * Set the sampling temperature
     * 
     * @param temperature Temperature
     */
    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }
    
    /**
     * Get the top-p sampling parameter
     * 
     * @return Top-p
     */
    public Double getTopP() {
        return topP;
    }
    
    /**
     * Set the top-p sampling parameter
     * 
     * @param topP Top-p
     */
    public void setTopP(Double topP) {
        this.topP = topP;
    }
    
    /**
     * Get the top-k sampling parameter
     * 
     * @return Top-k
     */
    public Integer getTopK() {
        return topK;
    }
    
    /**
     * Set the top-k sampling parameter
     * 
     * @param topK Top-k
     */
    public void setTopK(Integer topK) {
        this.topK = topK;
    }
    
    /**
     * Get the number of beams for beam search
     * 
     * @return Number of beams
     */
    public Integer getNumBeams() {
        return numBeams;
    }
    
    /**
     * Set the number of beams for beam search
     * 
     * @param numBeams Number of beams
     */
    public void setNumBeams(Integer numBeams) {
        this.numBeams = numBeams;
    }
    
    /**
     * Get whether to use sampling
     * 
     * @return Do sample flag
     */
    public Boolean getDoSample() {
        return doSample;
    }
    
    /**
     * Set whether to use sampling
     * 
     * @param doSample Do sample flag
     */
    public void setDoSample(Boolean doSample) {
        this.doSample = doSample;
    }
    
    /**
     * Get the language hint for generation
     * 
     * @return Language
     */
    public String getLanguage() {
        return language;
    }
    
    /**
     * Set the language hint for generation
     * 
     * @param language Language
     */
    public void setLanguage(String language) {
        this.language = language;
    }
    
    /**
     * Builder class for GenerationRequest
     */
    public static class Builder {
        private String prompt;
        private Integer maxTokens = 100;
        private Double temperature = 1.0;
        private Double topP = 1.0;
        private Integer topK = 50;
        private Integer numBeams = 1;
        private Boolean doSample = true;
        private String language;
        
        /**
         * Set the prompt
         * 
         * @param prompt Prompt
         * @return Builder
         */
        public Builder prompt(String prompt) {
            this.prompt = prompt;
            return this;
        }
        
        /**
         * Set the maximum tokens
         * 
         * @param maxTokens Maximum tokens
         * @return Builder
         */
        public Builder maxTokens(Integer maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }
        
        /**
         * Set the temperature
         * 
         * @param temperature Temperature
         * @return Builder
         */
        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }
        
        /**
         * Set the top-p
         * 
         * @param topP Top-p
         * @return Builder
         */
        public Builder topP(Double topP) {
            this.topP = topP;
            return this;
        }
        
        /**
         * Set the top-k
         * 
         * @param topK Top-k
         * @return Builder
         */
        public Builder topK(Integer topK) {
            this.topK = topK;
            return this;
        }
        
        /**
         * Set the number of beams
         * 
         * @param numBeams Number of beams
         * @return Builder
         */
        public Builder numBeams(Integer numBeams) {
            this.numBeams = numBeams;
            return this;
        }
        
        /**
         * Set the do sample flag
         * 
         * @param doSample Do sample flag
         * @return Builder
         */
        public Builder doSample(Boolean doSample) {
            this.doSample = doSample;
            return this;
        }
        
        /**
         * Set the language
         * 
         * @param language Language
         * @return Builder
         */
        public Builder language(String language) {
            this.language = language;
            return this;
        }
        
        /**
         * Build the GenerationRequest
         * 
         * @return GenerationRequest
         */
        public GenerationRequest build() {
            GenerationRequest request = new GenerationRequest();
            request.setPrompt(this.prompt);
            request.setMaxTokens(this.maxTokens);
            request.setTemperature(this.temperature);
            request.setTopP(this.topP);
            request.setTopK(this.topK);
            request.setNumBeams(this.numBeams);
            request.setDoSample(this.doSample);
            request.setLanguage(this.language);
            return request;
        }
    }
}