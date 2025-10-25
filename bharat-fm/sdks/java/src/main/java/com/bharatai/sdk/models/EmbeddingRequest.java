package com.bharatai.sdk.models;

/**
 * Request model for text embeddings
 */
public class EmbeddingRequest {
    
    private String text;
    private Boolean normalize;
    
    /**
     * Create a new embedding request
     */
    public EmbeddingRequest() {}
    
    /**
     * Create a new embedding request
     * 
     * @param text Input text for embedding
     */
    public EmbeddingRequest(String text) {
        this.text = text;
    }
    
    /**
     * Get the input text
     * 
     * @return Text
     */
    public String getText() {
        return text;
    }
    
    /**
     * Set the input text
     * 
     * @param text Text
     */
    public void setText(String text) {
        this.text = text;
    }
    
    /**
     * Get whether to normalize embeddings
     * 
     * @return Normalize flag
     */
    public Boolean getNormalize() {
        return normalize;
    }
    
    /**
     * Set whether to normalize embeddings
     * 
     * @param normalize Normalize flag
     */
    public void setNormalize(Boolean normalize) {
        this.normalize = normalize;
    }
    
    /**
     * Builder class for EmbeddingRequest
     */
    public static class Builder {
        private String text;
        private Boolean normalize = true;
        
        /**
         * Set the text
         * 
         * @param text Text
         * @return Builder
         */
        public Builder text(String text) {
            this.text = text;
            return this;
        }
        
        /**
         * Set the normalize flag
         * 
         * @param normalize Normalize flag
         * @return Builder
         */
        public Builder normalize(Boolean normalize) {
            this.normalize = normalize;
            return this;
        }
        
        /**
         * Build the EmbeddingRequest
         * 
         * @return EmbeddingRequest
         */
        public EmbeddingRequest build() {
            EmbeddingRequest request = new EmbeddingRequest();
            request.setText(this.text);
            request.setNormalize(this.normalize);
            return request;
        }
    }
}