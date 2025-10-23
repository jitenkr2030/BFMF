package com.bharatai.sdk.models;

import java.util.List;

/**
 * Response model for embeddings
 */
public class EmbeddingResponse {
    
    private List<Double> embeddings;
    private String text;
    private Integer embeddingDim;
    
    /**
     * Get the embedding vector
     * 
     * @return Embeddings
     */
    public List<Double> getEmbeddings() {
        return embeddings;
    }
    
    /**
     * Set the embedding vector
     * 
     * @param embeddings Embeddings
     */
    public void setEmbeddings(List<Double> embeddings) {
        this.embeddings = embeddings;
    }
    
    /**
     * Get the original text
     * 
     * @return Text
     */
    public String getText() {
        return text;
    }
    
    /**
     * Set the original text
     * 
     * @param text Text
     */
    public void setText(String text) {
        this.text = text;
    }
    
    /**
     * Get the embedding dimension
     * 
     * @return Embedding dimension
     */
    public Integer getEmbeddingDim() {
        return embeddingDim;
    }
    
    /**
     * Set the embedding dimension
     * 
     * @param embeddingDim Embedding dimension
     */
    public void setEmbeddingDim(Integer embeddingDim) {
        this.embeddingDim = embeddingDim;
    }
}