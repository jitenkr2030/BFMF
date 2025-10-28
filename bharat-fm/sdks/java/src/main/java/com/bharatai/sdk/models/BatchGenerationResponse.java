package com.bharatai.sdk.models;

import java.util.List;

/**
 * Response model for batch text generation
 */
public class BatchGenerationResponse {
    
    private List<GenerationResponse> responses;
    
    /**
     * Get the list of generation responses
     * 
     * @return Generation responses
     */
    public List<GenerationResponse> getResponses() {
        return responses;
    }
    
    /**
     * Set the list of generation responses
     * 
     * @param responses Generation responses
     */
    public void setResponses(List<GenerationResponse> responses) {
        this.responses = responses;
    }
}