package com.bharatai.sdk.models;

import java.util.List;

/**
 * Request model for batch text generation
 */
public class BatchGenerationRequest {
    
    private List<GenerationRequest> requests;
    
    /**
     * Create a new batch generation request
     */
    public BatchGenerationRequest() {}
    
    /**
     * Create a new batch generation request
     * 
     * @param requests List of generation requests
     */
    public BatchGenerationRequest(List<GenerationRequest> requests) {
        this.requests = requests;
    }
    
    /**
     * Get the list of generation requests
     * 
     * @return Generation requests
     */
    public List<GenerationRequest> getRequests() {
        return requests;
    }
    
    /**
     * Set the list of generation requests
     * 
     * @param requests Generation requests
     */
    public void setRequests(List<GenerationRequest> requests) {
        this.requests = requests;
    }
}