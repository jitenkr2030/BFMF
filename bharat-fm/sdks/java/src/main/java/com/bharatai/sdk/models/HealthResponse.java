package com.bharatai.sdk.models;

/**
 * Health check response
 */
public class HealthResponse {
    
    private String status;
    private String timestamp;
    private Boolean modelLoaded;
    private String device;
    
    /**
     * Get the health status
     * 
     * @return Status
     */
    public String getStatus() {
        return status;
    }
    
    /**
     * Set the health status
     * 
     * @param status Status
     */
    public void setStatus(String status) {
        this.status = status;
    }
    
    /**
     * Get the timestamp
     * 
     * @return Timestamp
     */
    public String getTimestamp() {
        return timestamp;
    }
    
    /**
     * Set the timestamp
     * 
     * @param timestamp Timestamp
     */
    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }
    
    /**
     * Get whether the model is loaded
     * 
     * @return Model loaded flag
     */
    public Boolean getModelLoaded() {
        return modelLoaded;
    }
    
    /**
     * Set whether the model is loaded
     * 
     * @param modelLoaded Model loaded flag
     */
    public void setModelLoaded(Boolean modelLoaded) {
        this.modelLoaded = modelLoaded;
    }
    
    /**
     * Get the device being used
     * 
     * @return Device
     */
    public String getDevice() {
        return device;
    }
    
    /**
     * Set the device being used
     * 
     * @param device Device
     */
    public void setDevice(String device) {
        this.device = device;
    }
}