package com.bharatai.sdk.models;

import java.util.List;

/**
 * Model information response
 */
public class ModelInfo {
    
    private String modelName;
    private String modelType;
    private String modelSize;
    private List<String> supportedLanguages;
    private Integer maxContextLength;
    private String version;
    
    /**
     * Get the model name
     * 
     * @return Model name
     */
    public String getModelName() {
        return modelName;
    }
    
    /**
     * Set the model name
     * 
     * @param modelName Model name
     */
    public void setModelName(String modelName) {
        this.modelName = modelName;
    }
    
    /**
     * Get the model type
     * 
     * @return Model type
     */
    public String getModelType() {
        return modelType;
    }
    
    /**
     * Set the model type
     * 
     * @param modelType Model type
     */
    public void setModelType(String modelType) {
        this.modelType = modelType;
    }
    
    /**
     * Get the model size
     * 
     * @return Model size
     */
    public String getModelSize() {
        return modelSize;
    }
    
    /**
     * Set the model size
     * 
     * @param modelSize Model size
     */
    public void setModelSize(String modelSize) {
        this.modelSize = modelSize;
    }
    
    /**
     * Get the supported languages
     * 
     * @return Supported languages
     */
    public List<String> getSupportedLanguages() {
        return supportedLanguages;
    }
    
    /**
     * Set the supported languages
     * 
     * @param supportedLanguages Supported languages
     */
    public void setSupportedLanguages(List<String> supportedLanguages) {
        this.supportedLanguages = supportedLanguages;
    }
    
    /**
     * Get the maximum context length
     * 
     * @return Maximum context length
     */
    public Integer getMaxContextLength() {
        return maxContextLength;
    }
    
    /**
     * Set the maximum context length
     * 
     * @param maxContextLength Maximum context length
     */
    public void setMaxContextLength(Integer maxContextLength) {
        this.maxContextLength = maxContextLength;
    }
    
    /**
     * Get the model version
     * 
     * @return Model version
     */
    public String getVersion() {
        return version;
    }
    
    /**
     * Set the model version
     * 
     * @param version Model version
     */
    public void setVersion(String version) {
        this.version = version;
    }
}