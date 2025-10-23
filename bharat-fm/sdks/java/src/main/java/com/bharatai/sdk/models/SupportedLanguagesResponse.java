package com.bharatai.sdk.models;

import java.util.List;

/**
 * Supported languages response
 */
public class SupportedLanguagesResponse {
    
    private List<String> supportedLanguages;
    private Boolean multilingualEnabled;
    
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
     * Get whether multilingual is enabled
     * 
     * @return Multilingual enabled flag
     */
    public Boolean getMultilingualEnabled() {
        return multilingualEnabled;
    }
    
    /**
     * Set whether multilingual is enabled
     * 
     * @param multilingualEnabled Multilingual enabled flag
     */
    public void setMultilingualEnabled(Boolean multilingualEnabled) {
        this.multilingualEnabled = multilingualEnabled;
    }
}