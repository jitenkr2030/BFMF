package com.bharatai.sdk.models;

/**
 * Response model for text generation
 */
public class GenerationResponse {
    
    private String generatedText;
    private String prompt;
    private Integer tokensGenerated;
    private Double generationTime;
    private String languageDetected;
    
    /**
     * Get the generated text
     * 
     * @return Generated text
     */
    public String getGeneratedText() {
        return generatedText;
    }
    
    /**
     * Set the generated text
     * 
     * @param generatedText Generated text
     */
    public void setGeneratedText(String generatedText) {
        this.generatedText = generatedText;
    }
    
    /**
     * Get the original prompt
     * 
     * @return Original prompt
     */
    public String getPrompt() {
        return prompt;
    }
    
    /**
     * Set the original prompt
     * 
     * @param prompt Original prompt
     */
    public void setPrompt(String prompt) {
        this.prompt = prompt;
    }
    
    /**
     * Get the number of tokens generated
     * 
     * @return Tokens generated
     */
    public Integer getTokensGenerated() {
        return tokensGenerated;
    }
    
    /**
     * Set the number of tokens generated
     * 
     * @param tokensGenerated Tokens generated
     */
    public void setTokensGenerated(Integer tokensGenerated) {
        this.tokensGenerated = tokensGenerated;
    }
    
    /**
     * Get the generation time in seconds
     * 
     * @return Generation time
     */
    public Double getGenerationTime() {
        return generationTime;
    }
    
    /**
     * Set the generation time in seconds
     * 
     * @param generationTime Generation time
     */
    public void setGenerationTime(Double generationTime) {
        this.generationTime = generationTime;
    }
    
    /**
     * Get the detected language
     * 
     * @return Detected language
     */
    public String getLanguageDetected() {
        return languageDetected;
    }
    
    /**
     * Set the detected language
     * 
     * @param languageDetected Detected language
     */
    public void setLanguageDetected(String languageDetected) {
        this.languageDetected = languageDetected;
    }
    
    @Override
    public String toString() {
        return "GenerationResponse{" +
                "generatedText='" + generatedText + '\'' +
                ", prompt='" + prompt + '\'' +
                ", tokensGenerated=" + tokensGenerated +
                ", generationTime=" + generationTime +
                ", languageDetected='" + languageDetected + '\'' +
                '}';
    }
}