package com.bharatai.sdk;

import com.bharatai.sdk.config.ClientConfig;
import com.bharatai.sdk.exceptions.BharatAIException;
import com.bharatai.sdk.http.HttpClient;
import com.bharatai.sdk.models.*;
import com.bharatai.sdk.domains.LanguageAIClient;
import com.bharatai.sdk.domains.GovernanceAIClient;
import com.bharatai.sdk.domains.EducationAIClient;
import com.bharatai.sdk.domains.FinanceAIClient;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Main client class for Bharat Foundation Model Framework Java SDK
 * 
 * This class provides methods to interact with BFMF APIs and integrate
 * Bharat AI capabilities into Java applications.
 */
public class BharatAIClient {
    
    private final HttpClient httpClient;
    private final ClientConfig config;
    private LanguageAIClient languageClient;
    private GovernanceAIClient governanceClient;
    private EducationAIClient educationClient;
    private FinanceAIClient financeClient;
    
    /**
     * Creates a new BFMF client with default configuration
     */
    public BharatAIClient() {
        this(new ClientConfig.Builder().build());
    }
    
    /**
     * Creates a new BFMF client with custom configuration
     * 
     * @param config The client configuration
     */
    public BharatAIClient(ClientConfig config) {
        this.config = config;
        this.httpClient = new HttpClient(config);
        initializeDomainClients();
    }
    
    /**
     * Initialize domain-specific clients
     */
    private void initializeDomainClients() {
        this.languageClient = new LanguageAIClient(this);
        this.governanceClient = new GovernanceAIClient(this);
        this.educationClient = new EducationAIClient(this);
        this.financeClient = new FinanceAIClient(this);
    }
    
    /**
     * Generate text from a prompt
     * 
     * @param request The generation request
     * @return Generation response
     * @throws BharatAIException if the request fails
     */
    public GenerationResponse generateText(GenerationRequest request) throws BharatAIException {
        return httpClient.post("/generate", request, GenerationResponse.class);
    }
    
    /**
     * Generate text from a prompt asynchronously
     * 
     * @param request The generation request
     * @return CompletableFuture containing the generation response
     */
    public CompletableFuture<GenerationResponse> generateTextAsync(GenerationRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return generateText(request);
            } catch (BharatAIException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    /**
     * Generate text for multiple prompts (batch)
     * 
     * @param requests List of generation requests
     * @return Batch generation response
     * @throws BharatAIException if the request fails
     */
    public BatchGenerationResponse generateTextBatch(List<GenerationRequest> requests) throws BharatAIException {
        BatchGenerationRequest batchRequest = new BatchGenerationRequest(requests);
        return httpClient.post("/batch_generate", batchRequest, BatchGenerationResponse.class);
    }
    
    /**
     * Generate text for multiple prompts asynchronously
     * 
     * @param requests List of generation requests
     * @return CompletableFuture containing the batch generation response
     */
    public CompletableFuture<BatchGenerationResponse> generateTextBatchAsync(List<GenerationRequest> requests) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return generateTextBatch(requests);
            } catch (BharatAIException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    /**
     * Get text embeddings
     * 
     * @param request The embedding request
     * @return Embedding response
     * @throws BharatAIException if the request fails
     */
    public EmbeddingResponse getEmbeddings(EmbeddingRequest request) throws BharatAIException {
        return httpClient.post("/embeddings", request, EmbeddingResponse.class);
    }
    
    /**
     * Get text embeddings asynchronously
     * 
     * @param request The embedding request
     * @return CompletableFuture containing the embedding response
     */
    public CompletableFuture<EmbeddingResponse> getEmbeddingsAsync(EmbeddingRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return getEmbeddings(request);
            } catch (BharatAIException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    /**
     * Get model information
     * 
     * @return Model information
     * @throws BharatAIException if the request fails
     */
    public ModelInfo getModelInfo() throws BharatAIException {
        return httpClient.get("/model/info", ModelInfo.class);
    }
    
    /**
     * Get model information asynchronously
     * 
     * @return CompletableFuture containing the model information
     */
    public CompletableFuture<ModelInfo> getModelInfoAsync() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return getModelInfo();
            } catch (BharatAIException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    /**
     * Check API health
     * 
     * @return Health response
     * @throws BharatAIException if the request fails
     */
    public HealthResponse getHealth() throws BharatAIException {
        return httpClient.get("/health", HealthResponse.class);
    }
    
    /**
     * Check API health asynchronously
     * 
     * @return CompletableFuture containing the health response
     */
    public CompletableFuture<HealthResponse> getHealthAsync() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return getHealth();
            } catch (BharatAIException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    /**
     * Get supported languages
     * 
     * @return Supported languages response
     * @throws BharatAIException if the request fails
     */
    public SupportedLanguagesResponse getSupportedLanguages() throws BharatAIException {
        return httpClient.get("/languages", SupportedLanguagesResponse.class);
    }
    
    /**
     * Get supported languages asynchronously
     * 
     * @return CompletableFuture containing the supported languages response
     */
    public CompletableFuture<SupportedLanguagesResponse> getSupportedLanguagesAsync() {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return getSupportedLanguages();
            } catch (BharatAIException e) {
                throw new RuntimeException(e);
            }
        });
    }
    
    /**
     * Get the Language AI client
     * 
     * @return Language AI client
     */
    public LanguageAIClient getLanguageClient() {
        return languageClient;
    }
    
    /**
     * Get the Governance AI client
     * 
     * @return Governance AI client
     */
    public GovernanceAIClient getGovernanceClient() {
        return governanceClient;
    }
    
    /**
     * Get the Education AI client
     * 
     * @return Education AI client
     */
    public EducationAIClient getEducationClient() {
        return educationClient;
    }
    
    /**
     * Get the Finance AI client
     * 
     * @return Finance AI client
     */
    public FinanceAIClient getFinanceClient() {
        return financeClient;
    }
    
    /**
     * Get the HTTP client
     * 
     * @return HTTP client
     */
    HttpClient getHttpClient() {
        return httpClient;
    }
    
    /**
     * Get the client configuration
     * 
     * @return Client configuration
     */
    public ClientConfig getConfig() {
        return config;
    }
    
    /**
     * Update client configuration
     * 
     * @param newConfig New configuration values
     */
    public void updateConfig(ClientConfig newConfig) {
        config.update(newConfig);
        httpClient.updateConfig(config);
    }
    
    /**
     * Close the client and release resources
     */
    public void close() {
        httpClient.close();
    }
    
    /**
     * Get SDK version
     * 
     * @return SDK version
     */
    public static String getVersion() {
        return "1.0.0";
    }
}