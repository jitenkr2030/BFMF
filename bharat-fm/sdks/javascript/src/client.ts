/**
 * Main client class for Bharat Foundation Model Framework SDK
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { 
  GenerationRequest, 
  GenerationResponse, 
  BatchGenerationRequest,
  BatchGenerationResponse,
  EmbeddingRequest,
  EmbeddingResponse,
  ModelInfo,
  HealthResponse,
  SupportedLanguagesResponse,
  ClientConfig
} from './types';
import { API_ENDPOINTS } from './constants';
import { BharatAIError, NetworkError, TimeoutError } from './errors';

/**
 * Main client for interacting with BFMF API
 */
export class BharatAIClient {
  private client: AxiosInstance;
  private config: Required<ClientConfig>;

  /**
   * Create a new BFMF client instance
   */
  constructor(config: ClientConfig = {}) {
    this.config = {
      baseURL: config.baseURL || 'http://localhost:8000',
      apiKey: config.apiKey,
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      retryDelay: config.retryDelay || 1000,
      debug: config.debug || false
    };

    // Create axios instance
    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': `@bharat-ai/sdk/${this.getVersion()}`
      }
    });

    // Add authentication header if API key is provided
    if (this.config.apiKey) {
      this.client.defaults.headers.common['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    // Add request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        if (this.config.debug) {
          console.log(`[BFMF SDK] Request: ${config.method?.toUpperCase()} ${config.url}`, config.data);
        }
        return config;
      },
      (error) => {
        if (this.config.debug) {
          console.error('[BFMF SDK] Request Error:', error);
        }
        return Promise.reject(error);
      }
    );

    // Add response interceptor for error handling and logging
    this.client.interceptors.response.use(
      (response) => {
        if (this.config.debug) {
          console.log(`[BFMF SDK] Response: ${response.status} ${response.config.url}`, response.data);
        }
        return response;
      },
      (error: AxiosError) => {
        if (this.config.debug) {
          console.error('[BFMF SDK] Response Error:', error);
        }
        return Promise.reject(this.handleError(error));
      }
    );
  }

  /**
   * Get SDK version
   */
  private getVersion(): string {
    return '1.0.0';
  }

  /**
   * Handle API errors
   */
  private handleError(error: AxiosError): BharatAIError {
    if (error.response) {
      // Server responded with error status
      return BharatAIError.fromResponse(error.response);
    } else if (error.request) {
      // Request was made but no response received
      if (error.code === 'ECONNABORTED') {
        return new TimeoutError('Request timeout');
      }
      return new NetworkError('Network error', error);
    } else {
      // Something else happened
      return new BharatAIError(error.message || 'Unknown error', 'UNKNOWN_ERROR');
    }
  }

  /**
   * Make request with retry logic
   */
  private async makeRequest<T>(
    config: AxiosRequestConfig,
    retries: number = this.config.maxRetries
  ): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.client.request(config);
      return response.data;
    } catch (error) {
      if (retries > 0 && this.shouldRetry(error as BharatAIError)) {
        await this.delay(this.config.retryDelay);
        return this.makeRequest(config, retries - 1);
      }
      throw error;
    }
  }

  /**
   * Check if request should be retried
   */
  private shouldRetry(error: BharatAIError): boolean {
    return error.code === 'NETWORK_ERROR' || 
           error.code === 'TIMEOUT_ERROR' ||
           error.statusCode === 429 || // Rate limit
           error.statusCode === 503;   // Service unavailable
  }

  /**
   * Delay execution
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generate text from prompt
   */
  async generateText(request: GenerationRequest): Promise<GenerationResponse> {
    return this.makeRequest<GenerationResponse>({
      method: 'POST',
      url: API_ENDPOINTS.GENERATE,
      data: {
        prompt: request.prompt,
        max_tokens: request.maxTokens || 100,
        temperature: request.temperature || 1.0,
        top_p: request.topP || 1.0,
        top_k: request.topK || 50,
        num_beams: request.numBeams || 1,
        do_sample: request.doSample !== false,
        language: request.language
      }
    });
  }

  /**
   * Generate text for multiple prompts (batch)
   */
  async generateTextBatch(request: BatchGenerationRequest): Promise<BatchGenerationResponse> {
    const formattedRequests = request.requests.map(req => ({
      prompt: req.prompt,
      max_tokens: req.maxTokens || 100,
      temperature: req.temperature || 1.0,
      top_p: req.topP || 1.0,
      top_k: req.topK || 50,
      num_beams: req.numBeams || 1,
      do_sample: req.doSample !== false,
      language: req.language
    }));

    return this.makeRequest<BatchGenerationResponse>({
      method: 'POST',
      url: API_ENDPOINTS.BATCH_GENERATE,
      data: formattedRequests
    });
  }

  /**
   * Get text embeddings
   */
  async getEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    return this.makeRequest<EmbeddingResponse>({
      method: 'POST',
      url: API_ENDPOINTS.EMBEDDINGS,
      data: {
        text: request.text,
        normalize: request.normalize !== false
      }
    });
  }

  /**
   * Get model information
   */
  async getModelInfo(): Promise<ModelInfo> {
    return this.makeRequest<ModelInfo>({
      method: 'GET',
      url: API_ENDPOINTS.MODEL_INFO
    });
  }

  /**
   * Check API health
   */
  async getHealth(): Promise<HealthResponse> {
    return this.makeRequest<HealthResponse>({
      method: 'GET',
      url: API_ENDPOINTS.HEALTH
    });
  }

  /**
   * Get supported languages
   */
  async getSupportedLanguages(): Promise<SupportedLanguagesResponse> {
    return this.makeRequest<SupportedLanguagesResponse>({
      method: 'GET',
      url: API_ENDPOINTS.LANGUAGES
    });
  }

  /**
   * Stream text generation
   */
  async *generateTextStream(request: GenerationRequest): AsyncGenerator<string, void, unknown> {
    const response = await fetch(`${this.config.baseURL}${API_ENDPOINTS.GENERATE}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': this.config.apiKey ? `Bearer ${this.config.apiKey}` : undefined,
      },
      body: JSON.stringify({
        prompt: request.prompt,
        max_tokens: request.maxTokens || 100,
        temperature: request.temperature || 1.0,
        top_p: request.topP || 1.0,
        top_k: request.topK || 50,
        num_beams: request.numBeams || 1,
        do_sample: request.doSample !== false,
        language: request.language,
        stream: true
      })
    });

    if (!response.ok) {
      throw BharatAIError.fromResponse({ 
        status: response.status, 
        statusText: response.statusText,
        data: await response.json()
      });
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new BharatAIError('Streaming not supported', 'STREAMING_NOT_SUPPORTED');
    }

    const decoder = new TextDecoder();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              if (data.generated_text) {
                yield data.generated_text;
              }
            } catch (e) {
              // Ignore malformed JSON lines
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Update client configuration
   */
  updateConfig(config: Partial<ClientConfig>): void {
    if (config.baseURL) {
      this.config.baseURL = config.baseURL;
      this.client.defaults.baseURL = config.baseURL;
    }
    if (config.apiKey) {
      this.config.apiKey = config.apiKey;
      this.client.defaults.headers.common['Authorization'] = `Bearer ${config.apiKey}`;
    }
    if (config.timeout !== undefined) {
      this.config.timeout = config.timeout;
      this.client.defaults.timeout = config.timeout;
    }
    if (config.maxRetries !== undefined) {
      this.config.maxRetries = config.maxRetries;
    }
    if (config.retryDelay !== undefined) {
      this.config.retryDelay = config.retryDelay;
    }
    if (config.debug !== undefined) {
      this.config.debug = config.debug;
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): ClientConfig {
    return { ...this.config };
  }

  /**
   * Close client and cleanup resources
   */
  async close(): Promise<void> {
    // Cleanup any resources if needed
    // Currently, no specific cleanup required for HTTP client
  }
}