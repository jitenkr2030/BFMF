/**
 * Main client class for the Bharat AI React Native SDK
 */

import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import DeviceInfo from 'react-native-device-info';

import { ClientConfig } from '../types/ClientConfig';
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
  ErrorResponse,
  StreamingResponse
} from '../types/ApiTypes';
import { BharatAIError } from '../errors/BharatAIError';
import { CacheManager } from '../utils/CacheManager';
import { NetworkManager } from '../utils/NetworkManager';
import { Logger } from '../utils/Logger';

// Import domain clients
import { LanguageAIClient } from '../domains/LanguageAIClient';
import { GovernanceAIClient } from '../domains/GovernanceAIClient';
import { EducationAIClient } from '../domains/EducationAIClient';
import { FinanceAIClient } from '../domains/FinanceAIClient';

export class BharatAIClient {
  private config: ClientConfig;
  private httpClient: AxiosInstance;
  private cacheManager: CacheManager;
  private networkManager: NetworkManager;
  private logger: Logger;
  private isInitialized: boolean = false;

  // Domain clients
  public language: LanguageAIClient;
  public governance: GovernanceAIClient;
  public education: EducationAIClient;
  public finance: FinanceAIClient;

  constructor(config: ClientConfig) {
    this.config = { ...config };
    this.logger = new Logger(this.config.enableLogging);
    this.cacheManager = new CacheManager(this.config, this.logger);
    this.networkManager = new NetworkManager(this.config, this.logger);
    
    // Initialize HTTP client
    this.httpClient = this.createHttpClient();
    
    // Initialize domain clients
    this.language = new LanguageAIClient(this);
    this.governance = new GovernanceAIClient(this);
    this.education = new EducationAIClient(this);
    this.finance = new FinanceAIClient(this);
  }

  /**
   * Initialize the client
   */
  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      this.logger.info('Initializing Bharat AI SDK...');
      
      // Initialize cache
      if (this.config.enableOfflineMode || this.config.cacheResponses) {
        await this.cacheManager.initialize();
      }
      
      // Initialize network manager
      await this.networkManager.initialize();
      
      // Get device information
      const deviceInfo = await this.getDeviceInfo();
      this.logger.info('Device info:', deviceInfo);
      
      this.isInitialized = true;
      this.logger.info('Bharat AI SDK initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize Bharat AI SDK:', error);
      throw new BharatAIError('INITIALIZATION_FAILED', 'Failed to initialize SDK', error);
    }
  }

  /**
   * Create HTTP client with configuration
   */
  private createHttpClient(): AxiosInstance {
    const client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': `Bharat-AI-React-Native-SDK/1.0.0`,
        'X-Platform': Platform.OS,
        'X-Device-Version': DeviceInfo.getVersion(),
        ...this.config.customHeaders,
      },
    });

    // Add authentication header if API key is provided
    if (this.config.apiKey) {
      client.defaults.headers.common['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    // Add request interceptor
    client.interceptors.request.use(
      (config) => this.networkManager.handleRequest(config),
      (error) => Promise.reject(error)
    );

    // Add response interceptor
    client.interceptors.response.use(
      (response) => this.networkManager.handleResponse(response),
      (error) => this.networkManager.handleError(error)
    );

    return client;
  }

  /**
   * Get device information
   */
  private async getDeviceInfo(): Promise<Record<string, any>> {
    try {
      return {
        platform: Platform.OS,
        version: DeviceInfo.getVersion(),
        buildNumber: DeviceInfo.getBuildNumber(),
        brand: await DeviceInfo.getBrand(),
        model: await DeviceInfo.getModel(),
        systemVersion: await DeviceInfo.getSystemVersion(),
        deviceId: await DeviceInfo.getDeviceId(),
        isEmulator: await DeviceInfo.isEmulator(),
        isTablet: DeviceInfo.isTablet(),
      };
    } catch (error) {
      this.logger.warn('Failed to get device info:', error);
      return {};
    }
  }

  /**
   * Generate text from prompt
   */
  public async generateText(request: GenerationRequest): Promise<GenerationResponse> {
    try {
      this.logger.info('Generating text...');
      
      // Check cache first
      const cacheKey = `generate_${JSON.stringify(request)}`;
      if (this.config.cacheResponses) {
        const cached = await this.cacheManager.get<GenerationResponse>(cacheKey);
        if (cached) {
          this.logger.info('Returning cached response');
          return cached;
        }
      }

      // Make API request
      const response = await this.httpClient.post<GenerationResponse>('/generate', request);
      
      // Cache response
      if (this.config.cacheResponses) {
        await this.cacheManager.set(cacheKey, response.data);
      }

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Generate text with streaming support
   */
  public async generateTextStream(
    request: GenerationRequest,
    onChunk: (chunk: StreamingResponse) => void
  ): Promise<GenerationResponse> {
    try {
      this.logger.info('Generating text stream...');
      
      const response = await this.httpClient.post('/generate/stream', request, {
        responseType: 'stream',
      });

      return new Promise((resolve, reject) => {
        let cumulativeText = '';
        let totalProcessingTime = 0;

        response.data.on('data', (chunk: Buffer) => {
          try {
            const chunkData: StreamingResponse = JSON.parse(chunk.toString());
            cumulativeText += chunkData.text;
            totalProcessingTime += chunkData.chunkProcessingTime;
            
            chunkData.cumulativeText = cumulativeText;
            chunkData.totalProcessingTime = totalProcessingTime;
            
            onChunk(chunkData);
            
            if (chunkData.isFinal) {
              resolve({
                id: chunkData.chunkId,
                text: cumulativeText,
                tokensUsed: this.estimateTokens(cumulativeText),
                confidence: 0.95,
                timestamp: new Date().toISOString(),
                model: 'default',
                language: request.language || 'en',
                processingTime: totalProcessingTime,
              });
            }
          } catch (error) {
            this.logger.error('Error processing stream chunk:', error);
          }
        });

        response.data.on('error', (error: Error) => {
          reject(this.handleError(error));
        });
      });
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Generate text for multiple prompts
   */
  public async generateBatch(request: BatchGenerationRequest): Promise<BatchGenerationResponse> {
    try {
      this.logger.info('Generating batch text...');
      
      const response = await this.httpClient.post<BatchGenerationResponse>('/generate/batch', request);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Generate embeddings for text
   */
  public async generateEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    try {
      this.logger.info('Generating embeddings...');
      
      // Check cache first
      const cacheKey = `embed_${JSON.stringify(request)}`;
      if (this.config.cacheResponses) {
        const cached = await this.cacheManager.get<EmbeddingResponse>(cacheKey);
        if (cached) {
          this.logger.info('Returning cached embeddings');
          return cached;
        }
      }

      const response = await this.httpClient.post<EmbeddingResponse>('/embeddings', request);
      
      // Cache response
      if (this.config.cacheResponses) {
        await this.cacheManager.set(cacheKey, response.data);
      }

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get model information
   */
  public async getModelInfo(modelName?: string): Promise<ModelInfo> {
    try {
      this.logger.info('Getting model info...');
      
      const endpoint = modelName ? `/models/${modelName}` : '/models';
      const response = await this.httpClient.get<ModelInfo>(endpoint);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get health status
   */
  public async getHealth(): Promise<HealthResponse> {
    try {
      this.logger.info('Getting health status...');
      
      const response = await this.httpClient.get<HealthResponse>('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get supported languages
   */
  public async getSupportedLanguages(): Promise<SupportedLanguagesResponse> {
    try {
      this.logger.info('Getting supported languages...');
      
      // Check cache first
      const cacheKey = 'supported_languages';
      if (this.config.cacheResponses) {
        const cached = await this.cacheManager.get<SupportedLanguagesResponse>(cacheKey);
        if (cached) {
          this.logger.info('Returning cached languages');
          return cached;
        }
      }

      const response = await this.httpClient.get<SupportedLanguagesResponse>('/languages');
      
      // Cache response
      if (this.config.cacheResponses) {
        await this.cacheManager.set(cacheKey, response.data);
      }

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Clear cache
   */
  public async clearCache(): Promise<void> {
    try {
      this.logger.info('Clearing cache...');
      await this.cacheManager.clear();
    } catch (error) {
      this.logger.error('Failed to clear cache:', error);
      throw new BharatAIError('CACHE_ERROR', 'Failed to clear cache', error);
    }
  }

  /**
   * Get cache statistics
   */
  public async getCacheStats(): Promise<Record<string, any>> {
    try {
      return await this.cacheManager.getStats();
    } catch (error) {
      this.logger.error('Failed to get cache stats:', error);
      return {};
    }
  }

  /**
   * Update configuration
   */
  public updateConfig(newConfig: Partial<ClientConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.logger.updateConfig(this.config.enableLogging);
    this.cacheManager.updateConfig(this.config);
    this.networkManager.updateConfig(this.config);
  }

  /**
   * Handle errors
   */
  private handleError(error: any): BharatAIError {
    if (error instanceof BharatAIError) {
      return error;
    }

    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ErrorResponse>;
      const errorResponse = axiosError.response?.data;
      
      if (errorResponse) {
        return new BharatAIError(
          errorResponse.code || 'API_ERROR',
          errorResponse.message,
          error,
          errorResponse.statusCode
        );
      }
    }

    return new BharatAIError(
      'UNKNOWN_ERROR',
      error?.message || 'An unknown error occurred',
      error
    );
  }

  /**
   * Estimate token count (rough approximation)
   */
  private estimateTokens(text: string): number {
    // Simple token estimation: ~4 characters per token on average
    return Math.ceil(text.length / 4);
  }

  /**
   * Check if client is initialized
   */
  public isClientInitialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Get client configuration
   */
  public getConfig(): ClientConfig {
    return { ...this.config };
  }

  /**
   * Destroy client and cleanup resources
   */
  public async destroy(): Promise<void> {
    try {
      this.logger.info('Destroying Bharat AI client...');
      
      if (this.cacheManager) {
        await this.cacheManager.destroy();
      }
      
      if (this.networkManager) {
        await this.networkManager.destroy();
      }
      
      this.isInitialized = false;
      this.logger.info('Bharat AI client destroyed');
    } catch (error) {
      this.logger.error('Error destroying client:', error);
    }
  }
}