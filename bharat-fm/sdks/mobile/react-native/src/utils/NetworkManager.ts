/**
 * Network manager for the Bharat AI SDK
 */

import { Platform } from 'react-native';
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import NetInfo from '@react-native-netinfo/netinfo';

import { ClientConfig } from '../types/ClientConfig';
import { Logger } from './Logger';
import { BharatAIError } from '../errors/BharatAIError';

export class NetworkManager {
  private config: ClientConfig;
  private logger: Logger;
  private retryCount: Map<string, number> = new Map();
  private isInitialized: boolean = false;

  constructor(config: ClientConfig, logger: Logger) {
    this.config = config;
    this.logger = logger.child('Network');
  }

  /**
   * Initialize network manager
   */
  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      this.logger.info('Initializing network manager...');
      
      // Set up network state monitoring
      if (this.config.enableOfflineMode) {
        await this.setupNetworkMonitoring();
      }
      
      this.isInitialized = true;
      this.logger.info('Network manager initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize network manager:', error);
      throw error;
    }
  }

  /**
   * Handle outgoing requests
   */
  public async handleRequest(config: AxiosRequestConfig): Promise<AxiosRequestConfig> {
    try {
      // Add request metadata
      config.metadata = {
        startTime: Date.now(),
        requestId: this.generateRequestId(),
      };

      // Add retry information
      const url = config.url || '';
      const retryCount = this.retryCount.get(url) || 0;
      config.headers = config.headers || {};
      config.headers['X-Retry-Count'] = retryCount.toString();

      this.logger.debug(`Request started: ${config.method?.toUpperCase()} ${url}`, {
        requestId: config.metadata.requestId,
        retryCount,
      });

      return config;
    } catch (error) {
      this.logger.error('Error handling request:', error);
      throw error;
    }
  }

  /**
   * Handle successful responses
   */
  public async handleResponse(response: AxiosResponse): Promise<AxiosResponse> {
    try {
      const duration = Date.now() - (response.config.metadata?.startTime || Date.now());
      const url = response.config.url || '';

      this.logger.debug(`Request completed: ${response.config.method?.toUpperCase()} ${url}`, {
        status: response.status,
        duration: `${duration}ms`,
        requestId: response.config.metadata?.requestId,
      });

      // Reset retry count on success
      this.retryCount.delete(url);

      return response;
    } catch (error) {
      this.logger.error('Error handling response:', error);
      throw error;
    }
  }

  /**
   * Handle errors
   */
  public async handleError(error: AxiosError): Promise<never> {
    try {
      const url = error.config?.url || '';
      const retryCount = this.retryCount.get(url) || 0;
      
      this.logger.debug(`Request failed: ${error.config?.method?.toUpperCase()} ${url}`, {
        status: error.response?.status,
        message: error.message,
        retryCount,
        requestId: error.config?.metadata?.requestId,
      });

      // Check if we should retry
      if (this.shouldRetry(error, retryCount)) {
        this.retryCount.set(url, retryCount + 1);
        
        const delay = this.calculateRetryDelay(retryCount);
        this.logger.info(`Retrying request (${retryCount + 1}/${this.config.maxRetries}) after ${delay}ms`);
        
        await this.delay(delay);
        
        // Retry the request
        return this.retryRequest(error.config!);
      }

      // Convert to BharatAIError
      throw this.convertToBharatAIError(error);
    } catch (conversionError) {
      if (conversionError instanceof BharatAIError) {
        throw conversionError;
      }
      throw BharatAIError.networkError('Network error occurred', conversionError);
    }
  }

  /**
   * Check network connectivity
   */
  public async isOnline(): Promise<boolean> {
    try {
      const netInfo = await NetInfo.fetch();
      return netInfo.isConnected === true && netInfo.isInternetReachable !== false;
    } catch (error) {
      this.logger.error('Error checking network status:', error);
      return false;
    }
  }

  /**
   * Wait for network connectivity
   */
  public async waitForOnline(timeout: number = 30000): Promise<boolean> {
    try {
      this.logger.info('Waiting for network connectivity...');
      
      const startTime = Date.now();
      
      while (Date.now() - startTime < timeout) {
        const isOnline = await this.isOnline();
        if (isOnline) {
          this.logger.info('Network connectivity restored');
          return true;
        }
        
        await this.delay(1000); // Check every second
      }
      
      this.logger.warn('Network connectivity timeout');
      return false;
    } catch (error) {
      this.logger.error('Error waiting for network connectivity:', error);
      return false;
    }
  }

  /**
   * Update configuration
   */
  public updateConfig(config: ClientConfig): void {
    this.config = config;
    this.logger.debug('Network configuration updated');
  }

  /**
   * Destroy network manager
   */
  public async destroy(): Promise<void> {
    try {
      this.retryCount.clear();
      this.isInitialized = false;
      this.logger.info('Network manager destroyed');
    } catch (error) {
      this.logger.error('Error destroying network manager:', error);
    }
  }

  /**
   * Setup network monitoring
   */
  private async setupNetworkMonitoring(): Promise<void> {
    try {
      NetInfo.addEventListener(state => {
        this.logger.debug('Network state changed:', state);
        
        if (state.isConnected && state.isInternetReachable !== false) {
          this.logger.info('Network connectivity restored');
          // Could trigger sync operations here
        } else {
          this.logger.warn('Network connectivity lost');
          // Could trigger offline mode here
        }
      });
    } catch (error) {
      this.logger.error('Error setting up network monitoring:', error);
    }
  }

  /**
   * Check if request should be retried
   */
  private shouldRetry(error: AxiosError, retryCount: number): boolean {
    if (retryCount >= this.config.maxRetries) {
      return false;
    }

    const status = error.response?.status;
    
    // Retry on network errors
    if (!error.response) {
      return true;
    }

    // Retry on specific HTTP status codes
    const retryableStatusCodes = [408, 429, 500, 502, 503, 504];
    if (status && retryableStatusCodes.includes(status)) {
      return true;
    }

    // Check for retry-after header
    const retryAfter = error.response?.headers['retry-after'];
    if (retryAfter) {
      return true;
    }

    return false;
  }

  /**
   * Calculate retry delay with exponential backoff
   */
  private calculateRetryDelay(retryCount: number): number {
    const baseDelay = this.config.retryDelay || 1000;
    const maxDelay = 30000; // Maximum 30 seconds
    
    // Exponential backoff with jitter
    const delay = Math.min(
      baseDelay * Math.pow(2, retryCount),
      maxDelay
    );
    
    // Add jitter (±25%)
    const jitter = delay * 0.25;
    const finalDelay = delay + (Math.random() * jitter * 2) - jitter;
    
    return Math.max(1000, Math.floor(finalDelay)); // Minimum 1 second
  }

  /**
   * Retry a request
   */
  private async retryRequest(config: AxiosRequestConfig): Promise<any> {
    try {
      const client = axios.create();
      return await client.request(config);
    } catch (error) {
      throw error;
    }
  }

  /**
   * Convert Axios error to BharatAIError
   */
  private convertToBharatAIError(error: AxiosError): BharatAIError {
    const status = error.response?.status;
    const message = error.response?.data?.message || error.message;
    const retryAfter = error.response?.headers['retry-after'];

    switch (status) {
      case 400:
        return BharatAIError.validationError(message, error);
      case 401:
        return BharatAIError.authenticationError(message, error);
      case 403:
        return BharatAIError.authorizationError(message, error);
      case 404:
        return BharatAIError.notFoundError(message, error);
      case 408:
        return BharatAIError.timeoutError(message, error);
      case 429:
        return BharatAIError.rateLimitError(
          message,
          retryAfter ? parseInt(retryAfter) * 1000 : undefined,
          error
        );
      case 500:
      case 502:
      case 503:
      case 504:
        return BharatAIError.serverError(message, error);
      default:
        if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
          return BharatAIError.timeoutError(message, error);
        }
        if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
          return BharatAIError.networkError(message, error);
        }
        return BharatAIError.networkError(message, error);
    }
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Delay execution
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}