/**
 * Client configuration for the Bharat AI React Native SDK
 */

export interface ClientConfig {
  /** Base URL for the API */
  baseURL: string;
  /** Request timeout in milliseconds */
  timeout: number;
  /** Maximum number of retry attempts */
  maxRetries: number;
  /** Delay between retries in milliseconds */
  retryDelay: number;
  /** Enable offline mode with local caching */
  enableOfflineMode: boolean;
  /** Cache API responses locally */
  cacheResponses: boolean;
  /** Enable debug logging */
  enableLogging: boolean;
  /** API key for authentication */
  apiKey?: string;
  /** Default language for requests */
  defaultLanguage?: string;
  /** Custom headers to include in requests */
  customHeaders?: Record<string, string>;
  /** Enable compression for requests */
  enableCompression?: boolean;
  /** Maximum cache size in MB */
  maxCacheSize?: number;
  /** Cache expiration time in hours */
  cacheExpiration?: number;
}

export interface PlatformConfig {
  /** Platform-specific configuration */
  platform: 'ios' | 'android';
  /** Device-specific optimizations */
  deviceOptimizations: {
    /** Enable GPU acceleration */
    enableGPU: boolean;
    /** Maximum concurrent requests */
    maxConcurrentRequests: number;
    /** Battery optimization mode */
    batteryOptimization: boolean;
  };
}

export interface NetworkConfig {
  /** Network timeout configuration */
  connectTimeout: number;
  readTimeout: number;
  writeTimeout: number;
  /** Retry configuration */
  retryOnNetworkError: boolean;
  retryOnServerError: boolean;
  retryStatusCodes: number[];
}

export interface CacheConfig {
  /** Cache configuration */
  enabled: boolean;
  maxSize: number; // in MB
  ttl: number; // in hours
  persistent: boolean;
  encryptionKey?: string;
}