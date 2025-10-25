/**
 * Cache manager for the Bharat AI SDK
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { ClientConfig } from '../types/ClientConfig';
import { Logger } from './Logger';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number; // Time to live in milliseconds
}

export class CacheManager {
  private config: ClientConfig;
  private logger: Logger;
  private memoryCache: Map<string, CacheEntry<any>>;
  private isInitialized: boolean = false;

  constructor(config: ClientConfig, logger: Logger) {
    this.config = config;
    this.logger = logger.child('Cache');
    this.memoryCache = new Map();
  }

  /**
   * Initialize cache manager
   */
  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      this.logger.info('Initializing cache manager...');
      
      // Load existing cache entries from storage
      if (this.config.enableOfflineMode) {
        await this.loadCacheFromStorage();
      }
      
      // Clean up expired entries
      await this.cleanupExpiredEntries();
      
      this.isInitialized = true;
      this.logger.info('Cache manager initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize cache manager:', error);
      throw error;
    }
  }

  /**
   * Get cached data
   */
  public async get<T>(key: string): Promise<T | null> {
    if (!this.isInitialized || !this.config.cacheResponses) {
      return null;
    }

    try {
      // Check memory cache first
      const memoryEntry = this.memoryCache.get(key);
      if (memoryEntry && !this.isExpired(memoryEntry)) {
        this.logger.debug('Cache hit (memory):', key);
        return memoryEntry.data;
      }

      // Check persistent storage
      if (this.config.enableOfflineMode) {
        const storageEntry = await AsyncStorage.getItem(`bharat_ai_cache_${key}`);
        if (storageEntry) {
          const parsedEntry: CacheEntry<T> = JSON.parse(storageEntry);
          if (!this.isExpired(parsedEntry)) {
            // Move to memory cache for faster access
            this.memoryCache.set(key, parsedEntry);
            this.logger.debug('Cache hit (storage):', key);
            return parsedEntry.data;
          } else {
            // Remove expired entry
            await AsyncStorage.removeItem(`bharat_ai_cache_${key}`);
          }
        }
      }

      this.logger.debug('Cache miss:', key);
      return null;
    } catch (error) {
      this.logger.error('Error getting from cache:', error);
      return null;
    }
  }

  /**
   * Set cached data
   */
  public async set<T>(key: string, data: T, ttl?: number): Promise<void> {
    if (!this.isInitialized || !this.config.cacheResponses) {
      return;
    }

    try {
      const cacheTtl = ttl || (this.config.cacheExpiration || 24) * 60 * 60 * 1000; // Default 24 hours
      const entry: CacheEntry<T> = {
        data,
        timestamp: Date.now(),
        ttl: cacheTtl,
      };

      // Store in memory cache
      this.memoryCache.set(key, entry);
      
      // Store in persistent storage if offline mode is enabled
      if (this.config.enableOfflineMode) {
        await AsyncStorage.setItem(`bharat_ai_cache_${key}`, JSON.stringify(entry));
        await this.enforceCacheSizeLimit();
      }

      this.logger.debug('Cache set:', key);
    } catch (error) {
      this.logger.error('Error setting cache:', error);
    }
  }

  /**
   * Remove cached data
   */
  public async remove(key: string): Promise<void> {
    try {
      this.memoryCache.delete(key);
      
      if (this.config.enableOfflineMode) {
        await AsyncStorage.removeItem(`bharat_ai_cache_${key}`);
      }
      
      this.logger.debug('Cache removed:', key);
    } catch (error) {
      this.logger.error('Error removing from cache:', error);
    }
  }

  /**
   * Clear all cached data
   */
  public async clear(): Promise<void> {
    try {
      this.memoryCache.clear();
      
      if (this.config.enableOfflineMode) {
        const allKeys = await AsyncStorage.getAllKeys();
        const cacheKeys = allKeys.filter(key => key.startsWith('bharat_ai_cache_'));
        await AsyncStorage.multiRemove(cacheKeys);
      }
      
      this.logger.info('Cache cleared');
    } catch (error) {
      this.logger.error('Error clearing cache:', error);
      throw error;
    }
  }

  /**
   * Get cache statistics
   */
  public async getStats(): Promise<Record<string, any>> {
    try {
      const memorySize = this.memoryCache.size;
      let storageSize = 0;
      let totalSize = 0;

      if (this.config.enableOfflineMode) {
        const allKeys = await AsyncStorage.getAllKeys();
        const cacheKeys = allKeys.filter(key => key.startsWith('bharat_ai_cache_'));
        storageSize = cacheKeys.length;
        
        // Calculate approximate size
        for (const key of cacheKeys) {
          const value = await AsyncStorage.getItem(key);
          if (value) {
            totalSize += key.length + value.length;
          }
        }
      }

      return {
        memoryEntries: memorySize,
        storageEntries: storageSize,
        totalEntries: memorySize + storageSize,
        approximateSizeBytes: totalSize,
        maxSizeBytes: (this.config.maxCacheSize || 100) * 1024 * 1024, // Convert MB to bytes
        enabled: this.config.cacheResponses,
        offlineMode: this.config.enableOfflineMode,
      };
    } catch (error) {
      this.logger.error('Error getting cache stats:', error);
      return {};
    }
  }

  /**
   * Update configuration
   */
  public updateConfig(config: ClientConfig): void {
    this.config = config;
    this.logger.debug('Cache configuration updated');
  }

  /**
   * Destroy cache manager
   */
  public async destroy(): Promise<void> {
    try {
      await this.clear();
      this.isInitialized = false;
      this.logger.info('Cache manager destroyed');
    } catch (error) {
      this.logger.error('Error destroying cache manager:', error);
    }
  }

  /**
   * Check if cache entry is expired
   */
  private isExpired(entry: CacheEntry<any>): boolean {
    return Date.now() - entry.timestamp > entry.ttl;
  }

  /**
   * Load cache from persistent storage
   */
  private async loadCacheFromStorage(): Promise<void> {
    try {
      const allKeys = await AsyncStorage.getAllKeys();
      const cacheKeys = allKeys.filter(key => key.startsWith('bharat_ai_cache_'));

      for (const key of cacheKeys) {
        const value = await AsyncStorage.getItem(key);
        if (value) {
          const entry: CacheEntry<any> = JSON.parse(value);
          const cacheKey = key.replace('bharat_ai_cache_', '');
          
          if (!this.isExpired(entry)) {
            this.memoryCache.set(cacheKey, entry);
          } else {
            await AsyncStorage.removeItem(key);
          }
        }
      }

      this.logger.debug(`Loaded ${cacheKeys.length} entries from storage`);
    } catch (error) {
      this.logger.error('Error loading cache from storage:', error);
    }
  }

  /**
   * Clean up expired entries
   */
  private async cleanupExpiredEntries(): Promise<void> {
    try {
      // Clean memory cache
      for (const [key, entry] of this.memoryCache.entries()) {
        if (this.isExpired(entry)) {
          this.memoryCache.delete(key);
        }
      }

      // Clean storage cache
      if (this.config.enableOfflineMode) {
        const allKeys = await AsyncStorage.getAllKeys();
        const cacheKeys = allKeys.filter(key => key.startsWith('bharat_ai_cache_'));

        for (const key of cacheKeys) {
          const value = await AsyncStorage.getItem(key);
          if (value) {
            const entry: CacheEntry<any> = JSON.parse(value);
            if (this.isExpired(entry)) {
              await AsyncStorage.removeItem(key);
            }
          }
        }
      }

      this.logger.debug('Cache cleanup completed');
    } catch (error) {
      this.logger.error('Error during cache cleanup:', error);
    }
  }

  /**
   * Enforce cache size limit
   */
  private async enforceCacheSizeLimit(): Promise<void> {
    try {
      const maxSize = (this.config.maxCacheSize || 100) * 1024 * 1024; // Convert MB to bytes
      const stats = await this.getStats();
      
      if (stats.approximateSizeBytes > maxSize) {
        this.logger.warn('Cache size limit exceeded, cleaning up oldest entries');
        
        // Get all entries with their timestamps
        const entries: Array<{ key: string; timestamp: number }> = [];
        
        for (const [key, entry] of this.memoryCache.entries()) {
          entries.push({ key, timestamp: entry.timestamp });
        }

        // Sort by timestamp (oldest first)
        entries.sort((a, b) => a.timestamp - b.timestamp);

        // Remove oldest entries until under limit
        let currentSize = stats.approximateSizeBytes;
        for (const entry of entries) {
          if (currentSize <= maxSize) {
            break;
          }

          await this.remove(entry.key);
          currentSize -= entry.key.length + JSON.stringify(this.memoryCache.get(entry.key)).length;
        }

        this.logger.debug(`Cache size reduced to ${currentSize} bytes`);
      }
    } catch (error) {
      this.logger.error('Error enforcing cache size limit:', error);
    }
  }
}