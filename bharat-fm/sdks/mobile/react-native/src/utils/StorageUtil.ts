/**
 * Storage utility for the Bharat AI SDK
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { Logger } from './Logger';
import { BharatAIError } from '../errors/BharatAIError';

export interface StorageOptions {
  /** Enable encryption */
  encryption?: boolean;
  /** Encryption key (if encryption is enabled) */
  encryptionKey?: string;
  /** Default TTL in milliseconds */
  defaultTTL?: number;
}

export class StorageUtil {
  private logger: Logger;
  private options: StorageOptions;

  constructor(options: StorageOptions = {}) {
    this.logger = new Logger(false, 'Storage');
    this.options = {
      encryption: false,
      defaultTTL: 24 * 60 * 60 * 1000, // 24 hours
      ...options,
    };
  }

  /**
   * Store data with optional TTL
   */
  public async set<T>(key: string, value: T, ttl?: number): Promise<void> {
    try {
      const storageKey = `bharat_ai_${key}`;
      const data = {
        value,
        timestamp: Date.now(),
        ttl: ttl || this.options.defaultTTL,
      };

      let serializedData = JSON.stringify(data);

      // Apply encryption if enabled
      if (this.options.encryption && this.options.encryptionKey) {
        serializedData = await this.encrypt(serializedData, this.options.encryptionKey);
      }

      await AsyncStorage.setItem(storageKey, serializedData);
      this.logger.debug(`Stored data for key: ${key}`);
    } catch (error) {
      this.logger.error(`Failed to store data for key ${key}:`, error);
      throw new BharatAIError('STORAGE_ERROR', `Failed to store data for key ${key}`, error);
    }
  }

  /**
   * Retrieve data
   */
  public async get<T>(key: string): Promise<T | null> {
    try {
      const storageKey = `bharat_ai_${key}`;
      const serializedData = await AsyncStorage.getItem(storageKey);

      if (!serializedData) {
        return null;
      }

      // Apply decryption if enabled
      let decryptedData = serializedData;
      if (this.options.encryption && this.options.encryptionKey) {
        decryptedData = await this.decrypt(serializedData, this.options.encryptionKey);
      }

      const data = JSON.parse(decryptedData);

      // Check TTL
      if (data.ttl && Date.now() - data.timestamp > data.ttl) {
        await this.remove(key);
        return null;
      }

      this.logger.debug(`Retrieved data for key: ${key}`);
      return data.value;
    } catch (error) {
      this.logger.error(`Failed to retrieve data for key ${key}:`, error);
      return null;
    }
  }

  /**
   * Remove data
   */
  public async remove(key: string): Promise<void> {
    try {
      const storageKey = `bharat_ai_${key}`;
      await AsyncStorage.removeItem(storageKey);
      this.logger.debug(`Removed data for key: ${key}`);
    } catch (error) {
      this.logger.error(`Failed to remove data for key ${key}:`, error);
      throw new BharatAIError('STORAGE_ERROR', `Failed to remove data for key ${key}`, error);
    }
  }

  /**
   * Clear all Bharat AI related data
   */
  public async clear(): Promise<void> {
    try {
      const allKeys = await AsyncStorage.getAllKeys();
      const bharatKeys = allKeys.filter(key => key.startsWith('bharat_ai_'));
      
      if (bharatKeys.length > 0) {
        await AsyncStorage.multiRemove(bharatKeys);
        this.logger.info(`Cleared ${bharatKeys.length} storage items`);
      }
    } catch (error) {
      this.logger.error('Failed to clear storage:', error);
      throw new BharatAIError('STORAGE_ERROR', 'Failed to clear storage', error);
    }
  }

  /**
   * Get storage statistics
   */
  public async getStats(): Promise<{
    totalKeys: number;
    totalSize: number;
    keys: Array<{ key: string; size: number; timestamp: number }>;
  }> {
    try {
      const allKeys = await AsyncStorage.getAllKeys();
      const bharatKeys = allKeys.filter(key => key.startsWith('bharat_ai_'));
      
      let totalSize = 0;
      const keys: Array<{ key: string; size: number; timestamp: number }> = [];

      for (const key of bharatKeys) {
        const value = await AsyncStorage.getItem(key);
        if (value) {
          const size = key.length + value.length;
          totalSize += size;
          
          try {
            const parsed = JSON.parse(value);
            keys.push({
              key: key.replace('bharat_ai_', ''),
              size,
              timestamp: parsed.timestamp || 0,
            });
          } catch {
            keys.push({
              key: key.replace('bharat_ai_', ''),
              size,
              timestamp: 0,
            });
          }
        }
      }

      return {
        totalKeys: bharatKeys.length,
        totalSize,
        keys: keys.sort((a, b) => b.timestamp - a.timestamp),
      };
    } catch (error) {
      this.logger.error('Failed to get storage stats:', error);
      return {
        totalKeys: 0,
        totalSize: 0,
        keys: [],
      };
    }
  }

  /**
   * Check if key exists
   */
  public async exists(key: string): Promise<boolean> {
    try {
      const storageKey = `bharat_ai_${key}`;
      const value = await AsyncStorage.getItem(storageKey);
      return value !== null;
    } catch (error) {
      this.logger.error(`Failed to check existence for key ${key}:`, error);
      return false;
    }
  }

  /**
   * Get all keys
   */
  public async keys(): Promise<string[]> {
    try {
      const allKeys = await AsyncStorage.getAllKeys();
      return allKeys
        .filter(key => key.startsWith('bharat_ai_'))
        .map(key => key.replace('bharat_ai_', ''));
    } catch (error) {
      this.logger.error('Failed to get keys:', error);
      return [];
    }
  }

  /**
   * Simple encryption (for demo purposes - use proper encryption in production)
   */
  private async encrypt(text: string, key: string): Promise<string> {
    // This is a simple XOR encryption for demonstration
    // In production, use proper encryption libraries like react-native-aes-crypto
    let result = '';
    for (let i = 0; i < text.length; i++) {
      result += String.fromCharCode(
        text.charCodeAt(i) ^ key.charCodeAt(i % key.length)
      );
    }
    return btoa(result); // Base64 encode
  }

  /**
   * Simple decryption (for demo purposes - use proper encryption in production)
   */
  private async decrypt(encryptedText: string, key: string): Promise<string> {
    try {
      const text = atob(encryptedText); // Base64 decode
      let result = '';
      for (let i = 0; i < text.length; i++) {
        result += String.fromCharCode(
          text.charCodeAt(i) ^ key.charCodeAt(i % key.length)
        );
      }
      return result;
    } catch (error) {
      throw new BharatAIError('DECRYPTION_ERROR', 'Failed to decrypt data', error);
    }
  }
}