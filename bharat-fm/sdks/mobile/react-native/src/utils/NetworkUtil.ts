/**
 * Network utility for the Bharat AI SDK
 */

import { Platform } from 'react-native';
import NetInfo from '@react-native-netinfo/netinfo';
import { Logger } from './Logger';
import { BharatAIError } from '../errors/BharatAIError';

export interface NetworkInfo {
  isConnected: boolean;
  isInternetReachable: boolean | null;
  type: string | null;
  details: any;
  lastChecked: Date;
}

export interface NetworkConfig {
  /** Timeout for network requests */
  timeout: number;
  ** Retry attempts */
  retryAttempts: number;
  ** Retry delay */
  retryDelay: number;
  ** Enable offline mode */
  enableOfflineMode: boolean;
}

export class NetworkUtil {
  private logger: Logger;
  private config: NetworkConfig;
  private networkInfo: NetworkInfo | null = null;
  private listeners: Set<(info: NetworkInfo) => void> = new Set();

  constructor(config: NetworkConfig, logger?: Logger) {
    this.config = config;
    this.logger = logger || new Logger(false, 'Network');
    
    // Start monitoring network state
    this.startMonitoring();
  }

  /**
   * Get current network information
   */
  public async getNetworkInfo(): Promise<NetworkInfo> {
    try {
      const netInfo = await NetInfo.fetch();
      const info: NetworkInfo = {
        isConnected: netInfo.isConnected === true,
        isInternetReachable: netInfo.isInternetReachable,
        type: netInfo.type,
        details: netInfo.details,
        lastChecked: new Date(),
      };

      this.networkInfo = info;
      this.logger.debug('Network info updated:', info);
      
      return info;
    } catch (error) {
      this.logger.error('Failed to get network info:', error);
      throw new BharatAIError('NETWORK_ERROR', 'Failed to get network information', error);
    }
  }

  /**
   * Check if device is online
   */
  public async isOnline(): Promise<boolean> {
    try {
      const info = await this.getNetworkInfo();
      return info.isConnected && info.isInternetReachable !== false;
    } catch (error) {
      this.logger.error('Failed to check online status:', error);
      return false;
    }
  }

  /**
   * Wait for network connectivity
   */
  public async waitForOnline(timeout: number = 30000): Promise<boolean> {
    try {
      this.logger.info(`Waiting for online connectivity (timeout: ${timeout}ms)`);
      
      const startTime = Date.now();
      
      while (Date.now() - startTime < timeout) {
        const isOnline = await this.isOnline();
        if (isOnline) {
          this.logger.info('Network connectivity restored');
          return true;
        }
        
        // Wait before checking again
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      this.logger.warn('Network connectivity timeout');
      return false;
    } catch (error) {
      this.logger.error('Error waiting for network connectivity:', error);
      return false;
    }
  }

  /**
   * Get network type
   */
  public async getNetworkType(): Promise<string> {
    try {
      const info = await this.getNetworkInfo();
      return info.type || 'unknown';
    } catch (error) {
      this.logger.error('Failed to get network type:', error);
      return 'unknown';
    }
  }

  /**
   * Get connection quality
   */
  public async getConnectionQuality(): Promise<'excellent' | 'good' | 'fair' | 'poor'> {
    try {
      const info = await this.getNetworkInfo();
      
      if (!info.isConnected || info.isInternetReachable === false) {
        return 'poor';
      }

      // Simple quality assessment based on connection type
      switch (info.type) {
        case 'wifi':
          return 'excellent';
        case 'cellular':
          // For cellular, check details if available
          if (info.details && info.details.cellularGeneration) {
            switch (info.details.cellularGeneration) {
              case '5g':
                return 'excellent';
              case '4g':
              case 'lte':
                return 'good';
              case '3g':
                return 'fair';
              default:
                return 'poor';
            }
          }
          return 'fair';
        case 'ethernet':
          return 'excellent';
        default:
          return 'fair';
      }
    } catch (error) {
      this.logger.error('Failed to get connection quality:', error);
      return 'poor';
    }
  }

  /**
   * Add network state listener
   */
  public addListener(listener: (info: NetworkInfo) => void): () => void {
    this.listeners.add(listener);
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Remove network state listener
   */
  public removeListener(listener: (info: NetworkInfo) => void): void {
    this.listeners.delete(listener);
  }

  /**
   * Start monitoring network state
   */
  private startMonitoring(): void {
    const unsubscribe = NetInfo.addEventListener(async (state) => {
      try {
        const info: NetworkInfo = {
          isConnected: state.isConnected === true,
          isInternetReachable: state.isInternetReachable,
          type: state.type,
          details: state.details,
          lastChecked: new Date(),
        };

        this.networkInfo = info;
        
        // Notify listeners
        this.listeners.forEach(listener => {
          try {
            listener(info);
          } catch (error) {
            this.logger.error('Error in network listener:', error);
          }
        });

        this.logger.debug('Network state changed:', info);
      } catch (error) {
        this.logger.error('Error handling network state change:', error);
      }
    });

    // Store unsubscribe function for cleanup
    (this as any).unsubscribe = unsubscribe;
  }

  /**
   * Stop monitoring network state
   */
  public stopMonitoring(): void {
    if ((this as any).unsubscribe) {
      (this as any).unsubscribe();
      delete (this as any).unsubscribe;
    }
    this.listeners.clear();
  }

  /**
   * Get network statistics
   */
  public async getStats(): Promise<{
    isOnline: boolean;
    networkType: string;
    connectionQuality: string;
    lastChecked: Date | null;
    listenerCount: number;
  }> {
    try {
      const isOnline = await this.isOnline();
      const networkType = await this.getNetworkType();
      const connectionQuality = await this.getConnectionQuality();
      
      return {
        isOnline,
        networkType,
        connectionQuality,
        lastChecked: this.networkInfo?.lastChecked || null,
        listenerCount: this.listeners.size,
      };
    } catch (error) {
      this.logger.error('Failed to get network stats:', error);
      return {
        isOnline: false,
        networkType: 'unknown',
        connectionQuality: 'poor',
        lastChecked: null,
        listenerCount: this.listeners.size,
      };
    }
  }

  /**
   * Update configuration
   */
  public updateConfig(config: Partial<NetworkConfig>): void {
    this.config = { ...this.config, ...config };
    this.logger.debug('Network configuration updated');
  }

  /**
   * Get current configuration
   */
  public getConfig(): NetworkConfig {
    return { ...this.config };
  }

  /**
   * Destroy network utility
   */
  public destroy(): void {
    this.stopMonitoring();
    this.listeners.clear();
    this.logger.info('Network utility destroyed');
  }
}