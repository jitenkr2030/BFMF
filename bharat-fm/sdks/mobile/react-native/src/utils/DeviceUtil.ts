/**
 * Device utility for the Bharat AI SDK
 */

import { Platform, NativeModules, Dimensions } from 'react-native';
import DeviceInfo from 'react-native-device-info';
import { Logger } from './Logger';
import { BharatAIError } from '../errors/BharatAIError';

export interface DeviceInfo {
  /** Device platform */
  platform: 'ios' | 'android';
  /** Device model */
  model: string;
  /** Device brand */
  brand: string;
  /** Operating system version */
  systemVersion: string;
  /** App version */
  appVersion: string;
  ** Build number */
  buildNumber: string;
  ** Device ID */
  deviceId: string;
  ** Is emulator */
  isEmulator: boolean;
  ** Is tablet */
  isTablet: boolean;
  ** Screen dimensions */
  screen: {
    width: number;
    height: number;
    scale: number;
  };
  ** Memory info */
  memory?: {
    totalRAM: number;
    freeRAM: number;
    usedRAM: number;
  };
  ** Storage info */
  storage?: {
    totalDisk: number;
    freeDisk: number;
    usedDisk: number;
  };
  ** Battery info */
  battery?: {
    level: number;
    isCharging: boolean;
  };
  ** Performance capabilities */
  performance: {
    cpuCores: number;
    maxFrequency: number;
    gpuModel?: string;
    canUseGPU: boolean;
    recommendedBatchSize: number;
    maxConcurrentRequests: number;
  };
}

export class DeviceUtil {
  private logger: Logger;
  private deviceInfo: DeviceInfo | null = null;
  private isInitialized: boolean = false;

  constructor(logger?: Logger) {
    this.logger = logger || new Logger(false, 'Device');
  }

  /**
   * Initialize device utility
   */
  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      this.logger.info('Initializing device utility...');
      
      const info = await this.collectDeviceInfo();
      this.deviceInfo = info;
      this.isInitialized = true;
      
      this.logger.info('Device utility initialized successfully');
      this.logger.debug('Device info:', info);
    } catch (error) {
      this.logger.error('Failed to initialize device utility:', error);
      throw new BharatAIError('DEVICE_ERROR', 'Failed to initialize device utility', error);
    }
  }

  /**
   * Get device information
   */
  public async getDeviceInfo(): Promise<DeviceInfo> {
    if (!this.isInitialized) {
      await this.initialize();
    }
    
    if (!this.deviceInfo) {
      throw new BharatAIError('DEVICE_ERROR', 'Device information not available');
    }
    
    return this.deviceInfo;
  }

  /**
   * Check if device supports GPU acceleration
   */
  public async supportsGPU(): Promise<boolean> {
    try {
      const info = await this.getDeviceInfo();
      return info.performance.canUseGPU;
    } catch (error) {
      this.logger.error('Failed to check GPU support:', error);
      return false;
    }
  }

  /**
   * Get recommended batch size for processing
   */
  public async getRecommendedBatchSize(): Promise<number> {
    try {
      const info = await this.getDeviceInfo();
      return info.performance.recommendedBatchSize;
    } catch (error) {
      this.logger.error('Failed to get recommended batch size:', error);
      return 1; // Default safe value
    }
  }

  /**
   * Get maximum concurrent requests
   */
  public async getMaxConcurrentRequests(): Promise<number> {
    try {
      const info = await this.getDeviceInfo();
      return info.performance.maxConcurrentRequests;
    } catch (error) {
      this.logger.error('Failed to get max concurrent requests:', error);
      return 1; // Default safe value
    }
  }

  /**
   * Check if device is low-end
   */
  public async isLowEndDevice(): Promise<boolean> {
    try {
      const info = await this.getDeviceInfo();
      
      // Simple heuristic for low-end device detection
      if (info.memory && info.memory.totalRAM < 2048) { // Less than 2GB RAM
        return true;
      }
      
      if (info.performance.cpuCores < 4) { // Less than 4 CPU cores
        return true;
      }
      
      return false;
    } catch (error) {
      this.logger.error('Failed to check if device is low-end:', error);
      return true; // Assume low-end if we can't determine
    }
  }

  /**
   * Get device capabilities for AI processing
   */
  public async getAICapabilities(): Promise<{
    canProcessLocally: boolean;
    maxModelSize: number;
    recommendedOptimizations: string[];
    estimatedPerformance: 'low' | 'medium' | 'high';
  }> {
    try {
      const info = await this.getDeviceInfo();
      const isLowEnd = await this.isLowEndDevice();
      
      let maxModelSize = 50; // MB
      let estimatedPerformance: 'low' | 'medium' | 'high' = 'medium';
      const recommendedOptimizations: string[] = [];
      
      if (isLowEnd) {
        maxModelSize = 10;
        estimatedPerformance = 'low';
        recommendedOptimizations.push(
          'use_quantization',
          'reduce_batch_size',
          'disable_gpu',
          'use_lightweight_models'
        );
      } else if (info.memory && info.memory.totalRAM > 6144) { // More than 6GB RAM
        maxModelSize = 200;
        estimatedPerformance = 'high';
        recommendedOptimizations.push(
          'enable_gpu',
          'use_large_batch_size',
          'enable_mixed_precision'
        );
      } else {
        recommendedOptimizations.push(
          'use_mixed_precision',
          'enable_gpu_if_available'
        );
      }
      
      // Check if we can process locally
      const canProcessLocally = maxModelSize > 10; // At least 10MB models
      
      return {
        canProcessLocally,
        maxModelSize,
        recommendedOptimizations,
        estimatedPerformance,
      };
    } catch (error) {
      this.logger.error('Failed to get AI capabilities:', error);
      return {
        canProcessLocally: false,
        maxModelSize: 0,
        recommendedOptimizations: ['use_cloud_only'],
        estimatedPerformance: 'low',
      };
    }
  }

  /**
   * Collect device information
   */
  private async collectDeviceInfo(): Promise<DeviceInfo> {
    const screen = Dimensions.get('window');
    
    // Basic device info
    const deviceInfo: DeviceInfo = {
      platform: Platform.OS as 'ios' | 'android',
      model: await DeviceInfo.getModel(),
      brand: await DeviceInfo.getBrand(),
      systemVersion: await DeviceInfo.getSystemVersion(),
      appVersion: DeviceInfo.getVersion(),
      buildNumber: DeviceInfo.getBuildNumber(),
      deviceId: await DeviceInfo.getDeviceId(),
      isEmulator: await DeviceInfo.isEmulator(),
      isTablet: DeviceInfo.isTablet(),
      screen: {
        width: screen.width,
        height: screen.height,
        scale: screen.scale,
      },
      performance: {
        cpuCores: await this.getCPUCores(),
        maxFrequency: await this.getMaxCPUFrequency(),
        canUseGPU: await this.canUseGPU(),
        recommendedBatchSize: await this.calculateRecommendedBatchSize(),
        maxConcurrentRequests: await this.calculateMaxConcurrentRequests(),
      },
    };

    // Try to get additional platform-specific info
    try {
      if (Platform.OS === 'android') {
        deviceInfo.memory = await this.getAndroidMemoryInfo();
        deviceInfo.storage = await this.getAndroidStorageInfo();
        deviceInfo.battery = await this.getAndroidBatteryInfo();
        deviceInfo.performance.gpuModel = await this.getAndroidGPUModel();
      } else if (Platform.OS === 'ios') {
        deviceInfo.memory = await this.getIOSMemoryInfo();
        deviceInfo.storage = await this.getIOSStorageInfo();
        deviceInfo.battery = await this.getIOSBatteryInfo();
      }
    } catch (error) {
      this.logger.warn('Failed to get platform-specific device info:', error);
    }

    return deviceInfo;
  }

  /**
   * Get CPU cores count
   */
  private async getCPUCores(): Promise<number> {
    try {
      if (Platform.OS === 'android') {
        const info = await DeviceInfo.getDeviceToken();
        // This is a simplified approach - in real implementation, you'd use proper Android APIs
        return 4; // Default assumption
      } else {
        // For iOS, we can use device model to estimate
        const model = await DeviceInfo.getModel();
        if (model.includes('iPhone')) {
          // iPhone models typically have 2-6 cores
          return 4; // Default assumption
        }
        return 2; // Default for other devices
      }
    } catch (error) {
      this.logger.error('Failed to get CPU cores:', error);
      return 2; // Safe default
    }
  }

  /**
   * Get max CPU frequency
   */
  private async getMaxCPUFrequency(): Promise<number> {
    try {
      // This is platform-specific and would require native modules
      // For now, return reasonable defaults
      if (Platform.OS === 'android') {
        return 2400; // MHz - typical for mid-range Android
      } else {
        return 2400; // MHz - typical for iPhone
      }
    } catch (error) {
      this.logger.error('Failed to get max CPU frequency:', error);
      return 1500; // Safe default
    }
  }

  /**
   * Check if GPU can be used
   */
  private async canUseGPU(): Promise<boolean> {
    try {
      // Basic GPU capability check
      if (Platform.OS === 'android') {
        const model = await DeviceInfo.getModel();
        // Most modern Android devices support GPU acceleration
        return true;
      } else {
        // iOS devices generally have good GPU support
        return true;
      }
    } catch (error) {
      this.logger.error('Failed to check GPU capability:', error);
      return false; // Safe default
    }
  }

  /**
   * Calculate recommended batch size
   */
  private async calculateRecommendedBatchSize(): Promise<number> {
    try {
      const isLowEnd = await this.isLowEndDevice();
      const cpuCores = await this.getCPUCores();
      
      if (isLowEnd) {
        return 1;
      } else if (cpuCores >= 6) {
        return 8;
      } else if (cpuCores >= 4) {
        return 4;
      } else {
        return 2;
      }
    } catch (error) {
      this.logger.error('Failed to calculate recommended batch size:', error);
      return 1; // Safe default
    }
  }

  /**
   * Calculate max concurrent requests
   */
  private async calculateMaxConcurrentRequests(): Promise<number> {
    try {
      const isLowEnd = await this.isLowEndDevice();
      const cpuCores = await this.getCPUCores();
      
      if (isLowEnd) {
        return 1;
      } else if (cpuCores >= 6) {
        return 4;
      } else if (cpuCores >= 4) {
        return 3;
      } else {
        return 2;
      }
    } catch (error) {
      this.logger.error('Failed to calculate max concurrent requests:', error);
      return 1; // Safe default
    }
  }

  /**
   * Get Android memory info
   */
  private async getAndroidMemoryInfo(): Promise<{ totalRAM: number; freeRAM: number; usedRAM: number }> {
    try {
      // This would require native Android implementation
      // For now, return reasonable defaults
      return {
        totalRAM: 4096, // MB
        freeRAM: 2048, // MB
        usedRAM: 2048, // MB
      };
    } catch (error) {
      this.logger.error('Failed to get Android memory info:', error);
      return { totalRAM: 0, freeRAM: 0, usedRAM: 0 };
    }
  }

  /**
   * Get Android storage info
   */
  private async getAndroidStorageInfo(): Promise<{ totalDisk: number; freeDisk: number; usedDisk: number }> {
    try {
      // This would require native Android implementation
      // For now, return reasonable defaults
      return {
        totalDisk: 65536, // MB (64GB)
        freeDisk: 32768, // MB (32GB)
        usedDisk: 32768, // MB (32GB)
      };
    } catch (error) {
      this.logger.error('Failed to get Android storage info:', error);
      return { totalDisk: 0, freeDisk: 0, usedDisk: 0 };
    }
  }

  /**
   * Get Android battery info
   */
  private async getAndroidBatteryInfo(): Promise<{ level: number; isCharging: boolean }> {
    try {
      // This would require native Android implementation
      // For now, return reasonable defaults
      return {
        level: 80, // Percentage
        isCharging: false,
      };
    } catch (error) {
      this.logger.error('Failed to get Android battery info:', error);
      return { level: 0, isCharging: false };
    }
  }

  /**
   * Get Android GPU model
   */
  private async getAndroidGPUModel(): Promise<string | undefined> {
    try {
      // This would require native Android implementation
      return 'Adreno GPU'; // Common Android GPU
    } catch (error) {
      this.logger.error('Failed to get Android GPU model:', error);
      return undefined;
    }
  }

  /**
   * Get iOS memory info
   */
  private async getIOSMemoryInfo(): Promise<{ totalRAM: number; freeRAM: number; usedRAM: number }> {
    try {
      // This would require native iOS implementation
      // For now, return reasonable defaults
      return {
        totalRAM: 3072, // MB (3GB - typical for iPhone)
        freeRAM: 1536, // MB
        usedRAM: 1536, // MB
      };
    } catch (error) {
      this.logger.error('Failed to get iOS memory info:', error);
      return { totalRAM: 0, freeRAM: 0, usedRAM: 0 };
    }
  }

  /**
   * Get iOS storage info
   */
  private async getIOSStorageInfo(): Promise<{ totalDisk: number; freeDisk: number; usedDisk: number }> {
    try {
      // This would require native iOS implementation
      // For now, return reasonable defaults
      return {
        totalDisk: 65536, // MB (64GB)
        freeDisk: 32768, // MB (32GB)
        usedDisk: 32768, // MB (32GB)
      };
    } catch (error) {
      this.logger.error('Failed to get iOS storage info:', error);
      return { totalDisk: 0, freeDisk: 0, usedDisk: 0 };
    }
  }

  /**
   * Get iOS battery info
   */
  private async getIOSBatteryInfo(): Promise<{ level: number; isCharging: boolean }> {
    try {
      // This would require native iOS implementation
      // For now, return reasonable defaults
      return {
        level: 75, // Percentage
        isCharging: false,
      };
    } catch (error) {
      this.logger.error('Failed to get iOS battery info:', error);
      return { level: 0, isCharging: false };
    }
  }

  /**
   * Destroy device utility
   */
  public destroy(): void {
    this.deviceInfo = null;
    this.isInitialized = false;
    this.logger.info('Device utility destroyed');
  }
}