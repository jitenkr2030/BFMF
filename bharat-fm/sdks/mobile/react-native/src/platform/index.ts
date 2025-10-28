/**
 * Platform-specific utilities for Bharat AI SDK
 */

import { Platform } from 'react-native';

// Platform detection
export const isIOS = Platform.OS === 'ios';
export const isAndroid = Platform.OS === 'android';

// Platform-specific configurations
export const PlatformConfig = {
  ios: {
    defaultTimeout: 30000,
    maxRetries: 3,
    retryDelay: 1000,
    enableCompression: true,
    enableGPU: true,
    maxConcurrentRequests: 4,
    batteryOptimization: true,
  },
  android: {
    defaultTimeout: 35000,
    maxRetries: 4,
    retryDelay: 1500,
    enableCompression: true,
    enableGPU: true,
    maxConcurrentRequests: 3,
    batteryOptimization: true,
  },
};

// Get platform-specific configuration
export function getPlatformConfig() {
  return isIOS ? PlatformConfig.ios : PlatformConfig.android;
}

// Platform-specific optimizations
export const PlatformOptimizations = {
  ios: {
    useNativeModules: true,
    enableMetal: true,
    preferCoreML: true,
    enableBackgroundProcessing: true,
    useSiriShortcuts: true,
  },
  android: {
    useNativeModules: true,
    enableNNAPI: true,
    preferTensorFlowLite: true,
    enableBackgroundServices: true,
    useFirebase: true,
  },
};

// Get platform-specific optimizations
export function getPlatformOptimizations() {
  return isIOS ? PlatformOptimizations.ios : PlatformOptimizations.android;
}

// Platform-specific permissions
export const PlatformPermissions = {
  ios: [
    'camera',
    'microphone',
    'photo_library',
    'contacts',
    'location',
    'notifications',
  ],
  android: [
    'camera',
    'microphone',
    'read_external_storage',
    'write_external_storage',
    'access_fine_location',
    'access_coarse_location',
    'read_contacts',
    'write_contacts',
  ],
};

// Get platform-specific permissions
export function getPlatformPermissions() {
  return isIOS ? PlatformPermissions.ios : PlatformPermissions.android;
}

// Platform-specific features
export const PlatformFeatures = {
  ios: {
    faceId: true,
    touchId: true,
    siri: true,
    applePay: true,
    healthKit: true,
    homeKit: true,
    arKit: true,
  },
  android: {
    fingerprint: true,
    faceUnlock: true,
    googlePay: true,
    healthConnect: true,
    nearbyConnections: true,
    arCore: true,
  },
};

// Get platform-specific features
export function getPlatformFeatures() {
  return isIOS ? PlatformFeatures.ios : PlatformFeatures.android;
}

// Platform-specific UI adjustments
export const PlatformUI = {
  ios: {
    statusBarStyle: 'dark-content',
    navigationBarColor: '#ffffff',
    tabBarColor: '#ffffff',
    headerTintColor: '#007AFF',
    buttonStyle: 'rounded',
  },
  android: {
    statusBarStyle: 'light-content',
    navigationBarColor: '#007AFF',
    tabBarColor: '#ffffff',
    headerTintColor: '#ffffff',
    buttonStyle: 'rounded-rectangle',
  },
};

// Get platform-specific UI adjustments
export function getPlatformUI() {
  return isIOS ? PlatformUI.ios : PlatformUI.android;
}

// Platform-specific performance settings
export const PlatformPerformance = {
  ios: {
    animationEnabled: true,
    gestureEnabled: true,
    shadowEnabled: true,
    blurEnabled: true,
    maxTextureSize: 4096,
  },
  android: {
    animationEnabled: true,
    gestureEnabled: true,
    shadowEnabled: false,
    blurEnabled: false,
    maxTextureSize: 2048,
  },
};

// Get platform-specific performance settings
export function getPlatformPerformance() {
  return isIOS ? PlatformPerformance.ios : PlatformPerformance.android;
}

// Platform-specific network settings
export const PlatformNetwork = {
  ios: {
    defaultTimeout: 30000,
    enableTLS: true,
    enableCompression: true,
    enableCaching: true,
    maxConcurrentConnections: 6,
  },
  android: {
    defaultTimeout: 35000,
    enableTLS: true,
    enableCompression: true,
    enableCaching: true,
    maxConcurrentConnections: 5,
  },
};

// Get platform-specific network settings
export function getPlatformNetwork() {
  return isIOS ? PlatformNetwork.ios : PlatformNetwork.android;
}

// Platform-specific storage settings
export const PlatformStorage = {
  ios: {
    useKeychain: true,
    useICloud: true,
    encryptionEnabled: true,
    maxCacheSize: 100, // MB
  },
  android: {
    useKeystore: true,
    useFirebase: true,
    encryptionEnabled: true,
    maxCacheSize: 150, // MB
  },
};

// Get platform-specific storage settings
export function getPlatformStorage() {
  return isIOS ? PlatformStorage.ios : PlatformStorage.android;
}

// Platform-specific debugging
export const PlatformDebug = {
  ios: {
    enableConsole: true,
    enableNetworkLogger: true,
    enablePerformanceMonitor: true,
    enableCrashReporting: true,
  },
  android: {
    enableConsole: true,
    enableNetworkLogger: true,
    enablePerformanceMonitor: true,
    enableCrashReporting: true,
  },
};

// Get platform-specific debugging
export function getPlatformDebug() {
  return isIOS ? PlatformDebug.ios : PlatformDebug.android;
}

// Platform-specific testing
export const PlatformTesting = {
  ios: {
    useDetox: true,
    useXCTest: true,
    enableScreenshotTesting: true,
    enableAccessibilityTesting: true,
  },
  android: {
    useDetox: true,
    useEspresso: true,
    enableScreenshotTesting: true,
    enableAccessibilityTesting: true,
  },
};

// Get platform-specific testing
export function getPlatformTesting() {
  return isIOS ? PlatformTesting.ios : PlatformTesting.android;
}

// Platform-specific deployment
export const PlatformDeployment = {
  ios: {
    appStore: true,
    testFlight: true,
    enterprise: true,
    adHoc: true,
    simulator: true,
  },
  android: {
    playStore: true,
    internalTesting: true,
    alphaTesting: true,
    betaTesting: true,
    emulator: true,
  },
};

// Get platform-specific deployment
export function getPlatformDeployment() {
  return isIOS ? PlatformDeployment.ios : PlatformDeployment.android;
}

// Export all platform utilities
export * from './ios';
export * from './android';