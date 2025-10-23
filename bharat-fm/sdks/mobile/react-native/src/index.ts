/**
 * Bharat Foundation Model Framework - React Native SDK
 * 
 * This SDK provides React Native components and utilities for integrating
 * BFMF capabilities into mobile applications.
 */

import { Platform } from 'react-native';
import BharatAIClient from './client/BharatAIClient';
import { ClientConfig } from './types/ClientConfig';
import {
  GenerationRequest,
  GenerationResponse,
  EmbeddingRequest,
  EmbeddingResponse,
  ModelInfo,
  HealthResponse,
  SupportedLanguagesResponse
} from './types/ApiTypes';

// Main client export
export { BharatAIClient };

// Type exports
export type {
  ClientConfig,
  GenerationRequest,
  GenerationResponse,
  EmbeddingRequest,
  EmbeddingResponse,
  ModelInfo,
  HealthResponse,
  SupportedLanguagesResponse
};

// Re-export domain clients
export { LanguageAIClient } from './domains/LanguageAIClient';
export { GovernanceAIClient } from './domains/GovernanceAIClient';
export { EducationAIClient } from './domains/EducationAIClient';
export { FinanceAIClient } from './domains/FinanceAIClient';

// Components
export { ChatComponent } from './components/ChatComponent';
export { TranslationComponent } from './components/TranslationComponent';
export { TutoringComponent } from './components/TutoringComponent';
export { RTIAssistantComponent } from './components/RTIAssistantComponent';

// Hooks
export { useBharatAI } from './hooks/useBharatAI';
export { useLanguageDetection } from './hooks/useLanguageDetection';
export { useOfflineMode } from './hooks/useOfflineMode';

// Utilities
export { StorageUtil } from './utils/StorageUtil';
export { NetworkUtil } from './utils/NetworkUtil';
export { DeviceUtil } from './utils/DeviceUtil';

// Constants
export { SDK_VERSION, DEFAULT_CONFIG } from './constants';

// Platform specific utilities
export * from './platform';

// Version
export const SDK_VERSION = '1.0.0';

// Default configuration
export const DEFAULT_CONFIG: ClientConfig = {
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
  enableOfflineMode: true,
  cacheResponses: true,
  enableLogging: Platform.OS === 'ios' ? false : true,
};

/**
 * Create a new BFMF client with default configuration
 */
export function createClient(config?: Partial<ClientConfig>): BharatAIClient {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  return new BharatAIClient(finalConfig);
}

/**
 * Initialize the BFMF SDK with optional configuration
 */
export async function initializeSDK(config?: Partial<ClientConfig>): Promise<BharatAIClient> {
  const client = createClient(config);
  
  // Initialize platform-specific features
  if (Platform.OS === 'android') {
    await initializeAndroidFeatures(client);
  } else if (Platform.OS === 'ios') {
    await initializeIOSFeatures(client);
  }
  
  return client;
}

/**
 * Initialize Android-specific features
 */
async function initializeAndroidFeatures(client: BharatAIClient): Promise<void> {
  // Android-specific initialization
  // This could include:
  // - Firebase integration
  // - Push notifications
  // - Background services
  // - Device-specific optimizations
}

/**
 * Initialize iOS-specific features
 */
async function initializeIOSFeatures(client: BharatAIClient): Promise<void> {
  // iOS-specific initialization
  // This could include:
  // - Core ML integration
  // - Siri shortcuts
  // - Background app refresh
  // - Device-specific optimizations
}

// Default export
export default {
  createClient,
  initializeSDK,
  SDK_VERSION,
  BharatAIClient,
};