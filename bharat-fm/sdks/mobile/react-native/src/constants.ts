/**
 * Constants for the Bharat AI React Native SDK
 */

import { Platform } from 'react-native';
import { ClientConfig } from './types/ClientConfig';

// SDK Version
export const SDK_VERSION = '1.0.0';

// Default Configuration
export const DEFAULT_CONFIG: ClientConfig = {
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
  enableOfflineMode: true,
  cacheResponses: true,
  enableLogging: Platform.OS === 'ios' ? false : true,
};

// API Endpoints
export const API_ENDPOINTS = {
  generate: '/generate',
  generateStream: '/generate/stream',
  generateBatch: '/generate/batch',
  embeddings: '/embeddings',
  models: '/models',
  health: '/health',
  languages: '/languages',
  
  // Domain-specific endpoints
  language: {
    translate: '/language/translate',
    detect: '/language/detect',
    multilingual: '/language/multilingual',
  },
  governance: {
    rti: '/governance/rti',
    policy: '/governance/policy',
    audit: '/governance/audit',
  },
  education: {
    tutoring: '/education/tutoring',
    content: '/education/content',
    progress: '/education/progress',
  },
  finance: {
    analysis: '/finance/analysis',
    audit: '/finance/audit',
    risk: '/finance/risk',
  },
};

// HTTP Status Codes
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  ACCEPTED: 202,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
  GATEWAY_TIMEOUT: 504,
};

// Error Codes
export const ERROR_CODES = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  AUTHENTICATION_ERROR: 'AUTHENTICATION_ERROR',
  AUTHORIZATION_ERROR: 'AUTHORIZATION_ERROR',
  RATE_LIMIT_ERROR: 'RATE_LIMIT_ERROR',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
  NOT_FOUND_ERROR: 'NOT_FOUND_ERROR',
  INITIALIZATION_FAILED: 'INITIALIZATION_FAILED',
  CACHE_ERROR: 'CACHE_ERROR',
  CONFIGURATION_ERROR: 'CONFIGURATION_ERROR',
  LANGUAGE_AI_ERROR: 'LANGUAGE_AI_ERROR',
  GOVERNANCE_AI_ERROR: 'GOVERNANCE_AI_ERROR',
  EDUCATION_AI_ERROR: 'EDUCATION_AI_ERROR',
  FINANCE_AI_ERROR: 'FINANCE_AI_ERROR',
  OFFLINE_ERROR: 'OFFLINE_ERROR',
  SYNC_ERROR: 'SYNC_ERROR',
  STORAGE_ERROR: 'STORAGE_ERROR',
  DEVICE_ERROR: 'DEVICE_ERROR',
  PERMISSION_ERROR: 'PERMISSION_ERROR',
};

// Supported Languages
export const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English', nativeName: 'English' },
  { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी' },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்' },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు' },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी' },
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી' },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ' },
  { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം' },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ' },
  { code: 'as', name: 'Assamese', nativeName: 'অসমীয়া' },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ' },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو' },
  { code: 'sd', name: 'Sindhi', nativeName: 'سنڌي' },
  { code: 'ks', name: 'Kashmiri', nativeName: 'کٲشُر' },
  { code: 'ne', name: 'Nepali', nativeName: 'नेपाली' },
  { code: 'sa', name: 'Sanskrit', nativeName: 'संस्कृतम्' },
  { code: 'mai', name: 'Maithili', nativeName: 'मैथिली' },
  { code: 'mni', name: 'Manipuri', nativeName: 'মৈতৈলোন্' },
  { code: 'kok', name: 'Konkani', nativeName: 'कोंकणी' },
  { code: 'doi', name: 'Dogri', nativeName: 'डोगरी' },
  { code: 'brx', name: 'Bodo', nativeName: 'बड़ो' },
  { code: 'sat', name: 'Santali', nativeName: 'ᱥᱟᱱᱛᱟᱲᱤ' },
];

// Model Information
export const MODEL_INFO = {
  default: {
    name: 'bharat-gpt-v1',
    version: '1.0.0',
    description: 'Default Bharat AI model for general purposes',
    maxTokens: 2048,
    supportedLanguages: SUPPORTED_LANGUAGES.map(lang => lang.code),
  },
  language: {
    name: 'bharat-language-v1',
    version: '1.0.0',
    description: 'Specialized model for language processing',
    maxTokens: 4096,
    supportedLanguages: SUPPORTED_LANGUAGES.map(lang => lang.code),
  },
  governance: {
    name: 'bharat-governance-v1',
    version: '1.0.0',
    description: 'Specialized model for governance and policy analysis',
    maxTokens: 2048,
    supportedLanguages: ['en', 'hi'],
  },
  education: {
    name: 'bharat-education-v1',
    version: '1.0.0',
    description: 'Specialized model for educational content',
    maxTokens: 4096,
    supportedLanguages: ['en', 'hi', 'ta', 'te', 'kn', 'ml'],
  },
  finance: {
    name: 'bharat-finance-v1',
    version: '1.0.0',
    description: 'Specialized model for financial analysis',
    maxTokens: 2048,
    supportedLanguages: ['en', 'hi', 'gu', 'mr'],
  },
};

// Cache Configuration
export const CACHE_CONFIG = {
  defaultTTL: 24 * 60 * 60 * 1000, // 24 hours
  maxSize: 100, // MB
  persistent: true,
  encryptionEnabled: true,
  compressionEnabled: true,
};

// Network Configuration
export const NETWORK_CONFIG = {
  defaultTimeout: 30000,
  connectTimeout: 10000,
  readTimeout: 20000,
  writeTimeout: 20000,
  maxRetries: 3,
  retryDelay: 1000,
  retryBackoffMultiplier: 2,
  maxRetryDelay: 30000,
  enableCompression: true,
  enableCaching: true,
  enableKeepAlive: true,
  keepAliveTimeout: 30000,
  maxConcurrentConnections: 5,
};

// Performance Configuration
export const PERFORMANCE_CONFIG = {
  maxConcurrentRequests: 3,
  batchSize: 8,
  enableStreaming: true,
  enableCaching: true,
  enableCompression: true,
  enableGPUAcceleration: true,
  memoryOptimizationEnabled: true,
  batteryOptimizationEnabled: true,
  thermalOptimizationEnabled: true,
  lowPowerModeOptimization: true,
};

// Security Configuration
export const SECURITY_CONFIG = {
  enableEncryption: true,
  encryptionAlgorithm: 'AES-256-GCM',
  enableCertificatePinning: true,
  enableSSL: true,
  enableAuthentication: true,
  enableAuthorization: true,
  tokenRefreshThreshold: 300, // 5 minutes
  sessionTimeout: 3600, // 1 hour
  maxTokenAge: 86400, // 24 hours
};

// UI Configuration
export const UI_CONFIG = {
  theme: {
    primary: '#007AFF',
    secondary: '#5856D6',
    success: '#34C759',
    warning: '#FF9500',
    error: '#FF3B30',
    info: '#5AC8FA',
    background: '#F5F5F5',
    surface: '#FFFFFF',
    text: '#000000',
    textSecondary: '#8E8E93',
    border: '#C6C6C8',
  },
  typography: {
    fontFamily: Platform.OS === 'ios' ? 'San Francisco' : 'Roboto',
    fontSize: {
      xs: 12,
      sm: 14,
      md: 16,
      lg: 18,
      xl: 20,
      xxl: 24,
      xxxl: 32,
    },
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
    xxl: 24,
  },
  animation: {
    duration: {
      fast: 150,
      normal: 300,
      slow: 500,
    },
    easing: {
      linear: 'linear',
      easeIn: 'ease-in',
      easeOut: 'ease-out',
      easeInOut: 'ease-in-out',
    },
  },
};

// Event Types
export const EVENT_TYPES = {
  SDK_INITIALIZED: 'sdk_initialized',
  SDK_ERROR: 'sdk_error',
  REQUEST_START: 'request_start',
  REQUEST_SUCCESS: 'request_success',
  REQUEST_ERROR: 'request_error',
  RESPONSE_RECEIVED: 'response_received',
  CACHE_HIT: 'cache_hit',
  CACHE_MISS: 'cache_miss',
  NETWORK_ONLINE: 'network_online',
  NETWORK_OFFLINE: 'network_offline',
  AUTH_SUCCESS: 'auth_success',
  AUTH_ERROR: 'auth_error',
  TOKEN_REFRESHED: 'token_refreshed',
  SESSION_EXPIRED: 'session_expired',
  BATTERY_LOW: 'battery_low',
  MEMORY_WARNING: 'memory_warning',
  THERMAL_WARNING: 'thermal_warning',
};

// Log Levels
export const LOG_LEVELS = {
  DEBUG: 'debug',
  INFO: 'info',
  WARN: 'warn',
  ERROR: 'error',
} as const;

// Platform Detection
export const PLATFORM = {
  isIOS: Platform.OS === 'ios',
  isAndroid: Platform.OS === 'android',
  current: Platform.OS,
  version: Platform.Version,
};

// Device Capabilities
export const DEVICE_CAPABILITIES = {
  hasGPU: true,
  hasNeuralEngine: Platform.OS === 'ios',
  hasTensorFlow: true,
  hasCoreML: Platform.OS === 'ios',
  hasMLKit: Platform.OS === 'android',
  hasARKit: Platform.OS === 'ios',
  hasARCore: Platform.OS === 'android',
  hasFaceID: Platform.OS === 'ios',
  hasTouchID: Platform.OS === 'ios',
  hasFingerprint: Platform.OS === 'android',
  hasNFC: true,
  hasBluetooth: true,
  hasWiFi: true,
  hasCellular: true,
  hasGPS: true,
  hasCamera: true,
  hasMicrophone: true,
  hasAccelerometer: true,
  hasGyroscope: true,
  hasMagnetometer: true,
  hasLightSensor: true,
  hasProximitySensor: true,
  hasBarometer: Platform.OS === 'ios',
  hasStepCounter: true,
  hasHeartRate: false, // Device specific
};

// Feature Flags
export const FEATURE_FLAGS = {
  enableStreaming: true,
  enableCaching: true,
  enableOfflineMode: true,
  enableCompression: true,
  enableEncryption: true,
  enableAuthentication: true,
  enableAnalytics: false,
  enableCrashReporting: false,
  enablePerformanceMonitoring: false,
  enableNetworkLogging: false,
  enableDebugMode: false,
  enableBetaFeatures: false,
  enableExperimentalFeatures: false,
};

// Environment
export const ENVIRONMENT = {
  development: 'development',
  staging: 'staging',
  production: 'production',
} as const;

// Default User Agent
export const DEFAULT_USER_AGENT = `Bharat-AI-React-Native-SDK/${SDK_VERSION} (${Platform.OS}; ${Platform.Version})`;

// Default Headers
export const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
  'User-Agent': DEFAULT_USER_AGENT,
  'X-SDK-Version': SDK_VERSION,
  'X-Platform': Platform.OS,
  'X-Platform-Version': Platform.Version.toString(),
};

// Retry Strategies
export const RETRY_STRATEGIES = {
  exponential: {
    type: 'exponential',
    baseDelay: 1000,
    maxDelay: 30000,
    multiplier: 2,
  },
  linear: {
    type: 'linear',
    baseDelay: 1000,
    maxDelay: 10000,
    increment: 1000,
  },
  fixed: {
    type: 'fixed',
    delay: 2000,
  },
};

// Rate Limiting
export const RATE_LIMITING = {
  default: {
    requests: 100,
    window: 3600000, // 1 hour
  },
  streaming: {
    requests: 1000,
    window: 3600000, // 1 hour
  },
  embeddings: {
    requests: 500,
    window: 3600000, // 1 hour
  },
};

// Export all constants
export * from './platform';