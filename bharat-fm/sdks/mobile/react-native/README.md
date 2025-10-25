# Bharat AI React Native SDK

A comprehensive React Native SDK for integrating Bharat Foundation Model Framework capabilities into mobile applications. The SDK provides seamless access to India's sovereign AI capabilities across 22+ Indian languages and multiple domains.

## Features

### 🌍 Multi-Language Support
- Native support for 22+ Indian languages including Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, and more
- Automatic language detection
- Real-time translation capabilities
- Multilingual text generation

### 🏛️ Domain-Specific AI Capabilities
- **Language AI**: Translation, language detection, multilingual processing
- **Governance AI**: RTI response generation, policy analysis, compliance auditing
- **Education AI**: Tutoring sessions, content generation, progress tracking
- **Finance AI**: Financial analysis, transaction auditing, risk assessment

### 📱 Mobile-Optimized Features
- Offline mode with local caching
- Platform-specific optimizations (iOS/Android)
- Battery and memory optimization
- Background processing support
- Real-time streaming responses

### 🔧 Developer Experience
- TypeScript support with comprehensive type definitions
- Easy-to-use React components
- Custom React hooks
- Comprehensive error handling
- Automatic retry mechanisms
- Detailed logging and debugging

## Installation

```bash
npm install @bharat-ai/react-native-sdk
```

### Dependencies

The SDK requires the following peer dependencies:

```bash
npm install react-native react-native-async-storage/async-storage react-native-device-info axios
```

### iOS Setup

1. Install iOS dependencies:
```bash
cd ios && pod install
```

2. Add necessary permissions to `Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access for certain features</string>
<key>NSMicrophoneUsageDescription</key>
<string>Microphone access for voice features</string>
<key>NSLocationWhenInUseUsageDescription</key>
<string>Location access for location-based features</string>
```

### Android Setup

1. Add necessary permissions to `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

## Quick Start

### 1. Initialize the SDK

```typescript
import { initializeSDK, DEFAULT_CONFIG } from '@bharat-ai/react-native-sdk';

// Initialize with custom configuration
const config = {
  ...DEFAULT_CONFIG,
  baseURL: 'https://api.bharat-ai.example.com',
  apiKey: 'your-api-key',
  enableLogging: true,
};

const client = await initializeSDK(config);
```

### 2. Using React Hooks

```typescript
import { useBharatAI, useLanguageDetection } from '@bharat-ai/react-native-sdk';

function MyComponent() {
  const { client, isInitialized, error } = useBharatAI({
    config: { enableLogging: true },
    autoInitialize: true,
  });

  const { detectLanguage, detectedLanguage, confidence } = useLanguageDetection({
    client,
    autoDetect: true,
  });

  const handleTextAnalysis = async (text: string) => {
    const result = await detectLanguage(text);
    console.log(`Detected: ${result.language} (${result.confidence}% confidence)`);
  };

  // ... component logic
}
```

### 3. Using React Components

#### Chat Component
```typescript
import { ChatComponent } from '@bharat-ai/react-native-sdk';

function ChatScreen() {
  return (
    <ChatComponent
      client={client}
      title="Bharat AI Assistant"
      enableStreaming={true}
      onMessageSend={(message) => console.log('Sent:', message)}
      onMessageReceive={(response) => console.log('Received:', response)}
    />
  );
}
```

#### Translation Component
```typescript
import { TranslationComponent } from '@bharat-ai/react-native-sdk';

function TranslationScreen() {
  return (
    <TranslationComponent
      client={client}
      enableAutoDetection={true}
      showConfidence={true}
      onTranslationComplete={(result) => console.log('Translation:', result)}
    />
  );
}
```

#### Tutoring Component
```typescript
import { TutoringComponent } from '@bharat-ai/react-native-sdk';

function EducationScreen() {
  return (
    <TutoringComponent
      client={client}
      onSessionGenerated={(session) => console.log('Session:', session)}
    />
  );
}
```

#### RTI Assistant Component
```typescript
import { RTIAssistantComponent } from '@bharat-ai/react-native-sdk';

function GovernanceScreen() {
  return (
    <RTIAssistantComponent
      client={client}
      onRTIGenerated={(rti) => console.log('RTI:', rti)}
    />
  );
}
```

## API Reference

### Core Classes

#### BharatAIClient

The main client class for interacting with the Bharat AI API.

```typescript
const client = new BharatAIClient(config);
await client.initialize();

// Generate text
const response = await client.generateText({
  prompt: 'Hello, how are you?',
  maxTokens: 100,
  temperature: 0.7,
});

// Generate text with streaming
await client.generateTextStream(request, (chunk) => {
  console.log('Stream chunk:', chunk.text);
});

// Generate embeddings
const embeddings = await client.generateEmbeddings({
  text: 'Sample text for embedding',
  dimensions: 768,
});

// Get health status
const health = await client.getHealth();
```

### Domain Clients

#### LanguageAIClient

```typescript
const languageClient = new LanguageAIClient(client);

// Translate text
const translation = await languageClient.translate({
  text: 'Hello world',
  sourceLanguage: 'en',
  targetLanguage: 'hi',
});

// Detect language
const detection = await languageClient.detectLanguage({
  text: 'नमस्ते दुनिया',
  includeConfidence: true,
});

// Generate multilingual content
const multilingual = await languageClient.generateMultilingual({
  prompt: 'Hello world',
  targetLanguages: ['hi', 'ta', 'te'],
  parallel: true,
});
```

#### GovernanceAIClient

```typescript
const governanceClient = new GovernanceAIClient(client);

// Generate RTI application
const rti = await governanceClient.generateRTI({
  subject: 'Request for information on public spending',
  publicAuthority: 'Ministry of Finance',
  informationSought: 'Details of public spending in the last fiscal year',
  applicantDetails: {
    name: 'John Doe',
    address: '123 Main Street, City',
  },
});

// Analyze policy
const policyAnalysis = await governanceClient.analyzePolicy({
  policyText: 'Policy document content...',
  analysisType: 'impact',
  includeRecommendations: true,
});

// Conduct compliance audit
const audit = await governanceClient.conductComplianceAudit({
  target: 'Organization XYZ',
  framework: 'Regulatory Compliance',
  scope: ['Financial', 'Operational'],
});
```

#### EducationAIClient

```typescript
const educationClient = new EducationAIClient(client);

// Generate tutoring session
const session = await educationClient.generateTutoringSession({
  subject: 'Mathematics',
  gradeLevel: 'High School (9-12)',
  topic: 'Quadratic Equations',
  currentUnderstanding: 'intermediate',
  duration: 30,
});

// Generate educational content
const content = await educationClient.generateContent({
  contentType: 'lesson',
  subject: 'Science',
  gradeLevel: 'Middle School (6-8)',
  topics: ['Photosynthesis', 'Plant Biology'],
  includeVisuals: true,
});

// Track progress
const progress = await educationClient.trackProgress({
  studentId: 'student123',
  subject: 'Mathematics',
  completedActivities: [
    {
      activityType: 'lesson',
      topic: 'Algebra',
      score: 85,
      timeSpent: 45,
      completionDate: '2024-01-15',
    },
  ],
});
```

#### FinanceAIClient

```typescript
const financeClient = new FinanceAIClient(client);

// Analyze financials
const analysis = await financeClient.analyzeFinancials({
  analysisType: 'investment',
  financialData: 'Financial data...',
  riskTolerance: 'medium',
  investmentHorizon: '5 years',
});

// Audit transactions
const audit = await financeClient.auditTransactions({
  transactionData: 'Transaction records...',
  auditType: 'compliance',
  timePeriod: 'Q4 2023',
});

// Assess risk
const risk = await financeClient.assessRisk({
  entity: 'Company ABC',
  assessmentType: 'credit',
  assessmentData: 'Company data...',
  timeHorizon: '1 year',
});
```

### React Hooks

#### useBharatAI

```typescript
const {
  client,
  isInitialized,
  isLoading,
  error,
  initialize,
  updateConfig,
  getConfig,
  reset,
} = useBharatAI({
  config: { enableLogging: true },
  autoInitialize: true,
});
```

#### useLanguageDetection

```typescript
const {
  detectedLanguage,
  detectedLanguageName,
  confidence,
  isDetecting,
  error,
  detectLanguage,
  clearResults,
  alternatives,
} = useLanguageDetection({
  client,
  autoDetect: true,
  debounceTime: 500,
});
```

#### useOfflineMode

```typescript
const {
  isOnline,
  isOfflineMode,
  isSyncing,
  pendingOperations,
  lastSync,
  enableOfflineMode,
  disableOfflineMode,
  syncNow,
  clearCache,
  getOfflineStats,
} = useOfflineMode({
  client,
  enabled: true,
  autoSync: true,
  syncInterval: 30000,
});
```

## Configuration

### Client Configuration

```typescript
interface ClientConfig {
  baseURL: string;                    // API base URL
  timeout: number;                   // Request timeout in ms
  maxRetries: number;                // Maximum retry attempts
  retryDelay: number;                // Delay between retries in ms
  enableOfflineMode: boolean;        // Enable offline mode
  cacheResponses: boolean;           // Cache API responses
  enableLogging: boolean;            // Enable debug logging
  apiKey?: string;                  // API key for authentication
  defaultLanguage?: string;          // Default language
  customHeaders?: Record<string, string>; // Custom headers
  enableCompression?: boolean;       // Enable request compression
  maxCacheSize?: number;             // Max cache size in MB
  cacheExpiration?: number;          // Cache expiration in hours
}
```

### Platform-Specific Configuration

The SDK automatically detects the platform and applies optimal configurations:

```typescript
// iOS-specific optimizations
const iosConfig = {
  timeout: 30000,
  maxRetries: 3,
  enableGPU: true,
  maxConcurrentRequests: 4,
};

// Android-specific optimizations
const androidConfig = {
  timeout: 35000,
  maxRetries: 4,
  enableGPU: true,
  maxConcurrentRequests: 3,
};
```

## Error Handling

The SDK provides comprehensive error handling with custom error classes:

```typescript
try {
  const response = await client.generateText(request);
} catch (error) {
  if (error instanceof BharatAIError) {
    console.error('Bharat AI Error:', error.code, error.message);
    console.error('Should retry:', error.shouldRetry());
    console.error('Retry delay:', error.getRetryDelay());
    
    if (error.shouldRetry()) {
      // Implement retry logic
    }
  } else {
    console.error('Unknown error:', error);
  }
}
```

### Error Types

- `NETWORK_ERROR`: Network connectivity issues
- `TIMEOUT_ERROR`: Request timeout
- `AUTHENTICATION_ERROR`: Authentication failed
- `AUTHORIZATION_ERROR`: Authorization failed
- `RATE_LIMIT_ERROR`: Rate limit exceeded
- `VALIDATION_ERROR`: Invalid request data
- `SERVER_ERROR`: Server-side errors
- `CACHE_ERROR`: Cache-related errors
- `DEVICE_ERROR`: Device-related errors

## Offline Mode

The SDK supports offline mode with local caching and automatic synchronization:

```typescript
// Enable offline mode
await client.updateConfig({ enableOfflineMode: true });

// Check offline status
const offlineStats = await client.getCacheStats();
console.log('Cache size:', offlineStats.approximateSizeBytes);

// Force sync when online
if (await networkManager.isOnline()) {
  await syncNow();
}
```

## Performance Optimization

### Memory Management

```typescript
// Enable memory optimization
client.updateConfig({
  enableCompression: true,
  maxCacheSize: 50, // 50MB cache limit
});
```

### Battery Optimization

```typescript
// Enable battery optimization
client.updateConfig({
  batteryOptimization: true,
  enableOfflineMode: true,
});
```

### Network Optimization

```typescript
// Optimize network requests
client.updateConfig({
  enableCompression: true,
  maxRetries: 3,
  retryDelay: 1000,
});
```

## Security

### Data Encryption

```typescript
// Enable encryption for cached data
const storageUtil = new StorageUtil({
  encryption: true,
  encryptionKey: 'your-encryption-key',
});
```

### Authentication

```typescript
// Configure authentication
const config = {
  apiKey: 'your-api-key',
  customHeaders: {
    'Authorization': 'Bearer your-token',
  },
};
```

## Best Practices

### 1. Initialize Once
```typescript
// Initialize the SDK once at app startup
const client = await initializeSDK(config);
```

### 2. Handle Errors Gracefully
```typescript
try {
  const response = await client.generateText(request);
} catch (error) {
  // Handle errors appropriately
  showErrorToast(error.message);
}
```

### 3. Use Streaming for Long Responses
```typescript
// Use streaming for better user experience
await client.generateTextStream(request, (chunk) => {
  updateUI(chunk.text);
});
```

### 4. Optimize for Mobile
```typescript
// Use platform-specific optimizations
const config = {
  ...DEFAULT_CONFIG,
  enableOfflineMode: true,
  batteryOptimization: true,
  maxConcurrentRequests: Platform.OS === 'ios' ? 4 : 3,
};
```

### 5. Monitor Performance
```typescript
// Monitor SDK performance
const stats = await client.getCacheStats();
console.log('Performance stats:', stats);
```

## Troubleshooting

### Common Issues

#### 1. Network Errors
```typescript
// Check network connectivity
const isOnline = await networkManager.isOnline();
if (!isOnline) {
  // Show offline message
  showMessage('No internet connection');
}
```

#### 2. Authentication Errors
```typescript
// Check API key configuration
if (!config.apiKey) {
  console.error('API key is required');
}
```

#### 3. Memory Issues
```typescript
// Clear cache if memory is low
if (memoryWarning) {
  await client.clearCache();
}
```

#### 4. Platform-Specific Issues
```typescript
// Handle platform-specific requirements
if (Platform.OS === 'ios') {
  // iOS-specific handling
} else if (Platform.OS === 'android') {
  // Android-specific handling
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This SDK is licensed under the Apache License 2.0. See the LICENSE file for details.

## Support

For support, please:
- Check the documentation
- Search existing issues
- Create a new issue with detailed information
- Contact the Bharat AI team

## Changelog

### v1.0.0
- Initial release
- Support for 22+ Indian languages
- Domain-specific AI capabilities
- React Native components and hooks
- Offline mode support
- Platform-specific optimizations
- Comprehensive error handling
- TypeScript support