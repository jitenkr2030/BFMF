# Bharat Foundation Model Framework - JavaScript/TypeScript SDK

The official JavaScript/TypeScript SDK for the Bharat Foundation Model Framework (BFMF), enabling seamless integration of India's sovereign AI capabilities into your applications.

## 🌟 Features

- **Multi-Language Support**: Native support for 22+ Indian languages
- **Domain-Specific Clients**: Specialized clients for Language, Governance, Education, and Finance AI
- **TypeScript Support**: Full TypeScript support with comprehensive type definitions
- **Easy Integration**: Simple REST API client with automatic retries and error handling
- **Streaming Support**: Real-time text generation with streaming capabilities
- **Production Ready**: Built for production with robust error handling and logging

## 📦 Installation

### npm
```bash
npm install @bharat-ai/sdk
```

### yarn
```bash
yarn add @bharat-ai/sdk
```

### pnpm
```bash
pnpm add @bharat-ai/sdk
```

## 🚀 Quick Start

### Basic Usage

```typescript
import { BharatAIClient } from '@bharat-ai/sdk';

// Initialize the client
const client = new BharatAIClient({
  baseURL: 'http://localhost:8000', // Your BFMF server URL
  apiKey: 'your-api-key', // Optional
  timeout: 30000,
  debug: false
});

// Generate text
const response = await client.generateText({
  prompt: 'नमस्ते, आप कैसे हैं?',
  maxTokens: 100,
  language: 'hi'
});

console.log(response.generatedText);
```

### Domain-Specific Usage

```typescript
import { 
  BharatAIClient, 
  LanguageAIClient, 
  GovernanceAIClient,
  EducationAIClient,
  FinanceAIClient 
} from '@bharat-ai/sdk';

const client = new BharatAIClient();

// Language AI - Translate text
const languageClient = new LanguageAIClient(client);
const translation = await languageClient.translate({
  text: 'Hello, how are you?',
  sourceLanguage: 'en',
  targetLanguage: 'hi'
});

console.log(translation.translatedText); // "नमस्ते, आप कैसे हैं?"

// Governance AI - Generate RTI response
const governanceClient = new GovernanceAIClient(client);
const rtiResponse = await governanceClient.generateRTIResponse({
  applicationText: 'I would like to know about...',
  department: 'Ministry of Health',
  requestType: 'information'
});

// Education AI - Start tutoring session
const educationClient = new EducationAIClient(client);
const tutoringSession = await educationClient.startTutoringSession({
  subject: 'Mathematics',
  topic: 'Algebra',
  studentLevel: 'secondary',
  language: 'en'
});

// Finance AI - Analyze financial data
const financeClient = new FinanceAIClient(client);
const analysis = await financeClient.analyzeFinancials({
  financialData: {
    balanceSheet: { assets: 1000000, liabilities: 600000 },
    incomeStatement: { revenue: 500000, expenses: 300000 }
  },
  analysisType: 'ratio'
});
```

### Streaming Generation

```typescript
import { BharatAIClient } from '@bharat-ai/sdk';

const client = new BharatAIClient();

// Stream text generation
for await (const chunk of client.generateTextStream({
  prompt: 'Tell me about Indian culture',
  maxTokens: 200
})) {
  console.log(chunk); // Generated text chunks in real-time
}
```

### Batch Processing

```typescript
import { BharatAIClient } from '@bharat-ai/sdk';

const client = new BharatAIClient();

// Generate text for multiple prompts
const batchResponse = await client.generateTextBatch({
  requests: [
    { prompt: 'What is AI?' },
    { prompt: 'What is machine learning?' },
    { prompt: 'What is deep learning?' }
  ]
});

console.log(batchResponse.responses);
```

## 📚 API Reference

### BharatAIClient

The main client class for interacting with BFMF APIs.

#### Constructor

```typescript
new BharatAIClient(config: ClientConfig)
```

**Parameters:**
- `config.baseURL` (string): Base URL for the API (default: 'http://localhost:8000')
- `config.apiKey` (string): API key for authentication (optional)
- `config.timeout` (number): Request timeout in milliseconds (default: 30000)
- `config.maxRetries` (number): Maximum number of retries (default: 3)
- `config.retryDelay` (number): Delay between retries in milliseconds (default: 1000)
- `config.debug` (boolean): Enable debug logging (default: false)

#### Methods

##### `generateText(request: GenerationRequest): Promise<GenerationResponse>`

Generate text from a prompt.

##### `generateTextBatch(request: BatchGenerationRequest): Promise<BatchGenerationResponse>`

Generate text for multiple prompts.

##### `generateTextStream(request: GenerationRequest): AsyncGenerator<string>`

Stream text generation in real-time.

##### `getEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse>`

Get text embeddings.

##### `getModelInfo(): Promise<ModelInfo>`

Get model information.

##### `getHealth(): Promise<HealthResponse>`

Check API health.

##### `getSupportedLanguages(): Promise<SupportedLanguagesResponse>`

Get supported languages.

### Domain-Specific Clients

#### LanguageAIClient

Specialized client for Language AI capabilities.

**Key Methods:**
- `translate(request: TranslationRequest): Promise<TranslationResponse>`
- `detectLanguage(request: LanguageDetectionRequest): Promise<LanguageDetectionResponse>`
- `detectCodeSwitching(request: CodeSwitchingRequest): Promise<CodeSwitchingResponse>`
- `tokenize(request: TokenizationRequest): Promise<TokenizationResponse>`

#### GovernanceAIClient

Specialized client for Governance AI capabilities.

**Key Methods:**
- `generateRTIResponse(request: RTIRequest): Promise<RTIResponse>`
- `analyzePolicy(request: PolicyAnalysisRequest): Promise<PolicyAnalysisResponse>`
- `conductComplianceAudit(request: ComplianceAuditRequest): Promise<ComplianceAuditResponse>`
- `analyzeScheme(request: SchemeAnalysisRequest): Promise<SchemeAnalysisResponse>`

#### EducationAIClient

Specialized client for Education AI capabilities.

**Key Methods:**
- `startTutoringSession(request: TutoringRequest): Promise<TutoringResponse>`
- `generateContent(request: ContentGenerationRequest): Promise<ContentGenerationResponse>`
- `generateAssessment(request: AssessmentGenerationRequest): Promise<AssessmentResponse>`
- `trackProgress(request: ProgressTrackingRequest): Promise<ProgressTrackingResponse>`

#### FinanceAIClient

Specialized client for Finance AI capabilities.

**Key Methods:**
- `analyzeFinancials(request: FinancialAnalysisRequest): Promise<FinancialAnalysisResponse>`
- `auditTransactions(request: TransactionAuditRequest): Promise<TransactionAuditResponse>`
- `assessRisk(request: RiskAssessmentRequest): Promise<RiskAssessmentResponse>`
- `predictMarket(request: MarketPredictionRequest): Promise<MarketPredictionResponse>`

## 🌐 Supported Languages

The SDK supports all 22+ Indian languages:

- Hindi (hi)
- Bengali (bn)
- Tamil (ta)
- Telugu (te)
- Marathi (mr)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)
- And many more...

## 🛠️ Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm/sdks/javascript

# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test

# Run linting
npm run lint

# Format code
npm run format
```

### Testing

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Watch mode for development
npm run test:watch
```

## 📝 Examples

### Web Application

```typescript
// React example
import { useState } from 'react';
import { BharatAIClient } from '@bharat-ai/sdk';

const client = new BharatAIClient({
  baseURL: process.env.BFMF_API_URL,
  apiKey: process.env.BFMF_API_KEY
});

function ChatComponent() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const result = await client.generateText({
        prompt: input,
        maxTokens: 200,
        language: 'en'
      });
      setResponse(result.generatedText);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter your prompt..."
      />
      <button type="submit" disabled={loading}>
        {loading ? 'Generating...' : 'Generate'}
      </button>
      {response && <div>{response}</div>}
    </form>
  );
}
```

### Node.js Backend

```typescript
// Express.js example
import express from 'express';
import { BharatAIClient, LanguageAIClient } from '@bharat-ai/sdk';

const app = express();
const client = new BharatAIClient();
const languageClient = new LanguageAIClient(client);

app.post('/translate', async (req, res) => {
  try {
    const { text, sourceLanguage, targetLanguage } = req.body;
    
    const translation = await languageClient.translate({
      text,
      sourceLanguage,
      targetLanguage
    });
    
    res.json(translation);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](../../CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This SDK is licensed under the Apache License 2.0. See the [LICENSE](../../LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full Documentation](https://bharat-ai.github.io/bharat-fm/)
- **Issues**: [Report Bugs](https://github.com/bharat-ai/bharat-fm/issues)
- **Discussions**: [Community Forum](https://github.com/bharat-ai/bharat-fm/discussions)
- **Discord**: [Join our Discord](https://discord.gg/bharat-ai)

## 🙏 Acknowledgments

- Bharat AI Team for developing the framework
- Open source community for inspiration and tools
- Contributors who help improve this SDK

---

**Made with ❤️ for Bharat's AI Independence**