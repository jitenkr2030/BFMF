# Bharat Foundation Model Framework - Java SDK

The official Java SDK for the Bharat Foundation Model Framework (BFMF), enabling seamless integration of India's sovereign AI capabilities into Java applications.

## 🌟 Features

- **Multi-Language Support**: Native support for 22+ Indian languages
- **Domain-Specific Clients**: Specialized clients for Language, Governance, Education, and Finance AI
- **Type Safety**: Full type safety with modern Java features
- **Easy Integration**: Simple REST API client with automatic retries and error handling
- **Async Support**: CompletableFuture-based asynchronous operations
- **Production Ready**: Built for production with robust error handling and logging
- **Maven Support**: Easy integration with Maven-based projects

## 📦 Installation

### Maven

Add the following dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>com.bharat-ai</groupId>
    <artifactId>bharat-sdk</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Gradle

Add the following dependency to your `build.gradle`:

```groovy
implementation 'com.bharat-ai:bharat-sdk:1.0.0'
```

## 🚀 Quick Start

### Basic Usage

```java
import com.bharatai.sdk.BharatAIClient;
import com.bharatai.sdk.config.ClientConfig;
import com.bharatai.sdk.models.GenerationRequest;
import com.bharatai.sdk.models.GenerationResponse;
import com.bharatai.sdk.exceptions.BharatAIException;

public class BasicExample {
    public static void main(String[] args) {
        // Initialize the client
        ClientConfig config = new ClientConfig.Builder()
            .baseURL("http://localhost:8000") // Your BFMF server URL
            .apiKey("your-api-key") // Optional
            .timeout(30000)
            .debug(false)
            .build();
            
        BharatAIClient client = new BharatAIClient(config);
        
        try {
            // Generate text
            GenerationRequest request = new GenerationRequest.Builder()
                .prompt("नमस्ते, आप कैसे हैं?")
                .maxTokens(100)
                .language("hi")
                .build();
                
            GenerationResponse response = client.generateText(request);
            
            System.out.println("Generated text: " + response.getGeneratedText());
            System.out.println("Tokens generated: " + response.getTokensGenerated());
            System.out.println("Generation time: " + response.getGenerationTime() + "s");
            
        } catch (BharatAIException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.println("Error code: " + e.getCode());
            System.err.println("Status code: " + e.getStatusCode());
        } finally {
            client.close();
        }
    }
}
```

### Domain-Specific Usage

```java
import com.bharatai.sdk.BharatAIClient;
import com.bharatai.sdk.config.ClientConfig;
import com.bharatai.sdk.domains.LanguageAIClient;
import com.bharatai.sdk.domains.GovernanceAIClient;
import com.bharatai.sdk.domains.EducationAIClient;
import com.bharatai.sdk.domains.FinanceAIClient;
import com.bharatai.sdk.exceptions.BharatAIException;

public class DomainExample {
    public static void main(String[] args) {
        BharatAIClient client = new BharatAIClient();
        
        try {
            // Language AI - Translate text
            LanguageAIClient languageClient = client.getLanguageClient();
            String translation = languageClient.translate(
                "Hello, how are you?", 
                "en", 
                "hi"
            );
            System.out.println("Translation: " + translation);
            
            // Governance AI - Generate RTI response
            GovernanceAIClient governanceClient = client.getGovernanceClient();
            String rtiResponse = governanceClient.generateRTIResponse(
                "I would like to know about...", 
                "Ministry of Health"
            );
            System.out.println("RTI Response: " + rtiResponse);
            
            // Education AI - Start tutoring session
            EducationAIClient educationClient = client.getEducationClient();
            String tutoringSession = educationClient.startTutoringSession(
                "Mathematics", 
                "Algebra", 
                "secondary"
            );
            System.out.println("Tutoring: " + tutoringSession);
            
            // Finance AI - Analyze financial data
            FinanceAIClient financeClient = client.getFinanceClient();
            String analysis = financeClient.analyzeFinancials(
                "{\"revenue\": 1000000, \"expenses\": 600000}"
            );
            System.out.println("Financial Analysis: " + analysis);
            
        } catch (BharatAIException e) {
            System.err.println("Error: " + e.getMessage());
        } finally {
            client.close();
        }
    }
}
```

### Asynchronous Operations

```java
import com.bharatai.sdk.BharatAIClient;
import com.bharatai.sdk.models.GenerationRequest;
import com.bharatai.sdk.models.GenerationResponse;
import com.bharatai.sdk.exceptions.BharatAIException;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class AsyncExample {
    public static void main(String[] args) {
        BharatAIClient client = new BharatAIClient();
        
        try {
            // Generate text asynchronously
            GenerationRequest request = new GenerationRequest.Builder()
                .prompt("What is artificial intelligence?")
                .maxTokens(150)
                .build();
                
            CompletableFuture<GenerationResponse> future = client.generateTextAsync(request);
            
            // Do other work while waiting for the response
            
            // Get the result (this will block if not ready)
            GenerationResponse response = future.get();
            
            System.out.println("Generated text: " + response.getGeneratedText());
            
        } catch (BharatAIException e) {
            System.err.println("Error: " + e.getMessage());
        } catch (InterruptedException | ExecutionException e) {
            System.err.println("Async error: " + e.getMessage());
        } finally {
            client.close();
        }
    }
}
```

### Batch Processing

```java
import com.bharatai.sdk.BharatAIClient;
import com.bharatai.sdk.models.GenerationRequest;
import com.bharatai.sdk.models.BatchGenerationRequest;
import com.bharatai.sdk.models.BatchGenerationResponse;
import com.bharatai.sdk.models.GenerationResponse;
import com.bharatai.sdk.exceptions.BharatAIException;

import java.util.Arrays;
import java.util.List;

public class BatchExample {
    public static void main(String[] args) {
        BharatAIClient client = new BharatAIClient();
        
        try {
            // Create multiple generation requests
            List<GenerationRequest> requests = Arrays.asList(
                new GenerationRequest.Builder().prompt("What is AI?").build(),
                new GenerationRequest.Builder().prompt("What is machine learning?").build(),
                new GenerationRequest.Builder().prompt("What is deep learning?").build()
            );
            
            // Generate text for all prompts
            BatchGenerationRequest batchRequest = new BatchGenerationRequest(requests);
            BatchGenerationResponse batchResponse = client.generateTextBatch(batchRequest);
            
            // Process results
            for (GenerationResponse response : batchResponse.getResponses()) {
                System.out.println("Prompt: " + response.getPrompt());
                System.out.println("Response: " + response.getGeneratedText());
                System.out.println("---");
            }
            
        } catch (BharatAIException e) {
            System.err.println("Error: " + e.getMessage());
        } finally {
            client.close();
        }
    }
}
```

## 📚 API Reference

### BharatAIClient

The main client class for interacting with BFMF APIs.

#### Constructor

```java
// Default configuration
BharatAIClient client = new BharatAIClient();

// Custom configuration
ClientConfig config = new ClientConfig.Builder()
    .baseURL("http://localhost:8000")
    .apiKey("your-api-key")
    .timeout(30000)
    .maxRetries(3)
    .retryDelay(1000)
    .debug(false)
    .build();
    
BharatAIClient client = new BharatAIClient(config);
```

#### Configuration Options

- `baseURL`: Base URL for the API (default: "http://localhost:8000")
- `apiKey`: API key for authentication (optional)
- `timeout`: Request timeout in milliseconds (default: 30000)
- `maxRetries`: Maximum number of retries (default: 3)
- `retryDelay`: Delay between retries in milliseconds (default: 1000)
- `debug`: Enable debug logging (default: false)

#### Methods

##### `generateText(GenerationRequest request): GenerationResponse`

Generate text from a prompt.

##### `generateTextAsync(GenerationRequest request): CompletableFuture<GenerationResponse>`

Generate text from a prompt asynchronously.

##### `generateTextBatch(List<GenerationRequest> requests): BatchGenerationResponse`

Generate text for multiple prompts.

##### `generateTextBatchAsync(List<GenerationRequest> requests): CompletableFuture<BatchGenerationResponse>`

Generate text for multiple prompts asynchronously.

##### `getEmbeddings(EmbeddingRequest request): EmbeddingResponse`

Get text embeddings.

##### `getEmbeddingsAsync(EmbeddingRequest request): CompletableFuture<EmbeddingResponse>`

Get text embeddings asynchronously.

##### `getModelInfo(): ModelInfo`

Get model information.

##### `getModelInfoAsync(): CompletableFuture<ModelInfo>`

Get model information asynchronously.

##### `getHealth(): HealthResponse`

Check API health.

##### `getHealthAsync(): CompletableFuture<HealthResponse>`

Check API health asynchronously.

##### `getSupportedLanguages(): SupportedLanguagesResponse`

Get supported languages.

##### `getSupportedLanguagesAsync(): CompletableFuture<SupportedLanguagesResponse>`

Get supported languages asynchronously.

### Domain-Specific Clients

#### LanguageAIClient

Specialized client for Language AI capabilities.

**Key Methods:**
- `translate(String text, String sourceLanguage, String targetLanguage): String`
- `detectLanguage(String text): String`
- `getSupportedLanguages(): String[]`

#### GovernanceAIClient

Specialized client for Governance AI capabilities.

**Key Methods:**
- `generateRTIResponse(String applicationText, String department): String`
- `analyzePolicy(String policyText): String`

#### EducationAIClient

Specialized client for Education AI capabilities.

**Key Methods:**
- `startTutoringSession(String subject, String topic, String studentLevel): String`
- `generateContent(String subject, String topic, String contentType): String`

#### FinanceAIClient

Specialized client for Finance AI capabilities.

**Key Methods:**
- `analyzeFinancials(String financialData): String`
- `auditTransactions(String transactions): String`

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
cd bharat-fm/sdks/java

# Build the project
mvn clean compile

# Run tests
mvn test

# Package the JAR
mvn package

# Install to local Maven repository
mvn install
```

### Testing

```bash
# Run all tests
mvn test

# Run tests with coverage
mvn clean test jacoco:report

# Run specific test class
mvn test -Dtest=BasicExampleTest
```

## 📝 Examples

### Spring Boot Application

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;
import com.bharatai.sdk.BharatAIClient;
import com.bharatai.sdk.models.GenerationRequest;
import com.bharatai.sdk.models.GenerationResponse;
import com.bharatai.sdk.exceptions.BharatAIException;

@SpringBootApplication
@RestController
@RequestMapping("/api")
public class BFMFController {
    
    private final BharatAIClient client;
    
    public BFMFController() {
        this.client = new BharatAIClient();
    }
    
    @PostMapping("/generate")
    public GenerationResponse generateText(@RequestBody GenerationRequest request) {
        try {
            return client.generateText(request);
        } catch (BharatAIException e) {
            throw new RuntimeException("Failed to generate text", e);
        }
    }
    
    @GetMapping("/health")
    public String health() {
        try {
            return "BFMF Status: " + client.getHealth().getStatus();
        } catch (BharatAIException e) {
            return "BFMF Status: Error - " + e.getMessage();
        }
    }
    
    public static void main(String[] args) {
        SpringApplication.run(BFMFController.class, args);
    }
}
```

### Android Application

```java
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.bharatai.sdk.BharatAIClient;
import com.bharatai.sdk.models.GenerationRequest;
import com.bharatai.sdk.models.GenerationResponse;
import com.bharatai.sdk.exceptions.BharatAIException;

public class MainActivity extends AppCompatActivity {
    
    private BharatAIClient client;
    private EditText inputText;
    private Button generateButton;
    private TextView outputText;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize client
        client = new BharatAIClient();
        
        // Initialize UI components
        inputText = findViewById(R.id.inputText);
        generateButton = findViewById(R.id.generateButton);
        outputText = findViewById(R.id.outputText);
        
        generateButton.setOnClickListener(v -> {
            String prompt = inputText.getText().toString();
            if (!prompt.isEmpty()) {
                generateText(prompt);
            }
        });
    }
    
    private void generateText(String prompt) {
        new Thread(() -> {
            try {
                GenerationRequest request = new GenerationRequest.Builder()
                    .prompt(prompt)
                    .maxTokens(100)
                    .build();
                    
                GenerationResponse response = client.generateText(request);
                
                runOnUiThread(() -> {
                    outputText.setText(response.getGeneratedText());
                });
                
            } catch (BharatAIException e) {
                runOnUiThread(() -> {
                    Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (client != null) {
            client.close();
        }
    }
}
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