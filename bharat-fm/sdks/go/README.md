# Bharat Foundation Model Framework - Go SDK

The official Go SDK for the Bharat Foundation Model Framework (BFMF), enabling seamless integration of India's sovereign AI capabilities into Go applications.

## 🌟 Features

- **Multi-Language Support**: Native support for 22+ Indian languages
- **Domain-Specific Clients**: Specialized clients for Language, Governance, Education, and Finance AI
- **Type Safety**: Full type safety with Go's strong typing system
- **Easy Integration**: Simple REST API client with automatic retries and error handling
- **Streaming Support**: Real-time text generation with streaming capabilities
- **Production Ready**: Built for production with robust error handling and logging
- **Context Support**: Full context.Context support for cancellation and timeouts
- **Concurrent Operations**: Goroutine-safe with built-in concurrency support

## 📦 Installation

### Go Modules

Add the BFMF Go SDK to your project:

```bash
go get github.com/bharat-ai/bharat-fm/sdks/go
```

### Import

```go
import "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
```

## 🚀 Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

func main() {
    // Create a new BFMF client
    config := bharat.NewConfig()
    config.BaseURL = "http://localhost:8000" // Your BFMF server URL
    config.Debug = false // Enable debug logging
    
    client := bharat.NewClient(config)
    defer client.Close()

    // Create a generation request
    req := bharat.NewGenerationRequest("नमस्ते, आप कैसे हैं?")
    req.MaxTokens = 100
    req.Language = "hi"
    req.Temperature = 0.7

    // Generate text
    ctx := context.Background()
    resp, err := client.GenerateText(ctx, req)
    if err != nil {
        log.Fatalf("Failed to generate text: %v", err)
    }

    fmt.Printf("Generated text: %s\n", resp.GeneratedText)
    fmt.Printf("Tokens generated: %d\n", resp.TokensGenerated)
    fmt.Printf("Generation time: %.2f seconds\n", resp.GenerationTime)
}
```

### Domain-Specific Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

func main() {
    client := bharat.NewClient(bharat.NewConfig())
    defer client.Close()

    ctx := context.Background()

    // Language AI - Translate text
    langClient := bharat.NewLanguageAIClient(client)
    translated, err := langClient.SimpleTranslate(ctx, "Hello, how are you?", bharat.LanguageEnglish, bharat.LanguageHindi)
    if err != nil {
        log.Printf("Translation failed: %v", err)
    } else {
        fmt.Printf("Translation: %s\n", translated)
    }

    // Governance AI - Generate RTI response
    govClient := bharat.NewGovernanceAIClient(client)
    rtiResponse, err := govClient.SimpleGenerateRTIResponse(ctx, "I would like to know about...", "Ministry of Health")
    if err != nil {
        log.Printf("RTI generation failed: %v", err)
    } else {
        fmt.Printf("RTI Response: %s\n", rtiResponse)
    }

    // Education AI - Start tutoring session
    eduClient := bharat.NewEducationAIClient(client)
    tutoringResponse, err := eduClient.SimpleStartTutoringSession(ctx, "Mathematics", "Algebra", "secondary")
    if err != nil {
        log.Printf("Tutoring session failed: %v", err)
    } else {
        fmt.Printf("Tutoring: %s\n", tutoringResponse)
    }

    // Finance AI - Analyze financial data
    financeClient := bharat.NewFinanceAIClient(client)
    financialData := map[string]interface{}{
        "revenue":  1000000.0,
        "expenses": 600000.0,
    }
    analysis, err := financeClient.SimpleAnalyzeFinancials(ctx, financialData, "ratio")
    if err != nil {
        log.Printf("Financial analysis failed: %v", err)
    } else {
        fmt.Printf("Financial Analysis: %s\n", analysis)
    }
}
```

### Streaming Generation

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

func main() {
    client := bharat.NewClient(bharat.NewConfig())
    defer client.Close()

    req := bharat.NewGenerationRequest("Tell me about Indian culture")
    req.MaxTokens = 200
    req.Temperature = 0.8

    // Stream text generation
    ctx := context.Background()
    stream, err := client.GenerateTextStream(ctx, req)
    if err != nil {
        log.Fatalf("Failed to create stream: %v", err)
    }

    fmt.Println("Streaming response:")
    for chunk := range stream {
        fmt.Printf("%s", chunk)
    }
    fmt.Println() // New line after streaming
}
```

### Batch Processing

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

func main() {
    client := bharat.NewClient(bharat.NewConfig())
    defer client.Close()

    // Create multiple generation requests
    requests := []*bharat.GenerationRequest{
        bharat.NewGenerationRequest("What is AI?"),
        bharat.NewGenerationRequest("What is machine learning?"),
        bharat.NewGenerationRequest("What is deep learning?"),
    }

    // Set parameters for all requests
    for _, req := range requests {
        req.MaxTokens = 50
        req.Temperature = 0.8
    }

    // Generate text for all prompts
    batchReq := bharat.NewBatchGenerationRequest(requests)
    ctx := context.Background()
    resp, err := client.GenerateTextBatch(ctx, batchReq)
    if err != nil {
        log.Fatalf("Failed to generate batch text: %v", err)
    }

    // Process results
    for i, response := range resp.Responses {
        fmt.Printf("Response %d: %s\n", i+1, response.GeneratedText)
    }
}
```

### Concurrent Operations

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

func main() {
    client := bharat.NewClient(bharat.NewConfig())
    defer client.Close()

    ctx := context.Background()
    var wg sync.WaitGroup

    prompts := []string{
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is deep learning?",
        "What is neural network?",
        "What is natural language processing?",
    }

    results := make([]string, len(prompts))
    errors := make([]error, len(prompts))

    // Concurrent text generation
    for i, prompt := range prompts {
        wg.Add(1)
        go func(idx int, p string) {
            defer wg.Done()

            req := bharat.NewGenerationRequest(p)
            req.MaxTokens = 50

            resp, err := client.GenerateText(ctx, req)
            if err != nil {
                errors[idx] = err
                return
            }

            results[idx] = resp.GeneratedText
        }(i, prompt)
    }

    wg.Wait()

    // Process results
    for i, result := range results {
        if errors[i] != nil {
            log.Printf("Error for prompt %d: %v", i+1, errors[i])
            continue
        }
        fmt.Printf("Result %d: %s\n", i+1, result)
    }
}
```

## 📚 API Reference

### Client Configuration

```go
config := bharat.NewConfig()
config.BaseURL = "http://localhost:8000"          // Base URL for the API
config.APIKey = "your-api-key"                // API key for authentication
config.Timeout = 30 * time.Second            // Request timeout
config.MaxRetries = 3                       // Maximum number of retries
config.RetryDelay = 1 * time.Second         // Delay between retries
config.Debug = true                          // Enable debug logging

client := bharat.NewClient(config)
```

### Main Client Methods

#### `GenerateText(ctx context.Context, req *GenerationRequest) (*GenerationResponse, error)`

Generate text from a prompt.

#### `GenerateTextBatch(ctx context.Context, req *BatchGenerationRequest) (*BatchGenerationResponse, error)`

Generate text for multiple prompts.

#### `GenerateTextStream(ctx context.Context, req *GenerationRequest) (<-chan string, error)`

Stream text generation in real-time.

#### `GetEmbeddings(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)`

Get text embeddings.

#### `GetModelInfo(ctx context.Context) (*ModelInfo, error)`

Get model information.

#### `GetHealth(ctx context.Context) (*HealthResponse, error)`

Check API health.

#### `GetSupportedLanguages(ctx context.Context) (*SupportedLanguagesResponse, error)`

Get supported languages.

### Domain-Specific Clients

#### LanguageAIClient

```go
langClient := bharat.NewLanguageAIClient(client)

// Simple translation
translated, err := langClient.SimpleTranslate(ctx, "Hello", bharat.LanguageEnglish, bharat.LanguageHindi)

// Language detection
detected, err := langClient.SimpleDetectLanguage(ctx, "नमस्ते दुनिया!")

// Advanced translation
req := &bharat.TranslationRequest{
    Text:              "Hello world",
    SourceLanguage:    bharat.LanguageEnglish,
    TargetLanguage:    bharat.LanguageHindi,
    PreserveFormatting: true,
}
resp, err := langClient.Translate(ctx, req)
```

#### GovernanceAIClient

```go
govClient := bharat.NewGovernanceAIClient(client)

// Simple RTI response
rtiResponse, err := govClient.SimpleGenerateRTIResponse(ctx, applicationText, department)

// Policy analysis
policyAnalysis, err := govClient.SimpleAnalyzePolicy(ctx, policyText, "summary")

// Advanced RTI response
req := &bharat.RTIRequest{
    ApplicationText: "I would like to know about...",
    Department:      "Ministry of Health",
    RequestType:     "information",
    Urgency:         "normal",
}
resp, err := govClient.GenerateRTIResponse(ctx, req)
```

#### EducationAIClient

```go
eduClient := bharat.NewEducationAIClient(client)

// Simple tutoring session
tutoringResponse, err := eduClient.SimpleStartTutoringSession(ctx, "Mathematics", "Algebra", "secondary")

// Content generation
content, err := eduClient.SimpleGenerateContent(ctx, "Science", "Photosynthesis", "lesson-plan")

// Advanced tutoring session
req := &bharat.TutoringRequest{
    Subject:          "Mathematics",
    Topic:            "Algebra",
    StudentLevel:     "secondary",
    LearningStyle:    "visual",
    Language:         "en",
    LearningObjectives: []string{"Understand variables", "Solve equations"},
}
resp, err := eduClient.StartTutoringSession(ctx, req)
```

#### FinanceAIClient

```go
financeClient := bharat.NewFinanceAIClient(client)

// Simple financial analysis
financialData := map[string]interface{}{
    "revenue":  1000000.0,
    "expenses": 600000.0,
}
analysis, err := financeClient.SimpleAnalyzeFinancials(ctx, financialData, "ratio")

// Transaction audit
transactions := []map[string]interface{}{
    {
        "id":          "txn001",
        "date":        "2024-01-15",
        "amount":      15000.0,
        "description": "Office supplies",
        "category":    "expenses",
    },
}
auditResult, err := financeClient.SimpleAuditTransactions(ctx, transactions, "fraud")

// Advanced financial analysis
req := &bharat.FinancialAnalysisRequest{
    FinancialData: financialData,
    AnalysisType:  "ratio",
    Industry:      "technology",
    CompanySize:   "medium",
}
resp, err := financeClient.AnalyzeFinancials(ctx, req)
```

## 🌐 Supported Languages

The SDK supports all 22+ Indian languages:

```go
// Available languages
bharat.LanguageHindi      // "hi"
bharat.LanguageEnglish    // "en"
bharat.LanguageBengali    // "bn"
bharat.LanguageTamil      // "ta"
bharat.LanguageTelugu     // "te"
bharat.LanguageMarathi    // "mr"
bharat.LanguageGujarati   // "gu"
bharat.LanguageKannada    // "kn"
bharat.LanguageMalayalam  // "ml"
bharat.LanguagePunjabi    // "pa"
// ... and more

// Get all supported languages
languages := bharat.GetAllLanguages()
```

## 🛠️ Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm/sdks/go

# Build the SDK
go build ./...

# Run tests
go test ./...

# Run examples
go run examples/basic/main.go
```

### Testing

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests with coverage
go test -cover ./...

# Run specific test
go test -run TestClient ./internal/client/
```

## 📝 Examples

### Web Server Example

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

type GenerateRequest struct {
    Prompt      string  `json:"prompt"`
    MaxTokens   int     `json:"max_tokens"`
    Temperature float64 `json:"temperature"`
    Language    string  `json:"language"`
}

type GenerateResponse struct {
    GeneratedText string `json:"generated_text"`
    Error        string `json:"error,omitempty"`
}

func main() {
    // Initialize BFMF client
    client := bharat.NewClient(bharat.NewConfig())
    defer client.Close()

    // HTTP handler for text generation
    http.HandleFunc("/generate", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
            return
        }

        var req GenerateRequest
        if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
            http.Error(w, "Invalid request", http.StatusBadRequest)
            return
        }

        // Create BFMF request
        bfmfReq := bharat.NewGenerationRequest(req.Prompt)
        bfmfReq.MaxTokens = req.MaxTokens
        bfmfReq.Temperature = req.Temperature
        bfmfReq.Language = req.Language

        // Generate text
        resp, err := client.GenerateText(r.Context(), bfmfReq)
        if err != nil {
            json.NewEncoder(w).Encode(GenerateResponse{
                Error: err.Error(),
            })
            return
        }

        json.NewEncoder(w).Encode(GenerateResponse{
            GeneratedText: resp.GeneratedText,
        })
    })

    // Health check endpoint
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        health, err := client.GetHealth(r.Context())
        if err != nil {
            fmt.Fprintf(w, "Status: Error - %s", err.Error())
            return
        }
        fmt.Fprintf(w, "Status: %s", health.Status)
    })

    fmt.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### CLI Tool Example

```go
package main

import (
    "bufio"
    "context"
    "flag"
    "fmt"
    "log"
    "os"
    "strings"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

func main() {
    // Parse command line flags
    baseURL := flag.String("url", "http://localhost:8000", "BFMF server URL")
    apiKey := flag.String("key", "", "API key")
    debug := flag.Bool("debug", false, "Enable debug logging")
    domain := flag.String("domain", "general", "Domain (general, language, governance, education, finance)")
    flag.Parse()

    // Create client
    config := bharat.NewConfig()
    config.BaseURL = *baseURL
    config.APIKey = *apiKey
    config.Debug = *debug
    
    client := bharat.NewClient(config)
    defer client.Close()

    ctx := context.Background()

    // Interactive mode
    scanner := bufio.NewScanner(os.Stdin)
    
    for {
        fmt.Print("\n> ")
        if !scanner.Scan() {
            break
        }

        input := strings.TrimSpace(scanner.Text())
        if input == "exit" || input == "quit" {
            break
        }

        if input == "" {
            continue
        }

        switch *domain {
        case "language":
            handleLanguageDomain(ctx, client, input)
        case "governance":
            handleGovernanceDomain(ctx, client, input)
        case "education":
            handleEducationDomain(ctx, client, input)
        case "finance":
            handleFinanceDomain(ctx, client, input)
        default:
            handleGeneralDomain(ctx, client, input)
        }
    }

    if err := scanner.Err(); err != nil {
        log.Printf("Scanner error: %v", err)
    }
}

func handleGeneralDomain(ctx context.Context, client *bharat.Client, input string) {
    req := bharat.NewGenerationRequest(input)
    req.MaxTokens = 100
    req.Temperature = 0.7

    resp, err := client.GenerateText(ctx, req)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    fmt.Printf("Response: %s\n", resp.GeneratedText)
}

func handleLanguageDomain(ctx context.Context, client *bharat.Client, input string) {
    langClient := bharat.NewLanguageAIClient(client)
    
    // Try to detect if it's a translation request
    if strings.Contains(input, "translate") || strings.Contains(input, "अनुवाद") {
        // Simple translation (assume English to Hindi)
        translated, err := langClient.SimpleTranslate(ctx, input, bharat.LanguageEnglish, bharat.LanguageHindi)
        if err != nil {
            fmt.Printf("Translation error: %v\n", err)
            return
        }
        fmt.Printf("Translation: %s\n", translated)
    } else {
        // Language detection
        detected, err := langClient.SimpleDetectLanguage(ctx, input)
        if err != nil {
            fmt.Printf("Detection error: %v\n", err)
            return
        }
        fmt.Printf("Detected language: %s\n", detected)
    }
}

func handleGovernanceDomain(ctx context.Context, client *bharat.Client, input string) {
    govClient := bharat.NewGovernanceAIClient(client)
    
    // Simple RTI response generation
    response, err := govClient.SimpleGenerateRTIResponse(ctx, input, "General")
    if err != nil {
        fmt.Printf("RTI error: %v\n", err)
        return
    }
    fmt.Printf("RTI Response: %s\n", response)
}

func handleEducationDomain(ctx context.Context, client *bharat.Client, input string) {
    eduClient := bharat.NewEducationAIClient(client)
    
    // Simple tutoring response
    response, err := eduClient.SimpleStartTutoringSession(ctx, "General", input, "secondary")
    if err != nil {
        fmt.Printf("Education error: %v\n", err)
        return
    }
    fmt.Printf("Tutoring Response: %s\n", response)
}

func handleFinanceDomain(ctx context.Context, client *bharat.Client, input string) {
    financeClient := bharat.NewFinanceAIClient(client)
    
    // Simple financial query response
    response, err := financeClient.SimpleAnalyzeFinancials(ctx, map[string]interface{}{
        "query": input,
    }, "general")
    if err != nil {
        fmt.Printf("Finance error: %v\n", err)
        return
    }
    fmt.Printf("Finance Analysis: %s\n", response)
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
- Go community for excellent tooling and libraries
- Contributors who help improve this SDK

---

**Made with ❤️ for Bharat's AI Independence**