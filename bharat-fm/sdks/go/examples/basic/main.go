package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/bharat-ai/bharat-fm/sdks/go/bharat"
)

func main() {
    // Create a new BFMF client
    config := bharat.NewConfig()
    config.BaseURL = "http://localhost:8000" // Your BFMF server URL
    config.Debug = true // Enable debug logging
    
    client := bharat.NewClient(config)
    defer client.Close()

    // Set up signal handling for graceful shutdown
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

    // Example 1: Basic text generation
    fmt.Println("=== Example 1: Basic Text Generation ===")
    err := basicTextGeneration(ctx, client)
    if err != nil {
        log.Printf("Basic text generation failed: %v", err)
    }

    // Example 2: Batch text generation
    fmt.Println("\n=== Example 2: Batch Text Generation ===")
    err = batchTextGeneration(ctx, client)
    if err != nil {
        log.Printf("Batch text generation failed: %v", err)
    }

    // Example 3: Language translation
    fmt.Println("\n=== Example 3: Language Translation ===")
    err = languageTranslation(ctx, client)
    if err != nil {
        log.Printf("Language translation failed: %v", err)
    }

    // Example 4: RTI response generation
    fmt.Println("\n=== Example 4: RTI Response Generation ===")
    err = rtiResponseGeneration(ctx, client)
    if err != nil {
        log.Printf("RTI response generation failed: %v", err)
    }

    // Example 5: Educational content generation
    fmt.Println("\n=== Example 5: Educational Content Generation ===")
    err = educationalContentGeneration(ctx, client)
    if err != nil {
        log.Printf("Educational content generation failed: %v", err)
    }

    // Example 6: Financial analysis
    fmt.Println("\n=== Example 6: Financial Analysis ===")
    err = financialAnalysis(ctx, client)
    if err != nil {
        log.Printf("Financial analysis failed: %v", err)
    }

    // Example 7: Streaming text generation
    fmt.Println("\n=== Example 7: Streaming Text Generation ===")
    err = streamingTextGeneration(ctx, client)
    if err != nil {
        log.Printf("Streaming text generation failed: %v", err)
    }

    // Wait for interrupt signal
    <-sigChan
    fmt.Println("\nShutting down gracefully...")
}

func basicTextGeneration(ctx context.Context, client *bharat.Client) error {
    req := bharat.NewGenerationRequest("नमस्ते, आप कैसे हैं?")
    req.MaxTokens = 100
    req.Language = "hi"
    req.Temperature = 0.7

    resp, err := client.GenerateText(ctx, req)
    if err != nil {
        return fmt.Errorf("failed to generate text: %w", err)
    }

    fmt.Printf("Generated text: %s\n", resp.GeneratedText)
    fmt.Printf("Tokens generated: %d\n", resp.TokensGenerated)
    fmt.Printf("Generation time: %.2f seconds\n", resp.GenerationTime)
    if resp.LanguageDetected != "" {
        fmt.Printf("Detected language: %s\n", resp.LanguageDetected)
    }

    return nil
}

func batchTextGeneration(ctx context.Context, client *bharat.Client) error {
    requests := []*bharat.GenerationRequest{
        bharat.NewGenerationRequest("What is artificial intelligence?"),
        bharat.NewGenerationRequest("What is machine learning?"),
        bharat.NewGenerationRequest("What is deep learning?"),
    }

    // Set some parameters for all requests
    for _, req := range requests {
        req.MaxTokens = 50
        req.Temperature = 0.8
    }

    batchReq := bharat.NewBatchGenerationRequest(requests)
    resp, err := client.GenerateTextBatch(ctx, batchReq)
    if err != nil {
        return fmt.Errorf("failed to generate batch text: %w", err)
    }

    for i, response := range resp.Responses {
        fmt.Printf("Response %d: %s\n", i+1, response.GeneratedText)
    }

    return nil
}

func languageTranslation(ctx context.Context, client *bharat.Client) error {
    langClient := bharat.NewLanguageAIClient(client)

    // Simple translation
    translated, err := langClient.SimpleTranslate(ctx, "Hello, how are you today?", bharat.LanguageEnglish, bharat.LanguageHindi)
    if err != nil {
        return fmt.Errorf("failed to translate: %w", err)
    }

    fmt.Printf("Translation: %s\n", translated)

    // Language detection
    detected, err := langClient.SimpleDetectLanguage(ctx, "नमस्ते दुनिया!")
    if err != nil {
        return fmt.Errorf("failed to detect language: %w", err)
    }

    fmt.Printf("Detected language: %s\n", detected)

    return nil
}

func rtiResponseGeneration(ctx context.Context, client *bharat.Client) error {
    govClient := bharat.NewGovernanceAIClient(client)

    applicationText := "I would like to know about the status of the new highway construction project between Delhi and Mumbai."
    department := "Ministry of Road Transport and Highways"

    response, err := govClient.SimpleGenerateRTIResponse(ctx, applicationText, department)
    if err != nil {
        return fmt.Errorf("failed to generate RTI response: %w", err)
    }

    fmt.Printf("RTI Response: %s\n", response)

    return nil
}

func educationalContentGeneration(ctx context.Context, client *bharat.Client) error {
    eduClient := bharat.NewEducationAIClient(client)

    // Start tutoring session
    tutoringResponse, err := eduClient.SimpleStartTutoringSession(ctx, "Mathematics", "Algebra", "secondary")
    if err != nil {
        return fmt.Errorf("failed to start tutoring session: %w", err)
    }

    fmt.Printf("Tutoring Response: %s\n", tutoringResponse)

    // Generate educational content
    content, err := eduClient.SimpleGenerateContent(ctx, "Science", "Photosynthesis", "lesson-plan")
    if err != nil {
        return fmt.Errorf("failed to generate content: %w", err)
    }

    fmt.Printf("Generated Content: %s\n", content)

    return nil
}

func financialAnalysis(ctx context.Context, client *bharat.Client) error {
    financeClient := bharat.NewFinanceAIClient(client)

    financialData := map[string]interface{}{
        "balance_sheet": map[string]interface{}{
            "assets":  1000000.0,
            "liabilities": 600000.0,
            "equity": 400000.0,
        },
        "income_statement": map[string]interface{}{
            "revenue":    500000.0,
            "expenses":  300000.0,
            "net_income": 200000.0,
        },
    }

    analysis, err := financeClient.SimpleAnalyzeFinancials(ctx, financialData, "ratio")
    if err != nil {
        return fmt.Errorf("failed to analyze financials: %w", err)
    }

    fmt.Printf("Financial Analysis: %s\n", analysis)

    // Simple transaction audit
    transactions := []map[string]interface{}{
        {
            "id":          "txn001",
            "date":        "2024-01-15",
            "amount":      15000.0,
            "description": "Office supplies purchase",
            "category":    "expenses",
            "account":     "cash",
        },
        {
            "id":          "txn002",
            "date":        "2024-01-16",
            "amount":      25000.0,
            "description": "Client payment received",
            "category":    "income",
            "account":     "bank",
        },
    }

    auditResult, err := financeClient.SimpleAuditTransactions(ctx, transactions, "fraud")
    if err != nil {
        return fmt.Errorf("failed to audit transactions: %w", err)
    }

    fmt.Printf("Transaction Audit: %s\n", auditResult)

    return nil
}

func streamingTextGeneration(ctx context.Context, client *bharat.Client) error {
    req := bharat.NewGenerationRequest("Tell me about the rich cultural heritage of India")
    req.MaxTokens = 200
    req.Temperature = 0.8

    stream, err := client.GenerateTextStream(ctx, req)
    if err != nil {
        return fmt.Errorf("failed to create text stream: %w", err)
    }

    fmt.Println("Streaming response:")
    for chunk := range stream {
        fmt.Printf("%s", chunk)
    }
    fmt.Println() // New line after streaming

    return nil
}