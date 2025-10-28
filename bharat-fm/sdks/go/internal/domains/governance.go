package domains

import (
    "context"
    "fmt"

    "github.com/bharat-ai/bharat-fm/sdks/go/internal/client"
    "github.com/bharat-ai/bharat-fm/sdks/go/internal/models"
)

// GovernanceAIClient provides governance-specific AI capabilities
type GovernanceAIClient struct {
    client *client.Client
}

// NewGovernanceAIClient creates a new governance AI client
func NewGovernanceAIClient(client *client.Client) *GovernanceAIClient {
    return &GovernanceAIClient{
        client: client,
    }
}

// RTIRequest represents an RTI request
type RTIRequest struct {
    ApplicationText string `json:"application_text"`
    Department      string `json:"department"`
    RequestType     string `json:"request_type"`
    Urgency         string `json:"urgency,omitempty"`
    Language        string `json:"language,omitempty"`
}

// RTIResponse represents an RTI response
type RTIResponse struct {
    ResponseText        string `json:"response_text"`
    OriginalApplication string `json:"original_application"`
    Department          string `json:"department"`
    ResponseTimeEstimate string `json:"response_time_estimate"`
    RelevantSections    []struct {
        Section     string `json:"section"`
        Description string `json:"description"`
        Relevance   string `json:"relevance"`
    } `json:"relevant_sections"`
    Confidence float64 `json:"confidence"`
    ProcessingTime float64 `json:"processing_time"`
}

// PolicyAnalysisRequest represents a policy analysis request
type PolicyAnalysisRequest struct {
    PolicyText     string `json:"policy_text"`
    AnalysisType   string `json:"analysis_type"`
    Sector         string `json:"sector,omitempty"`
    TargetAudience string `json:"target_audience,omitempty"`
    Language       string `json:"language,omitempty"`
}

// PolicyAnalysisResponse represents a policy analysis response
type PolicyAnalysisResponse struct {
    Analysis         string   `json:"analysis"`
    KeyInsights     []string `json:"key_insights"`
    Recommendations  []string `json:"recommendations"`
    RiskAssessment  *struct {
        Level   string   `json:"level"`
        Factors []string `json:"factors"`
    } `json:"risk_assessment,omitempty"`
    ComplianceScore float64 `json:"compliance_score,omitempty"`
    AnalysisTime    float64 `json:"analysis_time"`
}

// GenerateRTIResponse generates an RTI response
func (c *GovernanceAIClient) GenerateRTIResponse(ctx context.Context, req *RTIRequest) (*RTIResponse, error) {
    var result RTIResponse
    
    apiReq := map[string]interface{}{
        "application_text": req.ApplicationText,
        "department":       req.Department,
        "request_type":     req.RequestType,
    }
    
    if req.Urgency != "" {
        apiReq["urgency"] = req.Urgency
    }
    if req.Language != "" {
        apiReq["language"] = req.Language
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/governance/rti-response")
    
    if err != nil {
        return nil, fmt.Errorf("RTI response request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("RTI response API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// AnalyzePolicy analyzes a policy document
func (c *GovernanceAIClient) AnalyzePolicy(ctx context.Context, req *PolicyAnalysisRequest) (*PolicyAnalysisResponse, error) {
    var result PolicyAnalysisResponse
    
    apiReq := map[string]interface{}{
        "policy_text":   req.PolicyText,
        "analysis_type": req.AnalysisType,
    }
    
    if req.Sector != "" {
        apiReq["sector"] = req.Sector
    }
    if req.TargetAudience != "" {
        apiReq["target_audience"] = req.TargetAudience
    }
    if req.Language != "" {
        apiReq["language"] = req.Language
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/governance/analyze-policy")
    
    if err != nil {
        return nil, fmt.Errorf("policy analysis request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("policy analysis API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// GetSupportedDepartments gets supported departments
func (c *GovernanceAIClient) GetSupportedDepartments(ctx context.Context) ([]struct {
    Name        string   `json:"name"`
    Code        string   `json:"code"`
    Description string   `json:"description"`
    Categories  []string `json:"categories"`
}, error) {
    var result []struct {
        Name        string   `json:"name"`
        Code        string   `json:"code"`
        Description string   `json:"description"`
        Categories  []string `json:"categories"`
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetResult(&result).
        Get("/governance/departments")
    
    if err != nil {
        return nil, fmt.Errorf("get departments request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("get departments API error: status %d", resp.StatusCode())
    }
    
    return result, nil
}

// SimpleGenerateRTIResponse provides a simple RTI response interface
func (c *GovernanceAIClient) SimpleGenerateRTIResponse(ctx context.Context, applicationText, department string) (string, error) {
    req := &RTIRequest{
        ApplicationText: applicationText,
        Department:      department,
        RequestType:     "information",
    }
    
    resp, err := c.GenerateRTIResponse(ctx, req)
    if err != nil {
        return "", err
    }
    
    return resp.ResponseText, nil
}

// SimpleAnalyzePolicy provides a simple policy analysis interface
func (c *GovernanceAIClient) SimpleAnalyzePolicy(ctx context.Context, policyText, analysisType string) (string, error) {
    req := &PolicyAnalysisRequest{
        PolicyText:   policyText,
        AnalysisType: analysisType,
    }
    
    resp, err := c.AnalyzePolicy(ctx, req)
    if err != nil {
        return "", err
    }
    
    return resp.Analysis, nil
}