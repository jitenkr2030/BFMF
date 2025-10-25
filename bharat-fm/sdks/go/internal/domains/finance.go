package domains

import (
    "context"
    "fmt"

    "github.com/bharat-ai/bharat-fm/sdks/go/internal/client"
    "github.com/bharat-ai/bharat-fm/sdks/go/internal/models"
)

// FinanceAIClient provides finance-specific AI capabilities
type FinanceAIClient struct {
    client *client.Client
}

// NewFinanceAIClient creates a new finance AI client
func NewFinanceAIClient(client *client.Client) *FinanceAIClient {
    return &FinanceAIClient{
        client: client,
    }
}

// FinancialAnalysisRequest represents a financial analysis request
type FinancialAnalysisRequest struct {
    FinancialData map[string]interface{} `json:"financial_data"`
    AnalysisType  string                 `json:"analysis_type"`
    TimePeriod    *struct {
        Start string `json:"start"`
        End   string `json:"end"`
    } `json:"time_period,omitempty"`
    Industry      string `json:"industry,omitempty"`
    CompanySize   string `json:"company_size,omitempty"`
    AnalysisDepth string `json:"analysis_depth,omitempty"`
}

// FinancialAnalysisResponse represents a financial analysis response
type FinancialAnalysisResponse struct {
    AnalysisID string `json:"analysis_id"`
    Analysis   struct {
        Summary      string `json:"summary"`
        KeyFindings  []string `json:"key_findings"`
        Ratios       map[string]struct {
            Value      float64 `json:"value"`
            Benchmark  float64 `json:"benchmark,omitempty"`
            Status     string   `json:"status"`
        } `json:"ratios,omitempty"`
        Trends []struct {
            Metric      string  `json:"metric"`
            Direction   string  `json:"direction"`
            Change      float64 `json:"change"`
            Significance string  `json:"significance"`
        } `json:"trends,omitempty"`
        Forecasts []struct {
            Metric     string  `json:"metric"`
            Period     string  `json:"period"`
            Predicted  float64 `json:"predicted"`
            Confidence float64 `json:"confidence"`
        } `json:"forecasts,omitempty"`
    } `json:"analysis"`
    RiskAssessment struct {
        OverallRisk   string `json:"overall_risk"`
        RiskFactors   []struct {
            Factor     string `json:"factor"`
            Level      string `json:"level"`
            Impact     string `json:"impact"`
            Description string `json:"description"`
        } `json:"risk_factors"`
    } `json:"risk_assessment"`
    Recommendations []struct {
        Priority      string `json:"priority"`
        Action        string `json:"action"`
        ExpectedImpact string `json:"expected_impact"`
        Timeline      string `json:"timeline"`
    } `json:"recommendations"`
    AnalysisTime float64 `json:"analysis_time"`
}

// TransactionAuditRequest represents a transaction audit request
type TransactionAuditRequest struct {
    Transactions []struct {
        ID           string  `json:"id"`
        Date         string  `json:"date"`
        Amount       float64 `json:"amount"`
        Description  string  `json:"description"`
        Category     string  `json:"category"`
        Account      string  `json:"account"`
        Counterparty string  `json:"counterparty,omitempty"`
        Reference    string  `json:"reference,omitempty"`
    } `json:"transactions"`
    AuditType    string `json:"audit_type"`
    RiskThreshold float64 `json:"risk_threshold,omitempty"`
    FocusAreas   []string `json:"focus_areas,omitempty"`
    TimePeriod   *struct {
        Start string `json:"start"`
        End   string `json:"end"`
    } `json:"time_period,omitempty"`
    AccountTypes []string `json:"account_types,omitempty"`
}

// TransactionAuditResponse represents a transaction audit response
type TransactionAuditResponse struct {
    AuditID string `json:"audit_id"`
    Summary struct {
        TotalTransactions    int     `json:"total_transactions"`
        FlaggedTransactions int     `json:"flagged_transactions"`
        RiskScore           float64 `json:"risk_score"`
        AuditPeriod         string  `json:"audit_period"`
    } `json:"summary"`
    FlaggedTransactions []struct {
        TransactionID string `json:"transaction_id"`
        RiskLevel      string `json:"risk_level"`
        RiskScore      float64 `json:"risk_score"`
        Flags          []string `json:"flags"`
        Explanation    string `json:"explanation"`
        RecommendedAction string `json:"recommended_action"`
    } `json:"flagged_transactions"`
    Patterns []struct {
        Pattern     string  `json:"pattern"`
        Frequency   int     `json:"frequency"`
        RiskLevel   string  `json:"risk_level"`
        Description string  `json:"description"`
    } `json:"patterns"`
    ComplianceStatus struct {
        Overall string `json:"overall"`
        Issues []struct {
            Regulation string `json:"regulation"`
            Status     string `json:"status"`
            Description string `json:"description"`
            Severity    string `json:"severity"`
        } `json:"issues"`
    } `json:"compliance_status"`
    Recommendations []struct {
        Category      string `json:"category"`
        Recommendation string `json:"recommendation"`
        Priority      string `json:"priority"`
    } `json:"recommendations"`
    AuditTime float64 `json:"audit_time"`
}

// AnalyzeFinancials analyzes financial data
func (c *FinanceAIClient) AnalyzeFinancials(ctx context.Context, req *FinancialAnalysisRequest) (*FinancialAnalysisResponse, error) {
    var result FinancialAnalysisResponse
    
    apiReq := map[string]interface{}{
        "financial_data": req.FinancialData,
        "analysis_type": req.AnalysisType,
    }
    
    if req.TimePeriod != nil {
        apiReq["time_period"] = req.TimePeriod
    }
    if req.Industry != "" {
        apiReq["industry"] = req.Industry
    }
    if req.CompanySize != "" {
        apiReq["company_size"] = req.CompanySize
    }
    if req.AnalysisDepth != "" {
        apiReq["analysis_depth"] = req.AnalysisDepth
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/finance/analyze-financials")
    
    if err != nil {
        return nil, fmt.Errorf("financial analysis request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("financial analysis API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// AuditTransactions audits transactions
func (c *FinanceAIClient) AuditTransactions(ctx context.Context, req *TransactionAuditRequest) (*TransactionAuditResponse, error) {
    var result TransactionAuditResponse
    
    apiReq := map[string]interface{}{
        "transactions": req.Transactions,
        "audit_type":   req.AuditType,
    }
    
    if req.RiskThreshold > 0 {
        apiReq["risk_threshold"] = req.RiskThreshold
    }
    if len(req.FocusAreas) > 0 {
        apiReq["focus_areas"] = req.FocusAreas
    }
    if req.TimePeriod != nil {
        apiReq["time_period"] = req.TimePeriod
    }
    if len(req.AccountTypes) > 0 {
        apiReq["account_types"] = req.AccountTypes
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetBody(apiReq).
        SetResult(&result).
        Post("/finance/audit-transactions")
    
    if err != nil {
        return nil, fmt.Errorf("transaction audit request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("transaction audit API error: status %d", resp.StatusCode())
    }
    
    return &result, nil
}

// GetComplianceRequirements gets compliance requirements
func (c *FinanceAIClient) GetComplianceRequirements(ctx context.Context, industry, jurisdiction string) ([]struct {
    Regulation   string   `json:"regulation"`
    Description  string   `json:"description"`
    Requirements []string `json:"requirements"`
    Penalties    []string `json:"penalties"`
    Frequency    string   `json:"frequency"`
    LastUpdated  string   `json:"last_updated"`
}, error) {
    var result []struct {
        Regulation   string   `json:"regulation"`
        Description  string   `json:"description"`
        Requirements []string `json:"requirements"`
        Penalties    []string `json:"penalties"`
        Frequency    string   `json:"frequency"`
        LastUpdated  string   `json:"last_updated"`
    }

    resp, err := c.client.GetHttpClient().R().
        SetContext(ctx).
        SetResult(&result).
        Get("/finance/compliance-requirements").
        SetQueryParams(map[string]string{
            "industry":     industry,
            "jurisdiction": jurisdiction,
        })
    
    if err != nil {
        return nil, fmt.Errorf("get compliance requirements request failed: %w", err)
    }
    
    if resp.StatusCode() != 200 {
        return nil, fmt.Errorf("get compliance requirements API error: status %d", resp.StatusCode())
    }
    
    return result, nil
}

// SimpleAnalyzeFinancials provides a simple financial analysis interface
func (c *FinanceAIClient) SimpleAnalyzeFinancials(ctx context.Context, financialData map[string]interface{}, analysisType string) (string, error) {
    req := &FinancialAnalysisRequest{
        FinancialData: financialData,
        AnalysisType:  analysisType,
    }
    
    resp, err := c.AnalyzeFinancials(ctx, req)
    if err != nil {
        return "", err
    }
    
    return resp.Analysis.Summary, nil
}

// SimpleAuditTransactions provides a simple transaction audit interface
func (c *FinanceAIClient) SimpleAuditTransactions(ctx context.Context, transactions []map[string]interface{}, auditType string) (string, error) {
    // Convert transactions to the required format
    txnList := make([]struct {
        ID           string  `json:"id"`
        Date         string  `json:"date"`
        Amount       float64 `json:"amount"`
        Description  string  `json:"description"`
        Category     string  `json:"category"`
        Account      string  `json:"account"`
        Counterparty string  `json:"counterparty,omitempty"`
        Reference    string  `json:"reference,omitempty"`
    }, len(transactions))
    
    for i, txn := range transactions {
        txnList[i] = struct {
            ID           string  `json:"id"`
            Date         string  `json:"date"`
            Amount       float64 `json:"amount"`
            Description  string  `json:"description"`
            Category     string  `json:"category"`
            Account      string  `json:"account"`
            Counterparty string  `json:"counterparty,omitempty"`
            Reference    string  `json:"reference,omitempty"`
        }{
            ID:           getString(txn, "id"),
            Date:         getString(txn, "date"),
            Amount:       getFloat64(txn, "amount"),
            Description:  getString(txn, "description"),
            Category:     getString(txn, "category"),
            Account:      getString(txn, "account"),
            Counterparty: getString(txn, "counterparty"),
            Reference:    getString(txn, "reference"),
        }
    }
    
    req := &TransactionAuditRequest{
        Transactions: txnList,
        AuditType:    auditType,
    }
    
    resp, err := c.AuditTransactions(ctx, req)
    if err != nil {
        return "", err
    }
    
    return fmt.Sprintf("Audit completed. Risk score: %.2f, Flagged transactions: %d", 
        resp.Summary.RiskScore, resp.Summary.FlaggedTransactions), nil
}

// Helper functions for simple interface
func getString(m map[string]interface{}, key string) string {
    if val, ok := m[key]; ok {
        if str, ok := val.(string); ok {
            return str
        }
    }
    return ""
}

func getFloat64(m map[string]interface{}, key string) float64 {
    if val, ok := m[key]; ok {
        if num, ok := val.(float64); ok {
            return num
        }
        if num, ok := val.(int); ok {
            return float64(num)
        }
    }
    return 0
}