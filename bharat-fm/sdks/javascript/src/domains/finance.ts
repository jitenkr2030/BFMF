/**
 * Finance AI domain-specific client for BFMF
 */

import { BharatAIClient } from '../client';
import { GenerationRequest, GenerationResponse } from '../types';
import { BharatAIError } from '../errors';

/**
 * Financial analysis request
 */
export interface FinancialAnalysisRequest {
  /** Financial statements data */
  financialData: {
    balanceSheet?: Record<string, number>;
    incomeStatement?: Record<string, number>;
    cashFlowStatement?: Record<string, number>;
  };
  /** Analysis type */
  analysisType: 'ratio' | 'trend' | 'forecast' | 'valuation' | 'risk';
  /** Time period */
  timePeriod?: {
    start: string;
    end: string;
  };
  /** Industry */
  industry?: string;
  ** Company size */
  companySize?: 'small' | 'medium' | 'large';
  ** Analysis depth */
  analysisDepth?: 'basic' | 'detailed' | 'comprehensive';
}

/**
 * Financial analysis response
 */
export interface FinancialAnalysisResponse {
  /** Analysis ID */
  analysisId: string;
  ** Analysis results */
  analysis: {
    summary: string;
    keyFindings: string[];
    ratios?: Record<string, {
      value: number;
      benchmark?: number;
      status: 'good' | 'average' | 'poor';
    }>;
    trends?: Array<{
      metric: string;
      direction: 'increasing' | 'decreasing' | 'stable';
      change: number;
      significance: 'high' | 'medium' | 'low';
    }>;
    forecasts?: Array<{
      metric: string;
      period: string;
      predicted: number;
      confidence: number;
    }>;
  };
  ** Risk assessment */
  riskAssessment: {
    overallRisk: 'low' | 'medium' | 'high';
    riskFactors: Array<{
      factor: string;
      level: 'low' | 'medium' | 'high';
      impact: 'low' | 'medium' | 'high';
      description: string;
    }>;
  };
  ** Recommendations */
  recommendations: Array<{
    priority: 'high' | 'medium' | 'low';
    action: string;
    expectedImpact: string;
    timeline: string;
  }>;
  /** Analysis time */
  analysisTime: number;
}

/**
 * Transaction audit request
 */
export interface TransactionAuditRequest {
  /** Transaction data */
  transactions: Array<{
    id: string;
    date: string;
    amount: number;
    description: string;
    category: string;
    account: string;
    counterparty?: string;
    reference?: string;
  }>;
  /** Audit type */
  auditType: 'fraud' | 'compliance' | 'error' | 'anomaly';
  /** Risk threshold */
  riskThreshold?: number;
  ** Focus areas */
  focusAreas?: string[];
  ** Time period */
  timePeriod?: {
    start: string;
    end: string;
  };
  ** Account types */
  accountTypes?: string[];
}

/**
 * Transaction audit response
 */
export interface TransactionAuditResponse {
  /** Audit ID */
  auditId: string;
  ** Summary */
  summary: {
    totalTransactions: number;
    flaggedTransactions: number;
    riskScore: number;
    auditPeriod: string;
  };
  ** Flagged transactions */
  flaggedTransactions: Array<{
    transactionId: string;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    riskScore: number;
    flags: string[];
    explanation: string;
    recommendedAction: string;
  }>;
  ** Patterns detected */
  patterns: Array<{
    pattern: string;
    frequency: number;
    riskLevel: 'low' | 'medium' | 'high';
    description: string;
  }>;
  ** Compliance status */
  complianceStatus: {
    overall: 'compliant' | 'non-compliant' | 'partial';
    issues: Array<{
      regulation: string;
      status: 'compliant' | 'non-compliant';
      description: string;
      severity: 'low' | 'medium' | 'high';
    }>;
  };
  /** Recommendations */
  recommendations: Array<{
    category: 'process' | 'system' | 'training' | 'policy';
    recommendation: string;
    priority: 'high' | 'medium' | 'low';
  }>;
  /** Audit time */
  auditTime: number;
}

/**
 * Risk assessment request
 */
export interface RiskAssessmentRequest {
  /** Entity information */
  entity: {
    type: 'individual' | 'business' | 'investment';
    name: string;
    industry?: string;
    size?: string;
    location?: string;
  };
  /** Assessment type */
  assessmentType: 'credit' | 'market' | 'operational' | 'liquidity' | 'compliance';
  /** Time horizon */
  timeHorizon: 'short-term' | 'medium-term' | 'long-term';
  /** Risk factors to consider */
  riskFactors?: string[];
  ** Historical data */
  historicalData?: Record<string, any>;
}

/**
 * Risk assessment response
 */
export interface RiskAssessmentResponse {
  /** Assessment ID */
  assessmentId: string;
  ** Overall risk rating */
  overallRiskRating: {
    score: number; // 0-100
    level: 'low' | 'medium' | 'high' | 'critical';
    confidence: number; // 0-100
  };
  ** Risk breakdown */
  riskBreakdown: Array<{
    category: string;
    score: number;
    level: 'low' | 'medium' | 'high' | 'critical';
    factors: Array<{
      factor: string;
      impact: number;
      likelihood: number;
      description: string;
    }>;
  }>;
  ** Key risk drivers */
  keyRiskDrivers: Array<{
    driver: string;
    impact: 'high' | 'medium' | 'low';
    trend: 'increasing' | 'stable' | 'decreasing';
    description: string;
  }>;
  ** Mitigation strategies */
  mitigationStrategies: Array<{
    risk: string;
    strategy: string;
    effectiveness: 'high' | 'medium' | 'low';
      cost: 'low' | 'medium' | 'high';
      timeline: string;
  }>;
  ** Monitoring recommendations */
  monitoringRecommendations: Array<{
    metric: string;
    frequency: string;
    threshold: number;
    action: string;
  }>;
  /** Assessment time */
  assessmentTime: number;
}

/**
 * Market prediction request
 */
export interface MarketPredictionRequest {
  /** Market/Instrument */
  market: {
    type: 'stock' | 'bond' | 'commodity' | 'currency' | 'index';
    symbol: string;
    name: string;
  };
  /** Prediction type */
  predictionType: 'price' | 'trend' | 'volatility' | 'correlation';
  /** Time horizon */
  timeHorizon: '1d' | '1w' | '1m' | '3m' | '6m' | '1y';
  /** Confidence level */
  confidenceLevel?: number;
  /** Include factors */
  includeFactors?: string[];
}

/**
 * Market prediction response
 */
export interface MarketPredictionResponse {
  /** Prediction ID */
  predictionId: string;
  ** Market info */
  market: {
    type: string;
    symbol: string;
    name: string;
    currentPrice?: number;
  };
  ** Predictions */
  predictions: Array<{
    type: string;
    horizon: string;
    prediction: any;
    confidence: number;
    factors: Array<{
      factor: string;
      impact: 'positive' | 'negative';
      weight: number;
    }>;
  }>;
  ** Technical indicators */
  technicalIndicators?: Record<string, {
    value: number;
    signal: 'buy' | 'sell' | 'hold';
    strength: 'weak' | 'moderate' | 'strong';
  }>;
  ** Risk factors */
  riskFactors: Array<{
    factor: string;
    level: 'low' | 'medium' | 'high';
    description: string;
  }>;
  ** Disclaimer */
  disclaimer: string;
  /** Prediction time */
  predictionTime: number;
}

/**
 * Finance AI domain-specific client
 */
export class FinanceAIClient {
  private client: BharatAIClient;
  private domainEndpoint: string;

  /**
   * Create a new Finance AI client
   */
  constructor(client: BharatAIClient, domainEndpoint: string = '/finance') {
    this.client = client;
    this.domainEndpoint = domainEndpoint;
  }

  /**
   * Analyze financial data
   */
  async analyzeFinancials(request: FinancialAnalysisRequest): Promise<FinancialAnalysisResponse> {
    const response = await this.client['makeRequest']<FinancialAnalysisResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/analyze-financials`,
      data: {
        financial_data: request.financialData,
        analysis_type: request.analysisType,
        time_period: request.timePeriod,
        industry: request.industry,
        company_size: request.companySize,
        analysis_depth: request.analysisDepth
      }
    });

    return response;
  }

  /**
   * Audit transactions
   */
  async auditTransactions(request: TransactionAuditRequest): Promise<TransactionAuditResponse> {
    const response = await this.client['makeRequest']<TransactionAuditResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/audit-transactions`,
      data: {
        transactions: request.transactions,
        audit_type: request.auditType,
        risk_threshold: request.riskThreshold,
        focus_areas: request.focusAreas,
        time_period: request.timePeriod,
        account_types: request.accountTypes
      }
    });

    return response;
  }

  /**
   * Assess risk
   */
  async assessRisk(request: RiskAssessmentRequest): Promise<RiskAssessmentResponse> {
    const response = await this.client['makeRequest']<RiskAssessmentResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/assess-risk`,
      data: {
        entity: request.entity,
        assessment_type: request.assessmentType,
        time_horizon: request.timeHorizon,
        risk_factors: request.riskFactors,
        historical_data: request.historicalData
      }
    });

    return response;
  }

  /**
   * Predict market movements
   */
  async predictMarket(request: MarketPredictionRequest): Promise<MarketPredictionResponse> {
    const response = await this.client['makeRequest']<MarketPredictionResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/predict-market`,
      data: {
        market: request.market,
        prediction_type: request.predictionType,
        time_horizon: request.timeHorizon,
        confidence_level: request.confidenceLevel,
        include_factors: request.includeFactors
      }
    });

    return response;
  }

  /**
   * Generate financial report
   */
  async generateFinancialReport(params: {
    reportType: 'annual' | 'quarterly' | 'monthly' | 'custom';
    financialData: any;
    includeCharts?: boolean;
    includeRatios?: boolean;
    includeForecast?: boolean;
    language?: string;
  }): Promise<{
    reportId: string;
    report: string;
    sections: Array<{
      title: string;
      content: string;
      charts?: Array<{
        type: string;
        title: string;
        data: any;
      }>;
    }>;
    summary: string;
    generatedAt: string;
  }> {
    const response = await this.client['makeRequest']<{
      reportId: string;
      report: string;
      sections: Array<{
        title: string;
        content: string;
        charts?: Array<{
          type: string;
          title: string;
          data: any;
        }>;
      }>;
      summary: string;
      generatedAt: string;
    }>({
      method: 'POST',
      url: `${this.domainEndpoint}/generate-report`,
      data: params
    });

    return response;
  }

  /**
   * Monitor transactions in real-time
   */
  async monitorTransactions(params: {
    accounts: string[];
    rules: Array<{
      condition: string;
      action: 'alert' | 'block' | 'review';
      threshold?: number;
    }>;
    webhookUrl?: string;
  }): Promise<{
    monitoringId: string;
    status: 'active' | 'paused' | 'stopped';
    accounts: string[];
    rules: Array<{
      condition: string;
      action: string;
      threshold?: number;
      triggered: number;
    }>;
    createdAt: string;
  }> {
    const response = await this.client['makeRequest']<{
      monitoringId: string;
      status: 'active' | 'paused' | 'stopped';
      accounts: string[];
      rules: Array<{
        condition: string;
        action: string;
        threshold?: number;
        triggered: number;
      }>;
      createdAt: string;
    }>({
      method: 'POST',
      url: `${this.domainEndpoint}/monitor-transactions`,
      data: params
    });

    return response;
  }

  /**
   * Get compliance requirements
   */
  async getComplianceRequirements(industry: string, jurisdiction: string): Promise<Array<{
    regulation: string;
    description: string;
    requirements: string[];
    penalties: string[];
    frequency: string;
    lastUpdated: string;
  }>> {
    const response = await this.client['makeRequest']<Array<{
      regulation: string;
      description: string;
      requirements: string[];
      penalties: string[];
      frequency: string;
      lastUpdated: string;
    }>>({
      method: 'GET',
      url: `${this.domainEndpoint}/compliance-requirements`,
      params: { industry, jurisdiction }
    });

    return response;
  }

  /**
   * Calculate financial ratios
   */
  async calculateRatios(financialData: any): Promise<Record<string, {
    value: number;
    formula: string;
    interpretation: string;
    benchmark?: {
      industry: string;
      value: number;
    };
  }>> {
    const response = await this.client['makeRequest']<Record<string, {
      value: number;
      formula: string;
      interpretation: string;
      benchmark?: {
        industry: string;
        value: number;
      };
    }>>({
      method: 'POST',
      url: `${this.domainEndpoint}/calculate-ratios`,
      data: { financial_data: financialData }
    });

    return response;
  }

  /**
   * Generate investment recommendation
   */
  async generateInvestmentRecommendation(params: {
    riskProfile: 'conservative' | 'moderate' | 'aggressive';
    investmentAmount: number;
    timeHorizon: string;
    goals: string[];
    currentHoldings?: Array<{
      asset: string;
      value: number;
      allocation: number;
    }>;
  }): Promise<{
    recommendationId: string;
    riskProfile: string;
    recommendedAllocation: Array<{
      assetClass: string;
      percentage: number;
      amount: number;
      rationale: string;
    }>;
    specificInvestments: Array<{
      name: string;
      type: string;
      allocation: number;
      expectedReturn: number;
      riskLevel: string;
    }>;
    expectedReturns: {
      annual: number;
      timeframe: string;
    };
    risks: Array<{
      type: string;
      level: string;
      mitigation: string;
    }>;
    nextReviewDate: string;
  }> {
    const response = await this.client['makeRequest']<{
      recommendationId: string;
      riskProfile: string;
      recommendedAllocation: Array<{
        assetClass: string;
        percentage: number;
        amount: number;
        rationale: string;
      }>;
      specificInvestments: Array<{
        name: string;
        type: string;
        allocation: number;
        expectedReturn: number;
        riskLevel: string;
      }>;
      expectedReturns: {
        annual: number;
        timeframe: string;
      };
      risks: Array<{
        type: string;
        level: string;
        mitigation: string;
      }>;
      nextReviewDate: string;
    }>({
      method: 'POST',
      url: `${this.domainEndpoint}/investment-recommendation`,
      data: params
    });

    return response;
  }
}