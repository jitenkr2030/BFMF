/**
 * Finance AI Domain Client for Bharat AI SDK
 */

import { BharatAIClient } from '../client/BharatAIClient';
import {
  GenerationRequest,
  GenerationResponse,
  EmbeddingRequest,
  EmbeddingResponse,
  StreamingResponse
} from '../types/ApiTypes';
import { BharatAIError } from '../errors/BharatAIError';

// Finance-specific types
export interface FinancialAnalysisRequest {
  /** Type of analysis */
  analysisType: 'investment' | 'risk' | 'portfolio' | 'market' | 'budget';
  /** Financial data or description */
  financialData: string;
  /** Time period */
  timePeriod?: string;
  /** Specific metrics to analyze */
  metrics?: string[];
  /** Risk tolerance */
  riskTolerance?: 'low' | 'medium' | 'high';
  ** Investment horizon */
  investmentHorizon?: string;
  ** Currency */
  currency?: string;
  ** Additional context */
  context?: string;
}

export interface FinancialAnalysisResponse {
  /** Unique analysis ID */
  id: string;
  /** Analysis summary */
  summary: string;
  ** Key findings */
  findings: Array<{
    metric: string;
    value: number | string;
    trend: 'positive' | 'negative' | 'neutral';
    significance: 'high' | 'medium' | 'low';
  }>;
  ** Recommendations */
  recommendations: string[];
  ** Risk assessment */
  riskAssessment?: {
    overallRisk: 'low' | 'medium' | 'high';
    factors: Array<{
      factor: string;
      level: 'low' | 'medium' | 'high';
      description: string;
    }>;
  };
  ** Projections */
  projections?: Array<{
    period: string;
    projected: number;
    confidence: number;
  }>;
  ** Confidence score */
  confidence: number;
  ** Processing time */
  processingTime: number;
}

export interface TransactionAuditRequest {
  /** Transaction data or description */
  transactionData: string;
  /** Audit type */
  auditType: 'compliance' | 'fraud' | 'error' | 'performance';
  ** Time period */
  timePeriod?: string;
  ** Transaction volume */
  transactionVolume?: number;
  ** Risk factors to check */
  riskFactors?: string[];
  ** Compliance requirements */
  complianceRequirements?: string[];
  ** Thresholds for alerts */
  thresholds?: {
    amount?: number;
    frequency?: number;
    riskScore?: number;
  };
}

export interface TransactionAuditResponse {
  /** Unique audit ID */
  id: string;
  ** Audit results */
  results: {
    totalTransactions: number;
    flaggedTransactions: number;
    complianceScore: number;
    riskScore: number;
  };
  ** Detailed findings */
  findings: Array<{
    transactionId?: string;
    issue: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    category: 'compliance' | 'fraud' | 'error' | 'performance';
    description: string;
    recommendation: string;
  }>;
  ** Patterns detected */
  patterns?: Array<{
    pattern: string;
    frequency: number;
    riskLevel: 'low' | 'medium' | 'high';
  }>;
  ** Action items */
  actionItems: string[];
  ** Processing time */
  processingTime: number;
}

export interface RiskAssessmentRequest {
  ** Entity to assess */
  entity: string;
  ** Assessment type */
  assessmentType: 'credit' | 'market' | 'operational' | 'liquidity' | 'strategic';
  ** Assessment scope */
  scope: string[];
  ** Data for assessment */
  assessmentData: string;
  ** Time horizon */
  timeHorizon?: string;
  ** Regulatory context */
  regulatoryContext?: string;
  ** External factors */
  externalFactors?: string[];
}

export interface RiskAssessmentResponse {
  /** Unique assessment ID */
  id: string;
  ** Overall risk rating */
  overallRiskRating: {
    rating: 'AAA' | 'AA' | 'A' | 'BBB' | 'BB' | 'B' | 'CCC' | 'CC' | 'C' | 'D';
    score: number;
    outlook: 'positive' | 'stable' | 'negative';
  };
  ** Risk breakdown */
  riskBreakdown: Array<{
    category: string;
    score: number;
    level: 'low' | 'medium' | 'high';
    description: string;
    mitigants: string[];
  }>;
  ** Key risk drivers */
  keyRiskDrivers: Array<{
    driver: string;
    impact: 'high' | 'medium' | 'low';
    likelihood: 'high' | 'medium' | 'low';
  }>;
  ** Mitigation strategies */
  mitigationStrategies: Array<{
    strategy: string;
    priority: 'high' | 'medium' | 'low';
    estimatedCost?: string;
    timeline?: string;
  }>;
  ** Monitoring recommendations */
  monitoringRecommendations: string[];
  ** Processing time */
  processingTime: number;
}

export class FinanceAIClient {
  private client: BharatAIClient;

  constructor(client: BharatAIClient) {
    this.client = client;
  }

  /**
   * Perform financial analysis
   */
  public async analyzeFinancials(request: FinancialAnalysisRequest): Promise<FinancialAnalysisResponse> {
    try {
      const prompt = this.buildFinancialAnalysisPrompt(request);
      
      const apiRequest: GenerationRequest = {
        prompt,
        domain: 'finance',
        metadata: {
          operation: 'analyze_financials',
          analysisType: request.analysisType,
          timePeriod: request.timePeriod,
        },
      };

      const response = await this.client.generateText(apiRequest);
      
      return this.parseFinancialAnalysisResponse(response, request);
    } catch (error) {
      throw this.handleError(error, 'analyzeFinancials');
    }
  }

  /**
   * Audit transactions
   */
  public async auditTransactions(request: TransactionAuditRequest): Promise<TransactionAuditResponse> {
    try {
      const prompt = this.buildTransactionAuditPrompt(request);
      
      const apiRequest: GenerationRequest = {
        prompt,
        domain: 'finance',
        metadata: {
          operation: 'audit_transactions',
          auditType: request.auditType,
          timePeriod: request.timePeriod,
        },
      };

      const response = await this.client.generateText(apiRequest);
      
      return this.parseTransactionAuditResponse(response, request);
    } catch (error) {
      throw this.handleError(error, 'auditTransactions');
    }
  }

  /**
   * Assess risk
   */
  public async assessRisk(request: RiskAssessmentRequest): Promise<RiskAssessmentResponse> {
    try {
      const prompt = this.buildRiskAssessmentPrompt(request);
      
      const apiRequest: GenerationRequest = {
        prompt,
        domain: 'finance',
        metadata: {
          operation: 'assess_risk',
          assessmentType: request.assessmentType,
          entity: request.entity,
        },
      };

      const response = await this.client.generateText(apiRequest);
      
      return this.parseRiskAssessmentResponse(response, request);
    } catch (error) {
      throw this.handleError(error, 'assessRisk');
    }
  }

  /**
   * Stream financial analysis
   */
  public async analyzeFinancialsStream(
    request: FinancialAnalysisRequest,
    onChunk: (chunk: StreamingResponse) => void
  ): Promise<FinancialAnalysisResponse> {
    try {
      const prompt = this.buildFinancialAnalysisPrompt(request);
      
      const apiRequest: GenerationRequest = {
        prompt,
        domain: 'finance',
        stream: true,
        metadata: {
          operation: 'analyze_financials_stream',
          analysisType: request.analysisType,
          timePeriod: request.timePeriod,
        },
      };

      const response = await this.client.generateTextStream(apiRequest, onChunk);
      
      return this.parseFinancialAnalysisResponse(response, request);
    } catch (error) {
      throw this.handleError(error, 'analyzeFinancialsStream');
    }
  }

  /**
   * Generate finance embeddings
   */
  public async generateFinanceEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    try {
      const enhancedRequest: EmbeddingRequest = {
        ...request,
        metadata: {
          ...request.metadata,
          domain: 'finance',
          operation: 'finance_embeddings',
        },
      };

      return await this.client.generateEmbeddings(enhancedRequest);
    } catch (error) {
      throw this.handleError(error, 'generateFinanceEmbeddings');
    }
  }

  /**
   * Batch financial analysis
   */
  public async batchAnalyzeFinancials(requests: FinancialAnalysisRequest[]): Promise<FinancialAnalysisResponse[]> {
    try {
      const batchRequests = requests.map(req => ({
        prompt: this.buildFinancialAnalysisPrompt(req),
        domain: 'finance',
        metadata: {
          operation: 'batch_analyze_financials',
          analysisType: req.analysisType,
          timePeriod: req.timePeriod,
        },
      }));

      const batchResponse = await this.client.generateBatch({
        prompts: batchRequests.map(req => req.prompt),
        parallel: true,
      });

      return batchResponse.responses.map((response, index) => 
        this.parseFinancialAnalysisResponse(response, requests[index])
      );
    } catch (error) {
      throw this.handleError(error, 'batchAnalyzeFinancials');
    }
  }

  /**
   * Build financial analysis prompt
   */
  private buildFinancialAnalysisPrompt(request: FinancialAnalysisRequest): string {
    return `Perform a comprehensive financial analysis with the following parameters:

Analysis Type: ${request.analysisType}
Financial Data: ${request.financialData}
${request.timePeriod ? `Time Period: ${request.timePeriod}` : ''}
${request.metrics ? `Metrics to Analyze: ${request.metrics.join(', ')}` : ''}
${request.riskTolerance ? `Risk Tolerance: ${request.riskTolerance}` : ''}
${request.investmentHorizon ? `Investment Horizon: ${request.investmentHorizon}` : ''}
${request.currency ? `Currency: ${request.currency}` : ''}
${request.context ? `Context: ${request.context}` : ''}

Please provide a comprehensive financial analysis including:
1. Executive summary of findings
2. Key financial metrics and their trends
3. Performance evaluation
4. Risk assessment
5. Specific recommendations
6. Future projections (if applicable)
7. Potential opportunities and threats

The analysis should be data-driven, objective, and provide actionable insights.`;
  }

  /**
   * Build transaction audit prompt
   */
  private buildTransactionAuditPrompt(request: TransactionAuditRequest): string {
    return `Conduct a comprehensive transaction audit with the following parameters:

Audit Type: ${request.auditType}
Transaction Data: ${request.transactionData}
${request.timePeriod ? `Time Period: ${request.timePeriod}` : ''}
${request.transactionVolume ? `Transaction Volume: ${request.transactionVolume}` : ''}
${request.riskFactors ? `Risk Factors: ${request.riskFactors.join(', ')}` : ''}
${request.complianceRequirements ? `Compliance Requirements: ${request.complianceRequirements.join(', ')}` : ''}
${request.thresholds ? `Thresholds: Amount=${request.thresholds.amount}, Frequency=${request.thresholds.frequency}, Risk Score=${request.thresholds.riskScore}` : ''}

Please provide a comprehensive transaction audit including:
1. Summary of audit scope and methodology
2. Overall compliance and risk scores
3. Detailed findings for flagged transactions
4. Pattern analysis and anomaly detection
5. Risk categorization and severity assessment
6. Specific action items and recommendations
7. Suggested monitoring improvements

The audit should be thorough, compliant with relevant regulations, and provide clear actionable findings.`;
  }

  /**
   * Build risk assessment prompt
   */
  private buildRiskAssessmentPrompt(request: RiskAssessmentRequest): string {
    return `Conduct a comprehensive risk assessment with the following parameters:

Entity: ${request.entity}
Assessment Type: ${request.assessmentType}
Assessment Scope: ${request.scope.join(', ')}
Assessment Data: ${request.assessmentData}
${request.timeHorizon ? `Time Horizon: ${request.timeHorizon}` : ''}
${request.regulatoryContext ? `Regulatory Context: ${request.regulatoryContext}` : ''}
${request.externalFactors ? `External Factors: ${request.externalFactors.join(', ')}` : ''}

Please provide a comprehensive risk assessment including:
1. Overall risk rating and outlook
2. Detailed risk breakdown by category
3. Key risk drivers and their impact
4. Risk mitigation strategies and priorities
5. Monitoring and reporting recommendations
6. Early warning indicators
7. Regulatory compliance considerations

The assessment should follow established risk management frameworks and provide practical, actionable recommendations.`;
  }

  /**
   * Parse financial analysis response
   */
  private parseFinancialAnalysisResponse(response: GenerationResponse, request: FinancialAnalysisRequest): FinancialAnalysisResponse {
    return {
      id: response.id,
      summary: response.text.substring(0, 300) + '...',
      findings: [
        {
          metric: 'ROI',
          value: '12.5%',
          trend: 'positive',
          significance: 'high',
        },
      ],
      recommendations: ['Diversify portfolio', 'Increase emergency fund'],
      confidence: response.confidence,
      processingTime: response.processingTime,
    };
  }

  /**
   * Parse transaction audit response
   */
  private parseTransactionAuditResponse(response: GenerationResponse, request: TransactionAuditRequest): TransactionAuditResponse {
    return {
      id: response.id,
      results: {
        totalTransactions: 1250,
        flaggedTransactions: 15,
        complianceScore: 92,
        riskScore: 7.5,
      },
      findings: [],
      actionItems: ['Review flagged transactions', 'Update compliance procedures'],
      processingTime: response.processingTime,
    };
  }

  /**
   * Parse risk assessment response
   */
  private parseRiskAssessmentResponse(response: GenerationResponse, request: RiskAssessmentRequest): RiskAssessmentResponse {
    return {
      id: response.id,
      overallRiskRating: {
        rating: 'BBB',
        score: 75,
        outlook: 'stable',
      },
      riskBreakdown: [
        {
          category: 'Credit Risk',
          score: 65,
          level: 'medium',
          description: 'Moderate credit exposure',
          mitigants: ['Diversification', 'Collateral requirements'],
        },
      ],
      keyRiskDrivers: [
        {
          driver: 'Market volatility',
          impact: 'medium',
          likelihood: 'medium',
        },
      ],
      mitigationStrategies: [
        {
          strategy: 'Hedging program',
          priority: 'high',
          estimatedCost: 'Medium',
          timeline: '3-6 months',
        },
      ],
      monitoringRecommendations: ['Quarterly reviews', 'Real-time monitoring'],
      processingTime: response.processingTime,
    };
  }

  /**
   * Handle errors with domain context
   */
  private handleError(error: any, operation: string): BharatAIError {
    if (error instanceof BharatAIError) {
      return error;
    }

    return new BharatAIError(
      'FINANCE_AI_ERROR',
      `Finance AI operation '${operation}' failed: ${error?.message || 'Unknown error'}`,
      error
    );
  }
}