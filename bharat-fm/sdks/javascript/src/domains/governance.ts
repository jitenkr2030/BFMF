/**
 * Governance AI domain-specific client for BFMF
 */

import { BharatAIClient } from '../client';
import { GenerationRequest, GenerationResponse } from '../types';
import { BharatAIError } from '../errors';

/**
 * RTI (Right to Information) request
 */
export interface RTIRequest {
  /** RTI application text */
  applicationText: string;
  /** Department/Ministry */
  department: string;
  /** Request type */
  requestType: 'information' | 'inspection' | 'sample';
  /** Urgency level */
  urgency?: 'normal' | 'urgent' | 'immediate';
  /** Language */
  language?: string;
}

/**
 * RTI response
 */
export interface RTIResponse {
  /** Generated RTI response */
  responseText: string;
  /** Original application */
  originalApplication: string;
  /** Department */
  department: string;
  /** Response time estimate */
  responseTimeEstimate: string;
  /** Relevant sections */
  relevantSections: Array<{
    section: string;
    description: string;
    relevance: 'high' | 'medium' | 'low';
  }>;
  /** Confidence score */
  confidence: number;
  /** Processing time */
  processingTime: number;
}

/**
 * Policy analysis request
 */
export interface PolicyAnalysisRequest {
  /** Policy document text */
  policyText: string;
  /** Analysis type */
  analysisType: 'summary' | 'impact' | 'compliance' | 'stakeholder' | 'implementation';
  /** Sector/Domain */
  sector?: string;
  /** Target audience */
  targetAudience?: string;
  /** Language */
  language?: string;
}

/**
 * Policy analysis response
 */
export interface PolicyAnalysisResponse {
  /** Analysis results */
  analysis: string;
  /** Key insights */
  keyInsights: string[];
  /** Recommendations */
  recommendations: string[];
  /** Risk assessment */
  riskAssessment?: {
    level: 'low' | 'medium' | 'high';
    factors: string[];
  };
  /** Compliance score */
  complianceScore?: number;
  /** Analysis time */
  analysisTime: number;
}

/**
 * Compliance audit request
 */
export interface ComplianceAuditRequest {
  /** Document to audit */
  documentText: string;
  /** Compliance framework */
  framework: string;
  /** Audit type */
  auditType: 'full' | 'targeted' | 'risk-based';
  /** Specific requirements to check */
  requirements?: string[];
  /** Industry/sector */
  industry?: string;
}

/**
 * Compliance audit response
 */
export interface ComplianceAuditResponse {
  /** Overall compliance score */
  complianceScore: number;
  /** Detailed findings */
  findings: Array<{
    requirement: string;
    status: 'compliant' | 'non-compliant' | 'partial';
    evidence: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    recommendation: string;
  }>;
  /** Summary */
  summary: string;
  /** Action items */
  actionItems: string[];
  /** Audit time */
  auditTime: number;
}

/**
 * Government scheme analysis request
 */
export interface SchemeAnalysisRequest {
  /** Scheme name or description */
  scheme: string;
  /** Analysis type */
  analysisType: 'eligibility' | 'benefits' | 'application' | 'comparison';
  /** User context */
  userContext?: {
    income?: string;
    category?: string;
    location?: string;
    occupation?: string;
  };
  /** Language */
  language?: string;
}

/**
 * Government scheme analysis response
 */
export interface SchemeAnalysisResponse {
  /** Scheme details */
  schemeDetails: {
    name: string;
    description: string;
    ministry: string;
    category: string;
  };
  /** Analysis results */
  analysis: string;
  /** Eligibility criteria */
  eligibilityCriteria?: string[];
  /** Benefits */
  benefits?: string[];
  /** Application process */
  applicationProcess?: string[];
  /** Similar schemes */
  similarSchemes?: Array<{
    name: string;
    similarity: number;
    keyDifferences: string[];
  }>;
  /** Analysis time */
  analysisTime: number;
}

/**
 * Governance AI domain-specific client
 */
export class GovernanceAIClient {
  private client: BharatAIClient;
  private domainEndpoint: string;

  /**
   * Create a new Governance AI client
   */
  constructor(client: BharatAIClient, domainEndpoint: string = '/governance') {
    this.client = client;
    this.domainEndpoint = domainEndpoint;
  }

  /**
   * Generate RTI response
   */
  async generateRTIResponse(request: RTIRequest): Promise<RTIResponse> {
    const response = await this.client['makeRequest']<RTIResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/rti-response`,
      data: {
        application_text: request.applicationText,
        department: request.department,
        request_type: request.requestType,
        urgency: request.urgency || 'normal',
        language: request.language
      }
    });

    return response;
  }

  /**
   * Analyze policy document
   */
  async analyzePolicy(request: PolicyAnalysisRequest): Promise<PolicyAnalysisResponse> {
    const response = await this.client['makeRequest']<PolicyAnalysisResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/analyze-policy`,
      data: {
        policy_text: request.policyText,
        analysis_type: request.analysisType,
        sector: request.sector,
        target_audience: request.targetAudience,
        language: request.language
      }
    });

    return response;
  }

  /**
   * Conduct compliance audit
   */
  async conductComplianceAudit(request: ComplianceAuditRequest): Promise<ComplianceAuditResponse> {
    const response = await this.client['makeRequest']<ComplianceAuditResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/compliance-audit`,
      data: {
        document_text: request.documentText,
        framework: request.framework,
        audit_type: request.auditType,
        requirements: request.requirements,
        industry: request.industry
      }
    });

    return response;
  }

  /**
   * Analyze government scheme
   */
  async analyzeScheme(request: SchemeAnalysisRequest): Promise<SchemeAnalysisResponse> {
    const response = await this.client['makeRequest']<SchemeAnalysisResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/analyze-scheme`,
      data: {
        scheme: request.scheme,
        analysis_type: request.analysisType,
        user_context: request.userContext,
        language: request.language
      }
    });

    return response;
  }

  /**
   * Batch RTI response generation
   */
  async generateRTIResponseBatch(requests: RTIRequest[]): Promise<RTIResponse[]> {
    const response = await this.client['makeRequest']<RTIResponse[]>({
      method: 'POST',
      url: `${this.domainEndpoint}/rti-response-batch`,
      data: requests.map(req => ({
        application_text: req.applicationText,
        department: req.department,
        request_type: req.requestType,
        urgency: req.urgency || 'normal',
        language: req.language
      }))
    });

    return response;
  }

  /**
   * Get list of supported departments
   */
  async getSupportedDepartments(): Promise<Array<{
    name: string;
    code: string;
    description: string;
    categories: string[];
  }>> {
    const response = await this.client['makeRequest']<Array<{
      name: string;
      code: string;
      description: string;
      categories: string[];
    }>>({
      method: 'GET',
      url: `${this.domainEndpoint}/departments`
    });

    return response;
  }

  /**
   * Get compliance frameworks
   */
  async getComplianceFrameworks(): Promise<Array<{
    name: string;
    description: string;
    industry: string;
    version: string;
    requirements: string[];
  }>> {
    const response = await this.client['makeRequest']<Array<{
      name: string;
      description: string;
      industry: string;
      version: string;
      requirements: string[];
    }>>({
      method: 'GET',
      url: `${this.domainEndpoint}/compliance-frameworks`
    });

    return response;
  }

  /**
   * Get government schemes database
   */
  async getGovernmentSchemes(filters?: {
    ministry?: string;
    category?: string;
    beneficiary?: string;
    state?: string;
  }): Promise<Array<{
    name: string;
    description: string;
    ministry: string;
    category: string;
    beneficiary: string;
    eligibility: string[];
    benefits: string[];
    applicationProcess: string[];
    website?: string;
    contact?: string;
  }>> {
    const response = await this.client['makeRequest']<Array<{
      name: string;
      description: string;
      ministry: string;
      category: string;
      beneficiary: string;
      eligibility: string[];
      benefits: string[];
      applicationProcess: string[];
      website?: string;
      contact?: string;
    }>>({
      method: 'GET',
      url: `${this.domainEndpoint}/schemes`,
      params: filters
    });

    return response;
  }

  /**
   * Generate policy draft
   */
  async generatePolicyDraft(params: {
    title: string;
    scope: string;
    objectives: string[];
    stakeholders: string[];
    existingPolicies?: string[];
    language?: string;
  }): Promise<{
    draft: string;
    sections: Array<{
      title: string;
      content: string;
      importance: 'high' | 'medium' | 'low';
    }>;
    estimatedReviewTime: string;
  }> {
    const response = await this.client['makeRequest']<{
      draft: string;
      sections: Array<{
        title: string;
        content: string;
        importance: 'high' | 'medium' | 'low';
      }>;
      estimatedReviewTime: string;
    }>({
      method: 'POST',
      url: `${this.domainEndpoint}/generate-policy-draft`,
      data: params
    });

    return response;
  }

  /**
   * Track RTI application status
   */
  async trackRTIStatus(rtiId: string): Promise<{
    status: 'received' | 'processing' | 'responded' | 'rejected' | 'appealed';
    currentDepartment: string;
    estimatedResponseDate?: string;
    lastUpdate: string;
    updates: Array<{
      date: string;
      status: string;
      remarks: string;
      officer?: string;
    }>;
  }> {
    const response = await this.client['makeRequest']<{
      status: 'received' | 'processing' | 'responded' | 'rejected' | 'appealed';
      currentDepartment: string;
      estimatedResponseDate?: string;
      lastUpdate: string;
      updates: Array<{
        date: string;
        status: string;
        remarks: string;
        officer?: string;
      }>;
    }>({
      method: 'GET',
      url: `${this.domainEndpoint}/rti-status/${rtiId}`
    });

    return response;
  }
}