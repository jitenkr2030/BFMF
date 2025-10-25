/**
 * Language AI domain-specific client for BFMF
 */

import { BharatAIClient } from '../client';
import { GenerationRequest, GenerationResponse } from '../types';
import { Language } from '../constants';
import { BharatAIError } from '../errors';

/**
 * Language-specific request parameters
 */
export interface LanguageGenerationRequest extends GenerationRequest {
  /** Source language for translation */
  sourceLanguage?: Language;
  /** Target language for translation */
  targetLanguage?: Language;
  /** Translation mode */
  mode?: 'generation' | 'translation' | 'transliteration' | 'code-switching';
}

/**
 * Translation request
 */
export interface TranslationRequest {
  /** Text to translate */
  text: string;
  /** Source language */
  sourceLanguage: Language;
  /** Target language */
  targetLanguage: Language;
  /** Whether to preserve formatting */
  preserveFormatting?: boolean;
}

/**
 * Translation response
 */
export interface TranslationResponse {
  /** Translated text */
  translatedText: string;
  /** Original text */
  originalText: string;
  /** Source language */
  sourceLanguage: Language;
  /** Target language */
  targetLanguage: Language;
  /** Confidence score */
  confidence?: number;
  /** Translation time */
  translationTime: number;
}

/**
 * Language detection request
 */
export interface LanguageDetectionRequest {
  /** Text to detect language for */
  text: string;
  /** Whether to return confidence scores */
  includeConfidence?: boolean;
}

/**
 * Language detection response
 */
export interface LanguageDetectionResponse {
  /** Detected language */
  detectedLanguage: Language;
  /** Confidence score (0-1) */
  confidence?: number;
  /** All language probabilities */
  probabilities?: Record<Language, number>;
  /** Detection time */
  detectionTime: number;
}

/**
 * Code-switching detection request
 */
export interface CodeSwitchingRequest {
  /** Text to analyze for code-switching */
  text: string;
  /** Minimum segment length */
  minSegmentLength?: number;
}

/**
 * Code-switching response
 */
export interface CodeSwitchingResponse {
  /** Original text */
  originalText: string;
  /** Detected segments */
  segments: Array<{
    text: string;
    language: Language;
    confidence: number;
    startIndex: number;
    endIndex: number;
  }>;
  /** Analysis time */
  analysisTime: number;
}

/**
 * Tokenization request
 */
export interface TokenizationRequest {
  /** Text to tokenize */
  text: string;
  /** Language hint */
  language?: Language;
  /** Tokenization type */
  type?: 'word' | 'subword' | 'sentence';
}

/**
 * Tokenization response
 */
export interface TokenizationResponse {
  /** Original text */
  originalText: string;
  /** Tokens */
  tokens: string[];
  /** Token IDs */
  tokenIds?: number[];
  /** Language used */
  language: Language;
  /** Tokenization time */
  tokenizationTime: number;
}

/**
 * Language AI domain-specific client
 */
export class LanguageAIClient {
  private client: BharatAIClient;
  private domainEndpoint: string;

  /**
   * Create a new Language AI client
   */
  constructor(client: BharatAIClient, domainEndpoint: string = '/language') {
    this.client = client;
    this.domainEndpoint = domainEndpoint;
  }

  /**
   * Generate text with language-specific capabilities
   */
  async generateText(request: LanguageGenerationRequest): Promise<GenerationResponse> {
    const response = await this.client['makeRequest']<GenerationResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/generate`,
      data: {
        prompt: request.prompt,
        max_tokens: request.maxTokens || 100,
        temperature: request.temperature || 1.0,
        top_p: request.topP || 1.0,
        top_k: request.topK || 50,
        num_beams: request.numBeams || 1,
        do_sample: request.doSample !== false,
        language: request.language,
        source_language: request.sourceLanguage,
        target_language: request.targetLanguage,
        mode: request.mode || 'generation'
      }
    });

    return response;
  }

  /**
   * Translate text between languages
   */
  async translate(request: TranslationRequest): Promise<TranslationResponse> {
    const response = await this.client['makeRequest']<TranslationResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/translate`,
      data: {
        text: request.text,
        source_language: request.sourceLanguage,
        target_language: request.targetLanguage,
        preserve_formatting: request.preserveFormatting !== false
      }
    });

    return response;
  }

  /**
   * Detect language of text
   */
  async detectLanguage(request: LanguageDetectionRequest): Promise<LanguageDetectionResponse> {
    const response = await this.client['makeRequest']<LanguageDetectionResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/detect-language`,
      data: {
        text: request.text,
        include_confidence: request.includeConfidence !== false
      }
    });

    return response;
  }

  /**
   * Detect code-switching in text
   */
  async detectCodeSwitching(request: CodeSwitchingRequest): Promise<CodeSwitchingResponse> {
    const response = await this.client['makeRequest']<CodeSwitchingResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/detect-code-switching`,
      data: {
        text: request.text,
        min_segment_length: request.minSegmentLength || 3
      }
    });

    return response;
  }

  /**
   * Tokenize text
   */
  async tokenize(request: TokenizationRequest): Promise<TokenizationResponse> {
    const response = await this.client['makeRequest']<TokenizationResponse>({
      method: 'POST',
      url: `${this.domainEndpoint}/tokenize`,
      data: {
        text: request.text,
        language: request.language,
        type: request.type || 'subword'
      }
    });

    return response;
  }

  /**
   * Batch translate multiple texts
   */
  async translateBatch(requests: TranslationRequest[]): Promise<TranslationResponse[]> {
    const response = await this.client['makeRequest']<TranslationResponse[]>({
      method: 'POST',
      url: `${this.domainEndpoint}/translate-batch`,
      data: requests.map(req => ({
        text: req.text,
        source_language: req.sourceLanguage,
        target_language: req.targetLanguage,
        preserve_formatting: req.preserveFormatting !== false
      }))
    });

    return response;
  }

  /**
   * Get supported language pairs for translation
   */
  async getSupportedLanguagePairs(): Promise<Array<{
    source: Language;
    target: Language;
    supported: boolean;
    quality: 'high' | 'medium' | 'low';
  }>> {
    const response = await this.client['makeRequest']<Array<{
      source: Language;
      target: Language;
      supported: boolean;
      quality: 'high' | 'medium' | 'low';
    }>>({
      method: 'GET',
      url: `${this.domainEndpoint}/supported-pairs`
    });

    return response;
  }

  /**
   * Get language-specific model information
   */
  async getLanguageModelInfo(language: Language): Promise<{
    modelName: string;
    supportedLanguages: Language[];
    capabilities: string[];
    version: string;
    lastTrained: string;
  }> {
    const response = await this.client['makeRequest']<{
      modelName: string;
      supportedLanguages: Language[];
      capabilities: string[];
      version: string;
      lastTrained: string;
    }>({
      method: 'GET',
      url: `${this.domainEndpoint}/model-info/${language}`
    });

    return response;
  }

  /**
   * Train a custom language model
   */
  async trainLanguageModel(params: {
    languages: Language[];
    dataset: string;
    modelType?: string;
    epochs?: number;
    batchSize?: number;
    learningRate?: number;
  }): Promise<{
    modelId: string;
    status: 'training' | 'completed' | 'failed';
    estimatedTime: number;
  }> {
    const response = await this.client['makeRequest']<{
      modelId: string;
      status: 'training' | 'completed' | 'failed';
      estimatedTime: number;
    }>({
      method: 'POST',
      url: `${this.domainEndpoint}/train`,
      data: {
        languages: params.languages,
        dataset: params.dataset,
        model_type: params.modelType,
        epochs: params.epochs || 3,
        batch_size: params.batchSize || 32,
        learning_rate: params.learningRate || 1e-4
      }
    });

    return response;
  }

  /**
   * Get training status
   */
  async getTrainingStatus(modelId: string): Promise<{
    modelId: string;
    status: 'training' | 'completed' | 'failed';
    progress: number;
    currentEpoch: number;
    totalEpochs: number;
    loss?: number;
    estimatedTimeRemaining?: number;
    error?: string;
  }> {
    const response = await this.client['makeRequest']<{
      modelId: string;
      status: 'training' | 'completed' | 'failed';
      progress: number;
      currentEpoch: number;
      totalEpochs: number;
      loss?: number;
      estimatedTimeRemaining?: number;
      error?: string;
    }>({
      method: 'GET',
      url: `${this.domainEndpoint}/train-status/${modelId}`
    });

    return response;
  }
}