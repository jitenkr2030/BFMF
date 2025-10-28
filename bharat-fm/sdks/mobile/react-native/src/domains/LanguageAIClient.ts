/**
 * Language AI Domain Client for Bharat AI SDK
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

// Language-specific types
export interface TranslationRequest {
  /** Text to translate */
  text: string;
  /** Source language code */
  sourceLanguage: string;
  /** Target language code */
  targetLanguage: string;
  /** Domain-specific terminology */
  domain?: string;
  /** Formality level */
  formality?: 'formal' | 'informal' | 'neutral';
  /** Include confidence scores */
  includeConfidence?: boolean;
}

export interface TranslationResponse {
  /** Unique translation ID */
  id: string;
  /** Translated text */
  translatedText: string;
  /** Source language detected */
  detectedSourceLanguage?: string;
  /** Confidence score */
  confidence: number;
  /** Alternative translations */
  alternatives?: string[];
  /** Processing time */
  processingTime: number;
}

export interface LanguageDetectionRequest {
  /** Text to analyze */
  text: string;
  /** Include confidence scores */
  includeConfidence?: boolean;
  /** Return multiple possibilities */
  returnMultiple?: boolean;
  /** Minimum confidence threshold */
  minConfidence?: number;
}

export interface LanguageDetectionResponse {
  /** Unique detection ID */
  id: string;
  /** Detected language code */
  language: string;
  /** Language name */
  languageName: string;
  /** Confidence score */
  confidence: number;
  /** Alternative detections */
  alternatives?: Array<{
    language: string;
    languageName: string;
    confidence: number;
  }>;
  /** Processing time */
  processingTime: number;
}

export interface MultilingualRequest extends GenerationRequest {
  /** Target languages */
  targetLanguages: string[];
  /** Keep original text */
  includeOriginal?: boolean;
  /** Translate sequentially or in parallel */
  parallel?: boolean;
}

export interface MultilingualResponse {
  /** Unique response ID */
  id: string;
  /** Original text */
  originalText: string;
  /** Translations by language */
  translations: Array<{
    language: string;
    text: string;
    confidence: number;
  }>;
  /** Processing time */
  processingTime: number;
}

export class LanguageAIClient {
  private client: BharatAIClient;

  constructor(client: BharatAIClient) {
    this.client = client;
  }

  /**
   * Translate text between languages
   */
  public async translate(request: TranslationRequest): Promise<TranslationResponse> {
    try {
      const apiRequest: GenerationRequest = {
        prompt: `Translate the following text from ${request.sourceLanguage} to ${request.targetLanguage}:\n\n${request.text}`,
        language: request.targetLanguage,
        domain: request.domain || 'language',
        metadata: {
          sourceLanguage: request.sourceLanguage,
          targetLanguage: request.targetLanguage,
          formality: request.formality || 'neutral',
          operation: 'translate',
        },
      };

      const response = await this.client.generateText(apiRequest);
      
      return {
        id: response.id,
        translatedText: response.text,
        confidence: response.confidence,
        processingTime: response.processingTime,
      };
    } catch (error) {
      throw this.handleError(error, 'translate');
    }
  }

  /**
   * Detect language of text
   */
  public async detectLanguage(request: LanguageDetectionRequest): Promise<LanguageDetectionResponse> {
    try {
      const apiRequest: GenerationRequest = {
        prompt: `Detect the language of the following text and provide the language code and name:\n\n${request.text}`,
        metadata: {
          operation: 'detect_language',
          includeConfidence: request.includeConfidence,
          returnMultiple: request.returnMultiple,
          minConfidence: request.minConfidence,
        },
      };

      const response = await this.client.generateText(apiRequest);
      
      // Parse the response to extract language information
      const languageInfo = this.parseLanguageResponse(response.text);
      
      return {
        id: response.id,
        language: languageInfo.language,
        languageName: languageInfo.languageName,
        confidence: languageInfo.confidence,
        processingTime: response.processingTime,
      };
    } catch (error) {
      throw this.handleError(error, 'detectLanguage');
    }
  }

  /**
   * Generate text in multiple languages simultaneously
   */
  public async generateMultilingual(request: MultilingualRequest): Promise<MultilingualResponse> {
    try {
      const translations = [];
      const originalText = request.prompt;
      
      if (request.parallel) {
        // Process all languages in parallel
        const promises = request.targetLanguages.map(async (targetLang) => {
          const translationRequest: TranslationRequest = {
            text: originalText,
            sourceLanguage: request.language || 'en',
            targetLanguage: targetLang,
            domain: request.domain,
          };
          
          const translation = await this.translate(translationRequest);
          return {
            language: targetLang,
            text: translation.translatedText,
            confidence: translation.confidence,
          };
        });
        
        translations.push(...await Promise.all(promises));
      } else {
        // Process languages sequentially
        for (const targetLang of request.targetLanguages) {
          const translationRequest: TranslationRequest = {
            text: originalText,
            sourceLanguage: request.language || 'en',
            targetLanguage: targetLang,
            domain: request.domain,
          };
          
          const translation = await this.translate(translationRequest);
          translations.push({
            language: targetLang,
            text: translation.translatedText,
            confidence: translation.confidence,
          });
        }
      }

      return {
        id: `multi_${Date.now()}`,
        originalText,
        translations,
        processingTime: translations.reduce((sum, t) => sum + (t as any).processingTime || 0, 0),
      };
    } catch (error) {
      throw this.handleError(error, 'generateMultilingual');
    }
  }

  /**
   * Generate embeddings for language analysis
   */
  public async generateLanguageEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    try {
      const enhancedRequest: EmbeddingRequest = {
        ...request,
        metadata: {
          ...request.metadata,
          domain: 'language',
          operation: 'language_embeddings',
        },
      };

      return await this.client.generateEmbeddings(enhancedRequest);
    } catch (error) {
      throw this.handleError(error, 'generateLanguageEmbeddings');
    }
  }

  /**
   * Stream translation response
   */
  public async translateStream(
    request: TranslationRequest,
    onChunk: (chunk: StreamingResponse) => void
  ): Promise<TranslationResponse> {
    try {
      const apiRequest: GenerationRequest = {
        prompt: `Translate the following text from ${request.sourceLanguage} to ${request.targetLanguage}:\n\n${request.text}`,
        language: request.targetLanguage,
        domain: 'language',
        stream: true,
        metadata: {
          sourceLanguage: request.sourceLanguage,
          targetLanguage: request.targetLanguage,
          formality: request.formality || 'neutral',
          operation: 'translate_stream',
        },
      };

      const response = await this.client.generateTextStream(apiRequest, onChunk);
      
      return {
        id: response.id,
        translatedText: response.text,
        confidence: response.confidence,
        processingTime: response.processingTime,
      };
    } catch (error) {
      throw this.handleError(error, 'translateStream');
    }
  }

  /**
   * Get supported languages for translation
   */
  public async getSupportedLanguages(): Promise<Array<{
    code: string;
    name: string;
    nativeName: string;
    supportLevel: 'full' | 'partial' | 'experimental';
  }>> {
    try {
      const supportedResponse = await this.client.getSupportedLanguages();
      return supportedResponse.languages.map(lang => ({
        code: lang.code,
        name: lang.name,
        nativeName: lang.nativeName,
        supportLevel: lang.supportLevel,
      }));
    } catch (error) {
      throw this.handleError(error, 'getSupportedLanguages');
    }
  }

  /**
   * Batch translate multiple texts
   */
  public async batchTranslate(requests: TranslationRequest[]): Promise<TranslationResponse[]> {
    try {
      const batchRequests = requests.map(req => ({
        prompt: `Translate the following text from ${req.sourceLanguage} to ${req.targetLanguage}:\n\n${req.text}`,
        language: req.targetLanguage,
        domain: 'language',
        metadata: {
          sourceLanguage: req.sourceLanguage,
          targetLanguage: req.targetLanguage,
          formality: req.formality || 'neutral',
          operation: 'batch_translate',
        },
      }));

      const batchResponse = await this.client.generateBatch({
        prompts: batchRequests.map(req => req.prompt),
        parallel: true,
      });

      return batchResponse.responses.map((response, index) => ({
        id: response.id,
        translatedText: response.text,
        confidence: response.confidence,
        processingTime: response.processingTime,
      }));
    } catch (error) {
      throw this.handleError(error, 'batchTranslate');
    }
  }

  /**
   * Parse language detection response
   */
  private parseLanguageResponse(responseText: string): {
    language: string;
    languageName: string;
    confidence: number;
  } {
    try {
      // Try to parse JSON response first
      const parsed = JSON.parse(responseText);
      return {
        language: parsed.language || 'en',
        languageName: parsed.languageName || 'English',
        confidence: parsed.confidence || 0.95,
      };
    } catch {
      // Fallback to regex parsing
      const langMatch = responseText.match(/language[:\s]+([a-z]{2,3})/i);
      const nameMatch = responseText.match(/name[:\s]+([a-zA-Z\s]+)/i);
      const confMatch = responseText.match(/confidence[:\s]+([0-9.]+)/i);

      return {
        language: langMatch?.[1] || 'en',
        languageName: nameMatch?.[1]?.trim() || 'English',
        confidence: confMatch?.[1] ? parseFloat(confMatch[1]) : 0.95,
      };
    }
  }

  /**
   * Handle errors with domain context
   */
  private handleError(error: any, operation: string): BharatAIError {
    if (error instanceof BharatAIError) {
      return error;
    }

    return new BharatAIError(
      'LANGUAGE_AI_ERROR',
      `Language AI operation '${operation}' failed: ${error?.message || 'Unknown error'}`,
      error
    );
  }
}