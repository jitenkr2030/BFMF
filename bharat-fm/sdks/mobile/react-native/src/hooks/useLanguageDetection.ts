/**
 * Hook for language detection functionality
 */

import { useState, useCallback } from 'react';
import { BharatAIClient } from '../client/BharatAIClient';
import { LanguageAIClient, LanguageDetectionRequest, LanguageDetectionResponse } from '../domains/LanguageAIClient';
import { BharatAIError } from '../errors/BharatAIError';

interface UseLanguageDetectionProps {
  /** Bharat AI client instance */
  client: BharatAIClient;
  /** Auto-detect on text change */
  autoDetect?: boolean;
  /** Detection debounce time in milliseconds */
  debounceTime?: number;
}

interface UseLanguageDetectionReturn {
  /** Detected language */
  detectedLanguage: string | null;
  ** Detected language name */
  detectedLanguageName: string | null;
  ** Confidence score */
  confidence: number | null;
  ** Detection in progress */
  isDetecting: boolean;
  ** Detection error */
  error: BharatAIError | null;
  ** Detect language manually */
  detectLanguage: (text: string, options?: Partial<LanguageDetectionRequest>) => Promise<LanguageDetectionResponse>;
  ** Clear detection results */
  clearResults: () => void;
  ** Alternative detections */
  alternatives: Array<{
    language: string;
    languageName: string;
    confidence: number;
  }>;
}

export function useLanguageDetection({
  client,
  autoDetect = false,
  debounceTime = 500,
}: UseLanguageDetectionProps): UseLanguageDetectionReturn {
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const [detectedLanguageName, setDetectedLanguageName] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<BharatAIError | null>(null);
  const [alternatives, setAlternatives] = useState<Array<{
    language: string;
    languageName: string;
    confidence: number;
  }>>([]);

  const languageClient = new LanguageAIClient(client);

  const detectLanguage = useCallback(async (
    text: string,
    options: Partial<LanguageDetectionRequest> = {}
  ): Promise<LanguageDetectionResponse> => {
    if (!text.trim()) {
      throw new BharatAIError('VALIDATION_ERROR', 'Text cannot be empty');
    }

    setIsDetecting(true);
    setError(null);

    try {
      const request: LanguageDetectionRequest = {
        text: text.trim(),
        includeConfidence: true,
        returnMultiple: true,
        minConfidence: 0.5,
        ...options,
      };

      const response = await languageClient.detectLanguage(request);

      if (response.language) {
        setDetectedLanguage(response.language);
        setDetectedLanguageName(response.languageName);
        setConfidence(response.confidence);
        setAlternatives(response.alternatives || []);
      }

      return response;
    } catch (err) {
      const aiError = err instanceof BharatAIError 
        ? err 
        : new BharatAIError('LANGUAGE_DETECTION_ERROR', 'Failed to detect language', err);
      
      setError(aiError);
      throw aiError;
    } finally {
      setIsDetecting(false);
    }
  }, [languageClient]);

  const clearResults = useCallback(() => {
    setDetectedLanguage(null);
    setDetectedLanguageName(null);
    setConfidence(null);
    setError(null);
    setAlternatives([]);
  }, []);

  return {
    detectedLanguage,
    detectedLanguageName,
    confidence,
    isDetecting,
    error,
    detectLanguage,
    clearResults,
    alternatives,
  };
}