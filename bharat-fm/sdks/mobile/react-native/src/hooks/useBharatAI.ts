/**
 * Main hook for using Bharat AI SDK
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { BharatAIClient } from '../client/BharatAIClient';
import { ClientConfig } from '../types/ClientConfig';
import { BharatAIError } from '../errors/BharatAIError';
import { DEFAULT_CONFIG } from '../constants';

interface UseBharatAIProps {
  /** Initial configuration */
  config?: Partial<ClientConfig>;
  /** Auto-initialize on mount */
  autoInitialize?: boolean;
}

interface UseBharatAIReturn {
  /** Bharat AI client instance */
  client: BharatAIClient | null;
  /** Initialization status */
  isInitialized: boolean;
  /** Loading state */
  isLoading: boolean;
  /** Error state */
  error: BharatAIError | null;
  /** Initialize client manually */
  initialize: () => Promise<void>;
  /** Update configuration */
  updateConfig: (newConfig: Partial<ClientConfig>) => void;
  /** Get current configuration */
  getConfig: () => ClientConfig | null;
  /** Reset client */
  reset: () => void;
}

export function useBharatAI({
  config = {},
  autoInitialize = true,
}: UseBharatAIProps = {}): UseBharatAIReturn {
  const [client, setClient] = useState<BharatAIClient | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<BharatAIError | null>(null);
  
  const configRef = useRef<ClientConfig>({ ...DEFAULT_CONFIG, ...config });
  const isMountedRef = useRef(true);

  const initialize = useCallback(async () => {
    if (!isMountedRef.current) return;
    
    setIsLoading(true);
    setError(null);

    try {
      const newClient = new BharatAIClient(configRef.current);
      await newClient.initialize();
      
      if (isMountedRef.current) {
        setClient(newClient);
        setIsInitialized(true);
      }
    } catch (err) {
      const aiError = err instanceof BharatAIError 
        ? err 
        : new BharatAIError('INITIALIZATION_ERROR', 'Failed to initialize Bharat AI client', err);
      
      if (isMountedRef.current) {
        setError(aiError);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  const updateConfig = useCallback((newConfig: Partial<ClientConfig>) => {
    configRef.current = { ...configRef.current, ...newConfig };
    
    if (client) {
      client.updateConfig(configRef.current);
    }
  }, [client]);

  const getConfig = useCallback(() => {
    return configRef.current;
  }, []);

  const reset = useCallback(() => {
    if (client) {
      client.destroy();
    }
    
    setClient(null);
    setIsInitialized(false);
    setError(null);
    setIsLoading(false);
  }, [client]);

  useEffect(() => {
    if (autoInitialize) {
      initialize();
    }

    return () => {
      isMountedRef.current = false;
      if (client) {
        client.destroy();
      }
    };
  }, [autoInitialize, initialize, client]);

  return {
    client,
    isInitialized,
    isLoading,
    error,
    initialize,
    updateConfig,
    getConfig,
    reset,
  };
}