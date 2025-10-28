/**
 * Hook for offline mode functionality
 */

import { useState, useEffect, useCallback } from 'react';
import { BharatAIClient } from '../client/BharatAIClient';
import NetInfo from '@react-native-netinfo/netinfo';
import { BharatAIError } from '../errors/BharatAIError';

interface UseOfflineModeProps {
  /** Bharat AI client instance */
  client: BharatAIClient;
  /** Enable offline mode */
  enabled?: boolean;
  /** Auto-sync when online */
  autoSync?: boolean;
  /** Sync interval in milliseconds */
  syncInterval?: number;
}

interface UseOfflineModeReturn {
  /** Network connectivity status */
  isOnline: boolean;
  /** Offline mode status */
  isOfflineMode: boolean;
  /** Sync in progress */
  isSyncing: boolean;
  ** Pending operations count */
  pendingOperations: number;
  ** Last sync timestamp */
  lastSync: Date | null;
  ** Enable offline mode */
  enableOfflineMode: () => Promise<void>;
  ** Disable offline mode */
  disableOfflineMode: () => Promise<void>;
  ** Force sync now */
  syncNow: () => Promise<void>;
  ** Clear offline cache */
  clearCache: () => Promise<void>;
  ** Get offline statistics */
  getOfflineStats: () => Promise<{
    cacheSize: number;
    pendingOps: number;
    lastSync: Date | null;
  }>;
}

export function useOfflineMode({
  client,
  enabled = true,
  autoSync = true,
  syncInterval = 30000, // 30 seconds
}: UseOfflineModeProps): UseOfflineModeReturn {
  const [isOnline, setIsOnline] = useState(true);
  const [isOfflineMode, setIsOfflineMode] = useState(enabled);
  const [isSyncing, setIsSyncing] = useState(false);
  const [pendingOperations, setPendingOperations] = useState(0);
  const [lastSync, setLastSync] = useState<Date | null>(null);

  // Check network connectivity
  const checkConnectivity = useCallback(async () => {
    try {
      const netInfo = await NetInfo.fetch();
      const online = netInfo.isConnected === true && netInfo.isInternetReachable !== false;
      setIsOnline(online);
      return online;
    } catch (error) {
      console.error('Error checking connectivity:', error);
      setIsOnline(false);
      return false;
    }
  }, []);

  // Enable offline mode
  const enableOfflineMode = useCallback(async () => {
    try {
      const config = client.getConfig();
      const newConfig = { ...config, enableOfflineMode: true };
      client.updateConfig(newConfig);
      setIsOfflineMode(true);
    } catch (error) {
      throw new BharatAIError('OFFLINE_MODE_ERROR', 'Failed to enable offline mode', error);
    }
  }, [client]);

  // Disable offline mode
  const disableOfflineMode = useCallback(async () => {
    try {
      const config = client.getConfig();
      const newConfig = { ...config, enableOfflineMode: false };
      client.updateConfig(newConfig);
      setIsOfflineMode(false);
    } catch (error) {
      throw new BharatAIError('OFFLINE_MODE_ERROR', 'Failed to disable offline mode', error);
    }
  }, [client]);

  // Sync offline data
  const syncNow = useCallback(async () => {
    if (!isOnline || isSyncing) return;

    setIsSyncing(true);

    try {
      // Simulate sync process
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setLastSync(new Date());
      setPendingOperations(0);
    } catch (error) {
      throw new BharatAIError('SYNC_ERROR', 'Failed to sync offline data', error);
    } finally {
      setIsSyncing(false);
    }
  }, [isOnline, isSyncing]);

  // Clear offline cache
  const clearCache = useCallback(async () => {
    try {
      await client.clearCache();
      setPendingOperations(0);
    } catch (error) {
      throw new BharatAIError('CACHE_ERROR', 'Failed to clear offline cache', error);
    }
  }, [client]);

  // Get offline statistics
  const getOfflineStats = useCallback(async () => {
    try {
      const stats = await client.getCacheStats();
      return {
        cacheSize: stats.approximateSizeBytes || 0,
        pendingOps: pendingOperations,
        lastSync,
      };
    } catch (error) {
      throw new BharatAIError('STATS_ERROR', 'Failed to get offline statistics', error);
    }
  }, [client, pendingOperations, lastSync]);

  // Auto-sync when coming online
  useEffect(() => {
    if (!autoSync || !isOfflineMode) return;

    const handleConnectivityChange = async () => {
      const online = await checkConnectivity();
      if (online && pendingOperations > 0) {
        await syncNow();
      }
    };

    const unsubscribe = NetInfo.addEventListener(handleConnectivityChange);
    return unsubscribe;
  }, [autoSync, isOfflineMode, pendingOperations, checkConnectivity, syncNow]);

  // Periodic sync
  useEffect(() => {
    if (!autoSync || !isOfflineMode || !isOnline) return;

    const interval = setInterval(async () => {
      if (pendingOperations > 0) {
        await syncNow();
      }
    }, syncInterval);

    return () => clearInterval(interval);
  }, [autoSync, isOfflineMode, isOnline, pendingOperations, syncInterval, syncNow]);

  // Initial connectivity check
  useEffect(() => {
    checkConnectivity();
  }, [checkConnectivity]);

  return {
    isOnline,
    isOfflineMode,
    isSyncing,
    pendingOperations,
    lastSync,
    enableOfflineMode,
    disableOfflineMode,
    syncNow,
    clearCache,
    getOfflineStats,
  };
}