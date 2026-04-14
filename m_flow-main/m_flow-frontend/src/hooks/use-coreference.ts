"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api/client";
import { CorefConfig, CorefConfigUpdate, CorefStats } from "@/types";
import { STORAGE_KEYS } from "@/lib/config";

function generateSessionId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

// ============================================================================
// Query Keys
// ============================================================================

export const corefQueryKeys = {
  config: ["coreference", "config"] as const,
  stats: ["coreference", "stats"] as const,
};

// ============================================================================
// useCorefConfig
// ============================================================================

export interface UseCorefConfigOptions {
  enabled?: boolean;
}

/**
 * Hook to manage coreference configuration
 *
 * @example
 * const { config, isLoading, updateConfig } = useCorefConfig();
 *
 * // Toggle enabled state
 * updateConfig({ enabled: !config?.enabled });
 */
export function useCorefConfig(options: UseCorefConfigOptions = {}) {
  const { enabled = true } = options;
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: corefQueryKeys.config,
    queryFn: () => apiClient.getCorefConfig(),
    enabled,
    staleTime: 30000,
  });

  const mutation = useMutation({
    mutationFn: (update: CorefConfigUpdate) => apiClient.updateCorefConfig(update),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: corefQueryKeys.config });
      queryClient.invalidateQueries({ queryKey: corefQueryKeys.stats });
    },
  });

  const updateConfig = useCallback(
    (update: CorefConfigUpdate) => {
      mutation.mutate(update);
    },
    [mutation]
  );

  return {
    config: query.data,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,
    updateConfig,
    isUpdating: mutation.isPending,
    updateError: mutation.error,
    refetch: query.refetch,
  };
}

// ============================================================================
// useCorefStats
// ============================================================================

export interface UseCorefStatsOptions {
  enabled?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  includeSessions?: boolean;
  sessionLimit?: number;
}

/**
 * Hook to monitor coreference module statistics
 *
 * @example
 * const { stats, isLoading } = useCorefStats({ autoRefresh: true });
 *
 * console.log(`Active sessions: ${stats?.active_sessions}`);
 */
export function useCorefStats(options: UseCorefStatsOptions = {}) {
  const {
    enabled = true,
    autoRefresh = false,
    refreshInterval = 30000,
    includeSessions = false,
    sessionLimit = 100,
  } = options;

  const query = useQuery({
    queryKey: [...corefQueryKeys.stats, { includeSessions, sessionLimit }],
    queryFn: () => apiClient.getCorefStats(includeSessions, sessionLimit),
    enabled,
    staleTime: 10000,
    refetchInterval: autoRefresh ? refreshInterval : false,
  });

  return {
    stats: query.data,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,
    refetch: query.refetch,
  };
}

// ============================================================================
// useCorefSession
// ============================================================================

export interface UseCorefSessionOptions {
  userId?: string;
  persistKey?: string;
}

/**
 * Hook to manage a coreference session ID
 *
 * Generates a stable session ID per component instance,
 * with optional localStorage persistence.
 *
 * @example
 * const { sessionId, resetSession } = useCorefSession();
 *
 * // Use in search
 * search({ query, coref_session_id: sessionId, coref_enabled: true });
 *
 * // Reset context for new conversation
 * resetSession();
 */
export function useCorefSession(options: UseCorefSessionOptions = {}) {
  const { persistKey } = options;
  const queryClient = useQueryClient();

  // Generate initial session ID
  const getInitialSessionId = useCallback(() => {
    if (persistKey && typeof window !== "undefined") {
      const saved = localStorage.getItem(`${STORAGE_KEYS.COREF_SESSION_PREFIX}${persistKey}`);
      if (saved) return saved;
    }
    return generateSessionId();
  }, [persistKey]);

  const sessionIdRef = useRef<string>(getInitialSessionId());

  // Session ID is the ref's current value
  const sessionId = sessionIdRef.current;

  // Persist to localStorage if key provided
  useEffect(() => {
    if (persistKey && typeof window !== "undefined") {
      localStorage.setItem(`${STORAGE_KEYS.COREF_SESSION_PREFIX}${persistKey}`, sessionId);
    }
  }, [sessionId, persistKey]);

  const resetMutation = useMutation({
    mutationFn: () => apiClient.resetCorefSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: corefQueryKeys.stats });
    },
  });

  const resetSession = useCallback(() => {
    resetMutation.mutate();
  }, [resetMutation]);

  const generateNewSession = useCallback(() => {
    const newId = generateSessionId();
    sessionIdRef.current = newId;
    if (persistKey && typeof window !== "undefined") {
      localStorage.setItem(`${STORAGE_KEYS.COREF_SESSION_PREFIX}${persistKey}`, newId);
    }
    return newId;
  }, [persistKey]);

  return {
    sessionId,
    resetSession,
    generateNewSession,
    isResetting: resetMutation.isPending,
    resetError: resetMutation.error,
  };
}
