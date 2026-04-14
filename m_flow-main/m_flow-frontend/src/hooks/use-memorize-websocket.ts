"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { getMemorizeProgressUrl, WS_CLOSE_CODES } from "@/lib/api/websocket";
import type { RunStatus, WebSocketProgress } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface UseMemorizeWebSocketOptions {
  /** Max reconnection attempts */
  maxReconnects?: number;
  /** Connection timeout (ms) */
  connectionTimeout?: number;
  /** Auto reconnect (default true) */
  autoReconnect?: boolean;
}

export interface UseMemorizeWebSocketResult {
  /** Current progress info */
  progress: WebSocketProgress | null;
  /** Whether connected */
  isConnected: boolean;
  /** Connection state */
  connectionState: "idle" | "connecting" | "connected" | "disconnected" | "error" | "auth_failed";
  /** Error message */
  error: Error | null;
  /** Whether completed */
  isCompleted: boolean;
  /** Whether error occurred */
  isError: boolean;
  /** Whether processing */
  isProcessing: boolean;
  /** Manual disconnect */
  disconnect: () => void;
  /** Reconnect */
  reconnect: () => void;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_MAX_RECONNECTS = 5;
const DEFAULT_CONNECTION_TIMEOUT = 10000; // 10 seconds

// ============================================================================
// Hook
// ============================================================================

/**
 * WebSocket hook for tracking memorize pipeline progress
 * 
 * @param pipelineRunId - Pipeline run ID from memorize() response
 * @param options - Connection options
 * @returns Progress tracking state and controls
 * 
 * @example
 * ```tsx
 * const { progress, isConnected, isCompleted, error } = useMemorizeWebSocket(pipelineRunId);
 * 
 * if (isCompleted) {
 *   toast.success("Processing complete!");
 * }
 * 
 * if (error) {
 *   toast.error(error.message);
 * }
 * ```
 */
export function useMemorizeWebSocket(
  pipelineRunId: string | null,
  options: UseMemorizeWebSocketOptions = {}
): UseMemorizeWebSocketResult {
  const {
    maxReconnects = DEFAULT_MAX_RECONNECTS,
    connectionTimeout = DEFAULT_CONNECTION_TIMEOUT,
    autoReconnect = true,
  } = options;

  // State
  const [progress, setProgress] = useState<WebSocketProgress | null>(null);
  const [connectionState, setConnectionState] = useState<UseMemorizeWebSocketResult["connectionState"]>("idle");
  const [error, setError] = useState<Error | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef(true);

  // Computed states
  const isConnected = connectionState === "connected";
  const isCompleted = progress?.status === "RunCompleted" || 
                      progress?.status === "RunAlreadyCompleted";
  const isError = progress?.status === "RunFailed";
  const isProcessing = connectionState === "connecting" || 
                       (isConnected && !isCompleted && !isError);

  // Cleanup function
  const cleanup = useCallback(() => {
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close(WS_CLOSE_CODES.NORMAL);
      wsRef.current = null;
    }
  }, []);

  // Disconnect function
  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    cleanup();
    setConnectionState("disconnected");
  }, [cleanup]);

  // Connect function
  const connect = useCallback(() => {
    if (!pipelineRunId) return;

    // Cleanup previous connection
    cleanup();

    shouldReconnectRef.current = autoReconnect;
    setConnectionState("connecting");
    setError(null);

    const url = getMemorizeProgressUrl(pipelineRunId);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    // Connection timeout
    connectionTimeoutRef.current = setTimeout(() => {
      if (ws.readyState === WebSocket.CONNECTING) {
        ws.close();
        setError(new Error("Connection timeout"));
        setConnectionState("error");
      }
    }, connectionTimeout);

    ws.onopen = () => {
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
        connectionTimeoutRef.current = null;
      }
      reconnectAttemptsRef.current = 0;
      setConnectionState("connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketProgress;
        setProgress(data);

        // Auto-disconnect on completion
        if (data.status === "RunCompleted" || 
            data.status === "RunAlreadyCompleted" ||
            data.status === "RunFailed") {
          shouldReconnectRef.current = false;
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onerror = () => {
      setError(new Error("WebSocket connection error"));
    };

    ws.onclose = (event) => {
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
        connectionTimeoutRef.current = null;
      }

      wsRef.current = null;

      // Auth failure - don't retry
      if (event.code === WS_CLOSE_CODES.POLICY_VIOLATION) {
        setConnectionState("auth_failed");
        setError(new Error("Authentication failed. Please login again."));
        return;
      }

      // Normal closure - don't retry
      if (event.code === WS_CLOSE_CODES.NORMAL) {
        setConnectionState("disconnected");
        return;
      }

      // Retry with exponential backoff
      if (shouldReconnectRef.current && reconnectAttemptsRef.current < maxReconnects) {
        const delay = Math.min(Math.pow(2, reconnectAttemptsRef.current) * 1000, 30000);
        reconnectAttemptsRef.current++;
        
        setConnectionState("connecting");
        reconnectTimeoutRef.current = setTimeout(connect, delay);
      } else if (reconnectAttemptsRef.current >= maxReconnects) {
        setConnectionState("error");
        setError(new Error("Max reconnection attempts reached"));
      } else {
        setConnectionState("disconnected");
      }
    };
  }, [pipelineRunId, autoReconnect, connectionTimeout, maxReconnects, cleanup]);

  // Reconnect function
  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    shouldReconnectRef.current = true;
    connect();
  }, [connect]);

  // Effect: Connect when pipelineRunId changes
  useEffect(() => {
    if (pipelineRunId) {
      connect();
    } else {
      cleanup();
      setConnectionState("idle");
      setProgress(null);
      setError(null);
    }

    return cleanup;
  }, [pipelineRunId, connect, cleanup]);

  return {
    progress,
    isConnected,
    connectionState,
    error,
    isCompleted,
    isError,
    isProcessing,
    disconnect,
    reconnect,
  };
}

// ============================================================================
// Helper: Extract pipeline run IDs from memorize response
// ============================================================================

import type { MemorizeResponse } from "@/types";

/**
 * Extract pipeline run IDs from memorize() response
 */
export function extractPipelineRunIds(response: MemorizeResponse): string[] {
  return Object.values(response)
    .map((info) => info?.workflow_run_id)
    .filter((id): id is string => !!id);
}

/**
 * Check if all pipelines are completed
 */
export function areAllPipelinesCompleted(response: MemorizeResponse): boolean {
  const completedStatuses: RunStatus[] = [
    "RunCompleted",
    "RunAlreadyCompleted",
  ];
  
  return Object.values(response).every(
    (info) => completedStatuses.includes(info.status)
  );
}

/**
 * Check if any pipeline has error
 */
export function hasAnyPipelineError(response: MemorizeResponse): boolean {
  return Object.values(response).some(
    (info) => info.status === "RunFailed"
  );
}
