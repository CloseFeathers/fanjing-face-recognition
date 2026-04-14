/**
 * WebSocket configuration and utilities
 * 
 * Used for building WebSocket connection URLs, supporting progress tracking etc.
 */

import { config } from "@/lib/config";

// ============================================================================
// URL Builders
// ============================================================================

/**
 * Get WebSocket base URL
 * 
 * Prefers NEXT_PUBLIC_WS_URL environment variable
 * Otherwise derives from NEXT_PUBLIC_API_URL (http -> ws, https -> wss)
 */
export function getWebSocketBaseUrl(): string {
  return config.WS_BASE_URL;
}

/**
 * Build Memorize progress tracking WebSocket URL
 * 
 * @endpoint WS /api/v1/memorize/subscribe/{workflow_run_id}
 */
export function getMemorizeProgressUrl(pipelineRunId: string): string {
  const token = typeof window !== "undefined" ? localStorage.getItem("mflow_token") : null;
  const params = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${config.WS_BASE_URL}/api/v1/memorize/subscribe/${pipelineRunId}${params}`;
}

// ============================================================================
// Connection State
// ============================================================================

export type WebSocketConnectionState = 
  | "connecting"
  | "connected"
  | "disconnected"
  | "error"
  | "auth_failed";

// ============================================================================
// Error Codes
// ============================================================================

export const WS_CLOSE_CODES = {
  NORMAL: 1000,
  GOING_AWAY: 1001,
  PROTOCOL_ERROR: 1002,
  UNSUPPORTED: 1003,
  NO_STATUS: 1005,
  ABNORMAL: 1006,
  INVALID_PAYLOAD: 1007,
  POLICY_VIOLATION: 1008,  // Auth failure
  MESSAGE_TOO_BIG: 1009,
  EXTENSION_REQUIRED: 1010,
  INTERNAL_ERROR: 1011,
  SERVICE_RESTART: 1012,
  TRY_AGAIN_LATER: 1013,
  BAD_GATEWAY: 1014,
  TLS_HANDSHAKE: 1015,
} as const;
