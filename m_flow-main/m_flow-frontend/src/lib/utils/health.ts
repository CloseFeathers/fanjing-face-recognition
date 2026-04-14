/**
 * Health Check Utility Functions
 * 
 * This file provides helper functions for working with health check data.
 * Used by Setup page and system monitoring components.
 */

import type {
  HealthVerdict,
  ProbeResult,
  ProbeKey,
  ProbeMetadata,
  ProbeDisplayConfig,
  HealthProbes,
} from "@/types";

// ============================================================================
// Verdict Display Configuration
// ============================================================================

/**
 * Maps health verdict to UI display configuration
 * 
 * @param verdict - The health verdict from the API
 * @returns Display configuration for the verdict
 * 
 * @example
 * const config = getVerdictDisplayConfig('up');
 * // { label: 'Connected', color: 'green', icon: 'check', bgVariant: 'success' }
 */
export function getVerdictDisplayConfig(verdict: HealthVerdict): ProbeDisplayConfig {
  switch (verdict) {
    case "up":
      return {
        label: "Connected",
        color: "green",
        icon: "check",
        bgVariant: "success",
      };
    case "warn":
      return {
        label: "Degraded",
        color: "amber",
        icon: "warning",
        bgVariant: "warning",
      };
    case "down":
      return {
        label: "Offline",
        color: "red",
        icon: "error",
        bgVariant: "error",
      };
    default:
      return {
        label: "Unknown",
        color: "gray",
        icon: "unknown",
        bgVariant: "neutral",
      };
  }
}

// ============================================================================
// Probe Metadata
// ============================================================================

/**
 * Metadata for each probe type
 * Provides display names, descriptions, and criticality flags
 */
export const PROBE_METADATA: Record<ProbeKey, ProbeMetadata> = {
  relational_db: {
    displayName: "Relational Database",
    description: "SQLite or Postgres for metadata storage",
    isCritical: true,
  },
  vector_db: {
    displayName: "Vector Database",
    description: "LanceDB, Qdrant, or Milvus for embeddings",
    isCritical: true,
  },
  graph_db: {
    displayName: "Graph Database",
    description: "Kuzu or Neo4j for knowledge graph",
    isCritical: true,
  },
  file_storage: {
    displayName: "File Storage",
    description: "Local filesystem or S3 for documents",
    isCritical: true,
  },
  llm_provider: {
    displayName: "LLM Provider",
    description: "OpenAI, Anthropic, or Ollama for inference",
    isCritical: false,
  },
  embedding_service: {
    displayName: "Embedding Service",
    description: "Text embedding model for vector search",
    isCritical: false,
  },
};

/**
 * Ordered list of probe keys for consistent display
 * Critical services first, then non-critical
 */
export const PROBE_ORDER: ProbeKey[] = [
  // Critical services (databases)
  "relational_db",
  "vector_db",
  "graph_db",
  "file_storage",
  // Non-critical services (AI providers)
  "llm_provider",
  "embedding_service",
];

/**
 * Get metadata for a specific probe
 * 
 * @param probeKey - The probe key
 * @returns ProbeMetadata for the probe
 */
export function getProbeMetadata(probeKey: ProbeKey): ProbeMetadata {
  return PROBE_METADATA[probeKey];
}

// ============================================================================
// Health Status Aggregation
// ============================================================================

/**
 * Aggregate verdict from all probes
 * 
 * Logic:
 * - If any critical probe is 'down' → 'down'
 * - If any probe is 'warn' → 'warn'
 * - Otherwise → 'up'
 * 
 * @param probes - All probe results
 * @returns Aggregated verdict
 * 
 * @example
 * const overall = aggregateVerdict(data.probes);
 * if (overall === 'down') {
 *   showCriticalAlert();
 * }
 */
export function aggregateVerdict(probes: HealthProbes): HealthVerdict {
  const criticalKeys: ProbeKey[] = ["relational_db", "vector_db", "graph_db", "file_storage"];
  
  // Check if any critical service is down
  const anyCriticalDown = criticalKeys.some(
    (key) => probes[key].verdict === "down"
  );
  
  if (anyCriticalDown) {
    return "down";
  }
  
  // Check if any service has a warning
  const anyWarn = Object.values(probes).some(
    (probe) => probe.verdict === "warn"
  );
  
  if (anyWarn) {
    return "warn";
  }
  
  return "up";
}

/**
 * Count probes by verdict
 * 
 * @param probes - All probe results
 * @returns Object with counts for each verdict
 * 
 * @example
 * const counts = countByVerdict(data.probes);
 * console.log(`${counts.up}/${counts.total} services healthy`);
 */
export function countByVerdict(probes: HealthProbes): {
  up: number;
  warn: number;
  down: number;
  total: number;
} {
  const values = Object.values(probes);
  return {
    up: values.filter((p) => p.verdict === "up").length,
    warn: values.filter((p) => p.verdict === "warn").length,
    down: values.filter((p) => p.verdict === "down").length,
    total: values.length,
  };
}

// ============================================================================
// Latency Helpers
// ============================================================================

/**
 * Calculate total latency across all probes
 * 
 * @param probes - All probe results
 * @returns Total latency in milliseconds
 */
export function getTotalLatency(probes: HealthProbes): number {
  return Object.values(probes).reduce(
    (sum, probe) => sum + probe.latency_ms,
    0
  );
}

/**
 * Get the slowest probe
 * 
 * @param probes - All probe results
 * @returns The probe key and result with highest latency
 */
export function getSlowestProbe(probes: HealthProbes): {
  key: ProbeKey;
  probe: ProbeResult;
} | null {
  let maxLatency = -1;
  let slowestKey: ProbeKey | null = null;
  
  for (const key of PROBE_ORDER) {
    if (probes[key].latency_ms > maxLatency) {
      maxLatency = probes[key].latency_ms;
      slowestKey = key;
    }
  }
  
  if (slowestKey === null) return null;
  
  return {
    key: slowestKey,
    probe: probes[slowestKey],
  };
}

// ============================================================================
// Formatting Helpers
// ============================================================================

/**
 * Format latency for display
 * 
 * @param ms - Latency in milliseconds
 * @returns Formatted string (e.g., "23ms", "1.2s")
 */
export function formatLatency(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Format uptime for display
 * 
 * @param seconds - Uptime in seconds
 * @returns Formatted string (e.g., "5m 30s", "2h 15m", "3d 4h")
 */
export function formatUptime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`;
  }
  
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  }
  
  if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }
  
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
}

/**
 * Format ISO timestamp to local time
 * 
 * @param isoString - ISO 8601 timestamp
 * @returns Formatted local time (e.g., "12:54:32")
 */
export function formatCheckedAt(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString();
  } catch {
    return "-";
  }
}

// ============================================================================
// CSS Variable Mapping
// ============================================================================

/**
 * Get CSS variable names for verdict colors
 * 
 * @param verdict - The health verdict
 * @returns Object with CSS variable names
 * 
 * @example
 * const vars = getVerdictCSSVars('up');
 * // { text: 'var(--status-up)', bg: 'var(--bg-up)' }
 */
export function getVerdictCSSVars(verdict: HealthVerdict): {
  text: string;
  bg: string;
  border: string;
} {
  switch (verdict) {
    case "up":
      return {
        text: "var(--status-up, #22c55e)",
        bg: "var(--bg-up, rgba(34, 197, 94, 0.08))",
        border: "var(--status-up, #22c55e)",
      };
    case "warn":
      return {
        text: "var(--status-warn, #f59e0b)",
        bg: "var(--bg-warn, rgba(245, 158, 11, 0.08))",
        border: "var(--status-warn, #f59e0b)",
      };
    case "down":
      return {
        text: "var(--status-down, #ef4444)",
        bg: "var(--bg-down, rgba(239, 68, 68, 0.08))",
        border: "var(--status-down, #ef4444)",
      };
    default:
      return {
        text: "var(--text-muted, #6b7280)",
        bg: "var(--bg-surface, #18181b)",
        border: "var(--border-subtle, #27272a)",
      };
  }
}
