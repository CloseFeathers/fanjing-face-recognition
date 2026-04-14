"use client";

/**
 * HealthBar Component
 *
 * Compact inline health status indicator for the Dashboard.
 * Shows connection status for all services in a single row.
 *
 * Features:
 * - Real-time status updates
 * - Compact pill design
 * - Click to expand details
 * - Average latency display
 */

import React from "react";
import { cn } from "@/lib/utils";
import { useDetailedHealth } from "@/hooks/use-api";
import { Activity, RefreshCw } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface ServiceStatus {
  key: string;
  label: string;
  verdict: "up" | "warn" | "down";
  latency?: number;
}

// ============================================================================
// Service Configuration
// ============================================================================

const SERVICE_CONFIG: Record<string, { label: string; priority: number }> = {
  llm_provider: { label: "LLM", priority: 1 },
  graph_database: { label: "Neo4j", priority: 2 },
  embedding_service: { label: "Embed", priority: 3 },
  vector_database: { label: "Vector", priority: 4 },
  blob_storage: { label: "Storage", priority: 5 },
  api: { label: "API", priority: 6 },
};

// ============================================================================
// Status Dot Component
// ============================================================================

interface StatusDotProps {
  verdict: "up" | "warn" | "down";
  size?: "sm" | "md";
}

function StatusDot({ verdict, size = "sm" }: StatusDotProps) {
  return (
    <span
      className={cn(
        "rounded-full",
        size === "sm" ? "w-1.5 h-1.5" : "w-2 h-2",
        verdict === "up" && "bg-emerald-400",
        verdict === "warn" && "bg-amber-400",
        verdict === "down" && "bg-red-400"
      )}
    />
  );
}

// ============================================================================
// Service Pill Component
// ============================================================================

interface ServicePillProps {
  service: ServiceStatus;
}

function ServicePill({ service }: ServicePillProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-1.5 px-2 py-1 rounded-md",
        "text-[11px] font-medium",
        "bg-[var(--bg-elevated)] border border-[var(--border-subtle)]",
        "transition-colors hover:border-[var(--border-default)]"
      )}
    >
      <StatusDot verdict={service.verdict} />
      <span className="text-[var(--text-secondary)]">{service.label}</span>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function HealthBar() {
  const { data, isLoading, isRefetching, refetch } = useDetailedHealth({
    refetchInterval: 60000,
  });

  // Parse services from health data
  const services: ServiceStatus[] = React.useMemo(() => {
    if (!data?.probes) return [];

    return Object.entries(data.probes)
      .filter(([key]) => SERVICE_CONFIG[key])
      .map(([key, probe]) => ({
        key,
        label: SERVICE_CONFIG[key].label,
        verdict: probe.verdict,
        latency: probe.latency_ms,
        priority: SERVICE_CONFIG[key].priority,
      }))
      .sort((a, b) => a.priority - b.priority);
  }, [data]);

  // Calculate overall status
  const overallStatus = React.useMemo(() => {
    if (!services.length) return "down";
    if (services.every((s) => s.verdict === "up")) return "up";
    if (services.some((s) => s.verdict === "down")) return "down";
    return "warn";
  }, [services]);

  // Calculate average latency
  const avgLatency = React.useMemo(() => {
    const latencies = services
      .filter((s) => s.latency && s.latency > 0)
      .map((s) => s.latency!);
    if (!latencies.length) return 0;
    return Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length);
  }, [services]);

  // Count by status
  const statusCounts = React.useMemo(() => {
    return {
      up: services.filter((s) => s.verdict === "up").length,
      warn: services.filter((s) => s.verdict === "warn").length,
      down: services.filter((s) => s.verdict === "down").length,
      total: services.length,
    };
  }, [services]);

  // Loading state
  if (isLoading && !data) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)]">
        <div className="w-4 h-4 rounded-full bg-[var(--bg-hover)] animate-pulse" />
        <div className="flex-1 h-4 rounded bg-[var(--bg-hover)] animate-pulse" />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex items-center justify-between gap-4 p-3 rounded-lg",
        "border transition-colors",
        overallStatus === "up" && "bg-emerald-950/5 border-emerald-900/20",
        overallStatus === "warn" && "bg-amber-950/5 border-amber-900/20",
        overallStatus === "down" && "bg-red-950/5 border-red-900/20"
      )}
    >
      {/* Left: Status indicator and services */}
      <div className="flex items-center gap-4">
        {/* Overall status */}
        <div className="flex items-center gap-2">
          <Activity
            size={14}
            className={cn(
              overallStatus === "up" && "text-emerald-400",
              overallStatus === "warn" && "text-amber-400",
              overallStatus === "down" && "text-red-400"
            )}
          />
          <span
            className={cn(
              "text-[12px] font-medium",
              overallStatus === "up" && "text-emerald-400",
              overallStatus === "warn" && "text-amber-400",
              overallStatus === "down" && "text-red-400"
            )}
          >
            {statusCounts.up}/{statusCounts.total}
          </span>
        </div>

        {/* Divider */}
        <div className="w-px h-4 bg-[var(--border-subtle)]" />

        {/* Service pills */}
        <div className="flex items-center gap-2 flex-wrap">
          {services.map((service) => (
            <ServicePill key={service.key} service={service} />
          ))}
        </div>
      </div>

      {/* Right: Latency and refresh */}
      <div className="flex items-center gap-3">
        {avgLatency > 0 && (
          <span className="text-[11px] text-[var(--text-muted)]">
            {avgLatency}ms avg
          </span>
        )}
        <button
          onClick={() => refetch()}
          disabled={isRefetching}
          className={cn(
            "p-1 rounded text-[var(--text-muted)]",
            "hover:text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]",
            "transition-colors disabled:opacity-50"
          )}
          aria-label="Refresh status"
        >
          <RefreshCw
            size={12}
            className={cn(isRefetching && "animate-spin")}
          />
        </button>
      </div>
    </div>
  );
}

HealthBar.displayName = "HealthBar";

export default HealthBar;
