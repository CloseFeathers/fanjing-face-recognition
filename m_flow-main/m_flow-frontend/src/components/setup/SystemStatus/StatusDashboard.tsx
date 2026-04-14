"use client";

/**
 * StatusDashboard Component
 * 
 * Displays the health status of all system components in a responsive grid.
 * Uses the /health/detailed endpoint to fetch real-time status data.
 * 
 * Features:
 * - Auto-refresh every 30 seconds
 * - Manual refresh button
 * - Responsive 2x3 grid layout
 * - Loading skeletons
 * - Error state handling
 * 
 * @example
 * // Basic usage
 * <StatusDashboard />
 * 
 * @example
 * // With custom refresh interval
 * <StatusDashboard autoRefreshInterval={10000} />
 * 
 * @example
 * // With refresh callback
 * <StatusDashboard onRefresh={() => console.log('Refreshed!')} />
 */

import React, { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { useDetailedHealth } from "@/hooks/use-api";
import {
  PROBE_ORDER,
  aggregateVerdict,
  countByVerdict,
  formatUptime,
  formatCheckedAt,
  getVerdictDisplayConfig,
} from "@/lib/utils/health";
import { StatusCard, StatusCardSkeleton, StatusDetailModal } from "./StatusCard";
import { StatusIndicator } from "./StatusIndicator";
import { RefreshCw, AlertCircle, Clock, Server } from "lucide-react";
import type { ProbeKey } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface StatusDashboardProps {
  /** Optional callback when refresh is clicked */
  onRefresh?: () => void;
  /** Whether to enable auto-refresh */
  autoRefresh?: boolean;
  /** Auto-refresh interval in milliseconds */
  autoRefreshInterval?: number;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Component
// ============================================================================

export function StatusDashboard({
  onRefresh,
  autoRefresh = true,
  autoRefreshInterval = 30000,
  className,
}: StatusDashboardProps) {
  // Selected service for detail modal
  const [selectedService, setSelectedService] = useState<ProbeKey | null>(null);

  // Fetch health data
  const {
    data,
    isLoading,
    isRefetching,
    error,
    refetch,
    dataUpdatedAt,
  } = useDetailedHealth({
    refetchInterval: autoRefresh ? autoRefreshInterval : false,
    enabled: true,
  });

  // Handle refresh
  const handleRefresh = useCallback(() => {
    refetch();
    onRefresh?.();
  }, [refetch, onRefresh]);

  // Handle card click
  const handleCardClick = useCallback((key: ProbeKey) => {
    setSelectedService(key);
  }, []);

  // Handle modal close
  const handleModalClose = useCallback(() => {
    setSelectedService(null);
  }, []);

  // Calculate summary stats
  const summary = data
    ? {
        verdict: aggregateVerdict(data.probes),
        counts: countByVerdict(data.probes),
        uptime: formatUptime(data.alive_seconds),
        lastChecked: formatCheckedAt(data.checked_at),
        version: data.build,
      }
    : null;

  const verdictConfig = summary
    ? getVerdictDisplayConfig(summary.verdict)
    : null;

  // ============================================================================
  // Render: Error State
  // ============================================================================

  if (error && !data) {
    return (
      <div className={cn("space-y-2", className)}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-[14px] font-semibold text-zinc-100">System Status</h2>
          <button
            onClick={handleRefresh}
            className="flex items-center gap-1.5 px-2 py-1 text-[11px] text-zinc-400 hover:text-zinc-200 border border-zinc-700 rounded hover:border-zinc-600 transition-colors"
          >
            <RefreshCw size={10} />
            Retry
          </button>
        </div>

        {/* Error Message - Compact */}
        <div className="flex items-center gap-3 py-3 px-3 border border-red-900/50 bg-red-950/10 rounded-lg">
          <AlertCircle size={18} className="text-red-400 shrink-0" />
          <div>
            <p className="text-[12px] text-zinc-200">
              Unable to connect to backend
            </p>
            <p className="text-[10px] text-zinc-500">
              Ensure server is running at localhost:8000
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================================
  // Render: Loading State
  // ============================================================================

  if (isLoading && !data) {
    return (
      <div className={cn("space-y-2", className)}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-[14px] font-semibold text-zinc-100">System Status</h2>
          <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
            <RefreshCw size={10} className="animate-spin" />
            Loading...
          </div>
        </div>

        {/* Loading Skeletons */}
        <div className="grid grid-cols-2 gap-2">
          {PROBE_ORDER.map((key) => (
            <StatusCardSkeleton key={key} />
          ))}
        </div>
      </div>
    );
  }

  // ============================================================================
  // Render: Success State
  // ============================================================================

  return (
    <div className={cn("space-y-2", className)}>
      {/* Header Row - Compact */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-[14px] font-semibold text-zinc-100">System Status</h2>
          {summary && (
            <div className="flex items-center gap-1.5">
              <StatusIndicator verdict={summary.verdict} size="sm" />
              <span
                className={cn(
                  "text-[11px] font-medium",
                  summary.verdict === "up" && "text-emerald-400",
                  summary.verdict === "warn" && "text-amber-400",
                  summary.verdict === "down" && "text-red-400"
                )}
              >
                {summary.counts.up}/{summary.counts.total} healthy
              </span>
            </div>
          )}
        </div>

        {/* Refresh Button - Compact */}
        <button
          onClick={handleRefresh}
          disabled={isRefetching}
          className={cn(
            "flex items-center gap-1.5 px-2 py-1 text-[11px]",
            "text-zinc-400 hover:text-zinc-200",
            "border border-zinc-700 rounded hover:border-zinc-600",
            "transition-colors disabled:opacity-50"
          )}
        >
          <RefreshCw
            size={10}
            className={cn(isRefetching && "animate-spin")}
          />
          Refresh
        </button>
      </div>

      {/* Overall Status Banner - Compact */}
      {summary && (
        <div
          className={cn(
            "flex items-center justify-between px-3 py-2.5 rounded-lg border",
            summary.verdict === "up" &&
              "border-emerald-900/30 bg-emerald-950/10",
            summary.verdict === "warn" &&
              "border-amber-900/30 bg-amber-950/10",
            summary.verdict === "down" && "border-red-900/30 bg-red-950/10"
          )}
        >
          {/* Left: Status */}
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex items-center justify-center w-8 h-8 rounded-lg",
                summary.verdict === "up" && "bg-emerald-500/10",
                summary.verdict === "warn" && "bg-amber-500/10",
                summary.verdict === "down" && "bg-red-500/10"
              )}
            >
              <Server
                size={16}
                className={cn(
                  summary.verdict === "up" && "text-emerald-400",
                  summary.verdict === "warn" && "text-amber-400",
                  summary.verdict === "down" && "text-red-400"
                )}
              />
            </div>
            <div>
              <p
                className={cn(
                  "text-[12px] font-medium leading-tight",
                  summary.verdict === "up" && "text-emerald-400",
                  summary.verdict === "warn" && "text-amber-400",
                  summary.verdict === "down" && "text-red-400"
                )}
              >
                {summary.verdict === "up" && "All systems operational"}
                {summary.verdict === "warn" && "Partial degradation"}
                {summary.verdict === "down" && "System outage"}
              </p>
              <p className="text-[10px] text-zinc-500">
                {summary.counts.up} healthy
                {summary.counts.warn > 0 && ` · ${summary.counts.warn} degraded`}
                {summary.counts.down > 0 && ` · ${summary.counts.down} offline`}
              </p>
            </div>
          </div>

          {/* Right: Meta info - inline compact */}
          <div className="hidden sm:flex items-center gap-3 text-[10px] text-zinc-500">
            <span>Updated {summary.lastChecked}</span>
            <span>·</span>
            <span>v{summary.version}</span>
            <span>·</span>
            <span>Uptime: {summary.uptime}</span>
          </div>
        </div>
      )}

      {/* Service Cards Grid - Compact 2-column layout */}
      {data && (
        <div className="grid grid-cols-2 gap-2">
          {PROBE_ORDER.map((key) => (
            <StatusCard
              key={key}
              serviceKey={key as ProbeKey}
              probe={data.probes[key as ProbeKey]}
              isTesting={isRefetching}
              onClick={() => handleCardClick(key as ProbeKey)}
            />
          ))}
        </div>
      )}

      {/* Footer: Auto-refresh indicator */}
      {autoRefresh && (
        <p className="text-[9px] text-zinc-600 text-center mt-2">
          Auto-refreshes every {autoRefreshInterval / 1000} seconds
        </p>
      )}

      {/* Detail Modal */}
      {data && selectedService && (
        <StatusDetailModal
          serviceKey={selectedService}
          probe={data.probes[selectedService]}
          isOpen={!!selectedService}
          onClose={handleModalClose}
        />
      )}
    </div>
  );
}

// ============================================================================
// Display Name for DevTools
// ============================================================================

StatusDashboard.displayName = "StatusDashboard";

// ============================================================================
// Default Export
// ============================================================================

export default StatusDashboard;
