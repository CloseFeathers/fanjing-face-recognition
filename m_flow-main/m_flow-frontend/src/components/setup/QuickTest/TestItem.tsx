"use client";

/**
 * TestItem Component
 *
 * Displays the status and result of a single connectivity test.
 * Shows service name, status indicator, duration, and result message.
 *
 * Features:
 * - Visual status indicator (idle, running, passed, failed, skipped)
 * - Duration display
 * - Backend/provider information
 * - Expandable details
 * - Retry button for failed tests
 *
 * @example
 * <TestItem
 *   result={testResult}
 *   onRetry={() => runSingleTest(testResult.id)}
 *   isRunning={false}
 * />
 */

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Circle,
  SkipForward,
  RotateCcw,
  ChevronDown,
  Server,
  Clock,
  AlertCircle,
} from "lucide-react";
import type { TestResult, TestStatus } from "@/types/setup";
import type { ProbeKey } from "@/types";
import { PROBE_METADATA } from "@/lib/utils/health";

// ============================================================================
// Types
// ============================================================================

export interface TestItemProps {
  /** Test result data */
  result: TestResult;
  /** Retry handler for failed tests */
  onRetry?: () => void;
  /** Whether this specific test is currently running */
  isRunning?: boolean;
  /** Show expanded details by default */
  expandedByDefault?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Status Icon Component
// ============================================================================

interface StatusIconProps {
  status: TestStatus;
  size?: number;
}

function StatusIcon({ status, size = 18 }: StatusIconProps) {
  switch (status) {
    case "passed":
      return <CheckCircle2 size={size} className="text-emerald-400" />;
    case "failed":
      return <XCircle size={size} className="text-red-400" />;
    case "running":
      return <Loader2 size={size} className="text-zinc-400 animate-spin" />;
    case "skipped":
      return <SkipForward size={size} className="text-zinc-500" />;
    case "idle":
    default:
      return <Circle size={size} className="text-zinc-600" />;
  }
}

// ============================================================================
// Status Badge Component
// ============================================================================

interface StatusBadgeProps {
  status: TestStatus;
}

function StatusBadge({ status }: StatusBadgeProps) {
  const config = {
    passed: {
      label: "PASSED",
      bgClass: "bg-emerald-500/10",
      textClass: "text-emerald-400",
      borderClass: "border-emerald-500/30",
    },
    failed: {
      label: "FAILED",
      bgClass: "bg-red-500/10",
      textClass: "text-red-400",
      borderClass: "border-red-500/30",
    },
    running: {
      label: "RUNNING",
      bgClass: "bg-zinc-500/10",
      textClass: "text-zinc-400",
      borderClass: "border-zinc-500/30",
    },
    skipped: {
      label: "SKIPPED",
      bgClass: "bg-zinc-500/10",
      textClass: "text-zinc-500",
      borderClass: "border-zinc-500/30",
    },
    idle: {
      label: "PENDING",
      bgClass: "bg-zinc-800",
      textClass: "text-zinc-500",
      borderClass: "border-zinc-700",
    },
  };

  const c = config[status];

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium border",
        c.bgClass,
        c.textClass,
        c.borderClass
      )}
    >
      {c.label}
    </span>
  );
}

// ============================================================================
// Duration Display Component
// ============================================================================

interface DurationDisplayProps {
  durationMs: number | null;
}

function DurationDisplay({ durationMs }: DurationDisplayProps) {
  if (durationMs === null) return null;

  const formatted = durationMs < 1000
    ? `${durationMs}ms`
    : `${(durationMs / 1000).toFixed(2)}s`;

  // Color based on latency
  const colorClass =
    durationMs < 100
      ? "text-emerald-500"
      : durationMs < 500
      ? "text-zinc-400"
      : durationMs < 1000
      ? "text-amber-400"
      : "text-red-400";

  return (
    <span className={cn("text-[11px] font-mono flex items-center gap-1", colorClass)}>
      <Clock size={10} />
      {formatted}
    </span>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function TestItem({
  result,
  onRetry,
  isRunning = false,
  expandedByDefault = false,
  className,
}: TestItemProps) {
  const [isExpanded, setIsExpanded] = useState(expandedByDefault);
  const metadata = PROBE_METADATA[result.id as ProbeKey];

  // Determine current status (override with running if applicable)
  const currentStatus: TestStatus = isRunning ? "running" : result.status;

  // Whether to show retry button
  const showRetry = result.status === "failed" && onRetry && !isRunning;

  // Whether item is clickable (has details to show)
  const hasDetails = result.message || result.backend;
  const isClickable = hasDetails && currentStatus !== "idle" && currentStatus !== "running";

  return (
    <div
      className={cn(
        "rounded-lg border transition-all duration-200",
        // Border color based on status
        currentStatus === "passed" && "border-zinc-800 bg-zinc-900/30",
        currentStatus === "failed" && "border-red-900/40 bg-red-950/10",
        currentStatus === "running" && "border-zinc-700 bg-zinc-900/50",
        currentStatus === "skipped" && "border-zinc-800 bg-zinc-900/20",
        currentStatus === "idle" && "border-zinc-800/50 bg-zinc-900/10",
        className
      )}
    >
      {/* Main Row */}
      <div
        className={cn(
          "flex items-center justify-between p-3",
          isClickable && "cursor-pointer hover:bg-zinc-800/30"
        )}
        onClick={isClickable ? () => setIsExpanded(!isExpanded) : undefined}
        role={isClickable ? "button" : undefined}
        tabIndex={isClickable ? 0 : undefined}
        onKeyDown={
          isClickable
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setIsExpanded(!isExpanded);
                }
              }
            : undefined
        }
      >
        {/* Left: Icon + Service Info */}
        <div className="flex items-center gap-3 min-w-0">
          {/* Status Icon */}
          <div className="shrink-0">
            <StatusIcon status={currentStatus} />
          </div>

          {/* Service Name & Description */}
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p
                className={cn(
                  "text-[13px] font-medium truncate",
                  currentStatus === "passed" && "text-zinc-200",
                  currentStatus === "failed" && "text-red-300",
                  currentStatus === "running" && "text-zinc-300",
                  currentStatus === "skipped" && "text-zinc-500",
                  currentStatus === "idle" && "text-zinc-500"
                )}
              >
                {result.name}
              </p>
              {metadata.isCritical && (
                <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30">
                  CRITICAL
                </span>
              )}
            </div>
            <p className="text-[11px] text-zinc-600 truncate">
              {metadata.description}
            </p>
          </div>
        </div>

        {/* Right: Duration + Status Badge + Expand */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Duration */}
          <DurationDisplay durationMs={result.durationMs} />

          {/* Status Badge */}
          <StatusBadge status={currentStatus} />

          {/* Retry Button */}
          {showRetry && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRetry?.();
              }}
              className={cn(
                "p-1.5 rounded-md transition-colors",
                "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-700"
              )}
              title="Retry test"
            >
              <RotateCcw size={14} />
            </button>
          )}

          {/* Expand Icon */}
          {isClickable && (
            <ChevronDown
              size={16}
              className={cn(
                "text-zinc-500 transition-transform",
                isExpanded && "rotate-180"
              )}
            />
          )}
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && hasDetails && (
        <div className="px-3 pb-3 pt-0 border-t border-zinc-800/50">
          <div className="pt-3 space-y-2">
            {/* Backend/Provider */}
            {result.backend && (
              <div className="flex items-center gap-2 text-[12px]">
                <Server size={12} className="text-zinc-500" />
                <span className="text-zinc-500">Provider:</span>
                <span className="text-zinc-300 font-mono">{result.backend}</span>
              </div>
            )}

            {/* Message */}
            {result.message && (
              <div className="flex items-start gap-2 text-[12px]">
                <AlertCircle size={12} className="text-zinc-500 shrink-0 mt-0.5" />
                <span className="text-zinc-500">Message:</span>
                <span
                  className={cn(
                    result.status === "failed" ? "text-red-400" : "text-zinc-300"
                  )}
                >
                  {result.message}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Skeleton Loader
// ============================================================================

export function TestItemSkeleton() {
  return (
    <div className="rounded-lg border border-zinc-800/50 bg-zinc-900/10 p-3 animate-pulse">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 rounded-full bg-zinc-700" />
          <div className="space-y-1.5">
            <div className="h-4 w-32 bg-zinc-700 rounded" />
            <div className="h-3 w-48 bg-zinc-800 rounded" />
          </div>
        </div>
        <div className="h-5 w-16 bg-zinc-700 rounded" />
      </div>
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

TestItem.displayName = "TestItem";
TestItemSkeleton.displayName = "TestItemSkeleton";

// ============================================================================
// Default Export
// ============================================================================

export default TestItem;
