"use client";

/**
 * TestResults Component
 *
 * Displays a summary of all test results including:
 * - Overall pass/fail status
 * - Pass rate percentage
 * - Test counts by status
 * - Total execution time
 * - Visual progress indicators
 *
 * Features:
 * - Animated progress bar
 * - Color-coded status display
 * - Detailed breakdown view
 * - Celebratory animation on all pass
 *
 * @example
 * <TestResults
 *   summary={testSummary}
 *   lastRunAt={lastRunTime}
 *   isRunning={false}
 * />
 */

import React from "react";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Activity,
  PartyPopper,
  Loader2,
} from "lucide-react";
import type { TestSummary } from "@/hooks/use-quick-test";

// ============================================================================
// Types
// ============================================================================

export interface TestResultsProps {
  /** Test summary data */
  summary: TestSummary;
  /** Last run timestamp */
  lastRunAt: Date | null;
  /** Whether tests are currently running */
  isRunning?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Progress Bar Component
// ============================================================================

interface ProgressBarProps {
  summary: TestSummary;
  isAnimating?: boolean;
}

function ProgressBar({ summary, isAnimating = false }: ProgressBarProps) {
  const { total, passed, failed, running } = summary;

  // Calculate percentages
  const passedPercent = total > 0 ? (passed / total) * 100 : 0;
  const failedPercent = total > 0 ? (failed / total) * 100 : 0;
  const runningPercent = total > 0 ? (running / total) * 100 : 0;

  return (
    <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
      <div className="h-full flex">
        {/* Passed segment */}
        <div
          className={cn(
            "bg-emerald-500 transition-all duration-500",
            isAnimating && "animate-pulse"
          )}
          style={{ width: `${passedPercent}%` }}
        />
        {/* Failed segment */}
        <div
          className="bg-red-500 transition-all duration-500"
          style={{ width: `${failedPercent}%` }}
        />
        {/* Running segment */}
        <div
          className={cn(
            "bg-zinc-500 transition-all duration-500",
            running > 0 && "animate-pulse"
          )}
          style={{ width: `${runningPercent}%` }}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Stat Card Component
// ============================================================================

interface StatCardProps {
  label: string;
  value: number | string;
  icon: React.ReactNode;
  variant?: "default" | "success" | "error" | "warning";
}

function StatCard({ label, value, icon, variant = "default" }: StatCardProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 p-3 rounded-lg border",
        variant === "default" && "border-zinc-800 bg-zinc-900/30",
        variant === "success" && "border-emerald-900/40 bg-emerald-950/20",
        variant === "error" && "border-red-900/40 bg-red-950/20",
        variant === "warning" && "border-amber-900/40 bg-amber-950/20"
      )}
    >
      <div
        className={cn(
          "shrink-0",
          variant === "default" && "text-zinc-500",
          variant === "success" && "text-emerald-400",
          variant === "error" && "text-red-400",
          variant === "warning" && "text-amber-400"
        )}
      >
        {icon}
      </div>
      <div>
        <p
          className={cn(
            "text-[18px] font-semibold",
            variant === "default" && "text-zinc-200",
            variant === "success" && "text-emerald-400",
            variant === "error" && "text-red-400",
            variant === "warning" && "text-amber-400"
          )}
        >
          {value}
        </p>
        <p className="text-[11px] text-zinc-500">{label}</p>
      </div>
    </div>
  );
}

// ============================================================================
// Time Display Helper
// ============================================================================

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
}

function formatTimeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

// ============================================================================
// Main Component
// ============================================================================

export function TestResults({
  summary,
  lastRunAt,
  isRunning = false,
  className,
}: TestResultsProps) {
  const { total, passed, failed, skipped, passRate, totalDurationMs, isComplete } = summary;

  // Determine overall status
  const isAllPassed = isComplete && failed === 0 && passed > 0;
  const hasFailures = failed > 0;
  const isIdle = !isComplete && !isRunning;

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header Banner */}
      <div
        className={cn(
          "p-4 rounded-lg border",
          isRunning && "border-zinc-700 bg-zinc-900/50",
          isAllPassed && "border-emerald-900/40 bg-emerald-950/20",
          hasFailures && "border-red-900/40 bg-red-950/20",
          isIdle && "border-zinc-800 bg-zinc-900/30"
        )}
      >
        <div className="flex items-center justify-between">
          {/* Left: Status Icon + Message */}
          <div className="flex items-center gap-3">
            {isRunning ? (
              <Loader2 size={24} className="text-zinc-400 animate-spin" />
            ) : isAllPassed ? (
              <div className="relative">
                <CheckCircle2 size={24} className="text-emerald-400" />
                <PartyPopper
                  size={14}
                  className="absolute -top-1 -right-2 text-amber-400 animate-bounce"
                />
              </div>
            ) : hasFailures ? (
              <XCircle size={24} className="text-red-400" />
            ) : (
              <Activity size={24} className="text-zinc-500" />
            )}

            <div>
              <p
                className={cn(
                  "text-[16px] font-semibold",
                  isRunning && "text-zinc-300",
                  isAllPassed && "text-emerald-400",
                  hasFailures && "text-red-400",
                  isIdle && "text-zinc-400"
                )}
              >
                {isRunning
                  ? "Running Tests..."
                  : isAllPassed
                  ? "All Tests Passed!"
                  : hasFailures
                  ? `${failed} Test${failed > 1 ? "s" : ""} Failed`
                  : "Ready to Run Tests"}
              </p>
              <p className="text-[12px] text-zinc-500">
                {isRunning
                  ? `Testing ${total} services...`
                  : isComplete
                  ? `${passed} of ${total} tests passed`
                  : "Click 'Run All Tests' to start"}
              </p>
            </div>
          </div>

          {/* Right: Pass Rate */}
          {isComplete && (
            <div className="text-right">
              <p
                className={cn(
                  "text-[28px] font-bold",
                  isAllPassed && "text-emerald-400",
                  hasFailures && passRate >= 50 && "text-amber-400",
                  hasFailures && passRate < 50 && "text-red-400"
                )}
              >
                {passRate}%
              </p>
              <p className="text-[11px] text-zinc-500">Pass Rate</p>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {(isRunning || isComplete) && (
          <div className="mt-4">
            <ProgressBar summary={summary} isAnimating={isRunning} />
          </div>
        )}
      </div>

      {/* Stats Grid */}
      {isComplete && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard
            label="Passed"
            value={passed}
            icon={<CheckCircle2 size={20} />}
            variant={passed > 0 ? "success" : "default"}
          />
          <StatCard
            label="Failed"
            value={failed}
            icon={<XCircle size={20} />}
            variant={failed > 0 ? "error" : "default"}
          />
          <StatCard
            label="Skipped"
            value={skipped}
            icon={<AlertTriangle size={20} />}
            variant={skipped > 0 ? "warning" : "default"}
          />
          <StatCard
            label="Duration"
            value={formatDuration(totalDurationMs)}
            icon={<Clock size={20} />}
            variant="default"
          />
        </div>
      )}

      {/* Last Run Time */}
      {lastRunAt && (
        <div className="flex items-center justify-center gap-2 text-[11px] text-zinc-600">
          <Clock size={12} />
          <span>Last run: {formatTimeAgo(lastRunAt)}</span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Compact Results (for inline use)
// ============================================================================

export interface CompactResultsProps {
  summary: TestSummary;
  className?: string;
}

export function CompactResults({ summary, className }: CompactResultsProps) {
  const { passed, failed, total, passRate, isComplete } = summary;

  if (!isComplete) return null;

  return (
    <div className={cn("flex items-center gap-3", className)}>
      {/* Pass/Fail Counts */}
      <div className="flex items-center gap-2 text-[12px]">
        <span className="flex items-center gap-1 text-emerald-400">
          <CheckCircle2 size={14} />
          {passed}
        </span>
        <span className="text-zinc-600">/</span>
        <span
          className={cn(
            "flex items-center gap-1",
            failed > 0 ? "text-red-400" : "text-zinc-500"
          )}
        >
          <XCircle size={14} />
          {failed}
        </span>
      </div>

      {/* Pass Rate */}
      <span
        className={cn(
          "px-2 py-0.5 rounded text-[10px] font-medium",
          passRate === 100 && "bg-emerald-500/20 text-emerald-400",
          passRate >= 50 && passRate < 100 && "bg-amber-500/20 text-amber-400",
          passRate < 50 && "bg-red-500/20 text-red-400"
        )}
      >
        {passRate}%
      </span>
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

TestResults.displayName = "TestResults";
CompactResults.displayName = "CompactResults";

// ============================================================================
// Default Export
// ============================================================================

export default TestResults;
