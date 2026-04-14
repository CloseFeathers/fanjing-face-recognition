"use client";

/**
 * TestRunner Component
 *
 * Main component for the Quick Test panel in the Setup page.
 * Orchestrates running connectivity tests for all system components.
 *
 * Features:
 * - Run all tests with a single click
 * - Visual test progress indication
 * - Individual test results with details
 * - Summary statistics
 * - Retry failed tests
 * - Auto-run option
 *
 * @example
 * <TestRunner
 *   autoRun={false}
 *   onComplete={(results) => console.log(results)}
 * />
 */

import React, { useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Play,
  RotateCcw,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { useQuickTest } from "@/hooks/use-quick-test";
import { TestItem, TestItemSkeleton } from "./TestItem";
import { TestResults } from "./TestResults";
import type { TestResult } from "@/types/setup";
import type { ProbeKey } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface TestRunnerProps {
  /** Callback when all tests complete */
  onComplete?: (results: TestResult[]) => void;
  /** Whether to auto-run tests on mount */
  autoRun?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Action Button Component
// ============================================================================

interface ActionButtonProps {
  onClick: () => void;
  isLoading?: boolean;
  variant?: "primary" | "secondary" | "ghost";
  icon?: React.ReactNode;
  children: React.ReactNode;
  disabled?: boolean;
  className?: string;
}

function ActionButton({
  onClick,
  isLoading = false,
  variant = "primary",
  icon,
  children,
  disabled = false,
  className,
}: ActionButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className={cn(
        "inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all duration-200",
        "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-900",
        // Size
        "px-4 py-2.5 text-[13px]",
        // Variants
        variant === "primary" &&
          "bg-zinc-100 text-zinc-900 hover:bg-white focus:ring-zinc-400",
        variant === "secondary" &&
          "bg-zinc-800 text-zinc-200 border border-zinc-700 hover:bg-zinc-700 focus:ring-zinc-500",
        variant === "ghost" &&
          "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 focus:ring-zinc-500",
        // Disabled state
        (disabled || isLoading) && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {isLoading ? <Loader2 size={16} className="animate-spin" /> : icon}
      {children}
    </button>
  );
}

// ============================================================================
// Error Banner Component
// ============================================================================

interface ErrorBannerProps {
  error: string;
  onDismiss: () => void;
}

function ErrorBanner({ error, onDismiss }: ErrorBannerProps) {
  return (
    <div className="flex items-start justify-between gap-3 p-3 rounded-lg bg-red-950/20 border border-red-900/40">
      <div className="flex items-start gap-3">
        <AlertCircle size={18} className="text-red-400 shrink-0 mt-0.5" />
        <div>
          <p className="text-[13px] font-medium text-red-400">Test Error</p>
          <p className="text-[12px] text-red-300/80 mt-0.5">{error}</p>
        </div>
      </div>
      <button
        onClick={onDismiss}
        className="text-red-400 hover:text-red-300 p-1"
        title="Dismiss"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path d="M18 6L6 18M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

// ============================================================================
// Header Component
// ============================================================================

interface TestRunnerHeaderProps {
  isRunning: boolean;
  hasRun: boolean;
  isAllPassed: boolean;
  isAnyFailed: boolean;
  onRunAll: () => void;
  onReset: () => void;
}

function TestRunnerHeader({
  isRunning,
  hasRun,
  isAllPassed,
  isAnyFailed,
  onRunAll,
  onReset,
}: TestRunnerHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      {/* Left: Status Indicator */}
      <div className="flex items-center gap-3">
        {isRunning ? (
          <div className="flex items-center gap-2 text-zinc-400">
            <Loader2 size={18} className="animate-spin" />
            <span className="text-[13px]">Running tests...</span>
          </div>
        ) : hasRun ? (
          <div
            className={cn(
              "flex items-center gap-2",
              isAllPassed && "text-emerald-400",
              isAnyFailed && "text-red-400"
            )}
          >
            {isAllPassed ? (
              <CheckCircle2 size={18} />
            ) : (
              <AlertCircle size={18} />
            )}
            <span className="text-[13px]">
              {isAllPassed ? "All systems operational" : "Issues detected"}
            </span>
          </div>
        ) : (
          <span className="text-[13px] text-zinc-500">
            Ready to run connectivity tests
          </span>
        )}
      </div>

      {/* Right: Action Buttons */}
      <div className="flex items-center gap-2">
        {hasRun && (
          <ActionButton
            onClick={onReset}
            variant="ghost"
            icon={<RefreshCw size={14} />}
            disabled={isRunning}
          >
            Reset
          </ActionButton>
        )}
        <ActionButton
          onClick={onRunAll}
          variant="primary"
          icon={isRunning ? undefined : <Play size={16} />}
          isLoading={isRunning}
        >
          {isRunning ? "Running..." : hasRun ? "Run Again" : "Run All Tests"}
        </ActionButton>
      </div>
    </div>
  );
}

// ============================================================================
// Test List Component
// ============================================================================

interface TestListProps {
  results: TestResult[];
  currentTestIndex: number | null;
  isRunning: boolean;
  onRetry: (testId: ProbeKey) => void;
}

function TestList({
  results,
  currentTestIndex,
  isRunning,
  onRetry,
}: TestListProps) {
  return (
    <div className="space-y-2">
      {results.map((result, index) => (
        <TestItem
          key={result.id}
          result={result}
          isRunning={isRunning && currentTestIndex === index}
          onRetry={
            result.status === "failed"
              ? () => onRetry(result.id as ProbeKey)
              : undefined
          }
        />
      ))}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function TestRunner({
  onComplete,
  autoRun = false,
  className,
}: TestRunnerProps) {
  const test = useQuickTest();

  // Auto-run tests on mount if enabled
  useEffect(() => {
    if (autoRun && !test.hasRun && !test.isRunning) {
      test.runAllTests();
    }
  }, [autoRun, test.hasRun, test.isRunning, test.runAllTests]);

  // Call onComplete when tests finish
  useEffect(() => {
    if (test.summary.isComplete && test.hasRun && onComplete) {
      onComplete(test.results);
    }
  }, [test.summary.isComplete, test.hasRun, test.results, onComplete]);

  // Handle retry
  const handleRetry = useCallback(
    (testId: ProbeKey) => {
      test.runSingleTest(testId);
    },
    [test.runSingleTest]
  );

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header with Actions */}
      <TestRunnerHeader
        isRunning={test.isRunning}
        hasRun={test.hasRun}
        isAllPassed={test.isAllPassed}
        isAnyFailed={test.isAnyFailed}
        onRunAll={test.runAllTests}
        onReset={test.reset}
      />

      {/* Error Banner */}
      {test.error && (
        <ErrorBanner error={test.error} onDismiss={test.clearError} />
      )}

      {/* Results Summary (shown after tests run) */}
      {test.hasRun && (
        <TestResults
          summary={test.summary}
          lastRunAt={test.lastRunAt}
          isRunning={test.isRunning}
        />
      )}

      {/* Test List */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-[14px] font-medium text-zinc-200">
            Service Tests
          </h4>
          <span className="text-[11px] text-zinc-600">
            {test.results.length} services
          </span>
        </div>
        <TestList
          results={test.results}
          currentTestIndex={test.currentTestIndex}
          isRunning={test.isRunning}
          onRetry={handleRetry}
        />
      </div>

      {/* Quick Tips (shown before first run) */}
      {!test.hasRun && !test.isRunning && (
        <div className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800">
          <h4 className="text-[13px] font-medium text-zinc-300 mb-2">
            What This Tests
          </h4>
          <ul className="space-y-1.5 text-[12px] text-zinc-500">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Database connectivity (SQLite, Postgres)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Vector store connection (LanceDB, Qdrant, etc.)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Graph database connection (Kuzu, Neo4j)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              File storage access
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              LLM provider API connection
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Embedding service availability
            </li>
          </ul>
        </div>
      )}

      {/* All Passed Message */}
      {test.isAllPassed && (
        <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-900/40">
          <div className="flex items-start gap-3">
            <CheckCircle2 size={20} className="text-emerald-400 shrink-0" />
            <div>
              <p className="text-[14px] font-medium text-emerald-400">
                Ready to Go!
              </p>
              <p className="text-[12px] text-zinc-400 mt-1">
                All services are connected and operational. You can now start
                ingesting documents and building your knowledge graph.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Failed Tests Help */}
      {test.isAnyFailed && test.summary.isComplete && (
        <div className="p-4 rounded-lg bg-amber-950/20 border border-amber-900/40">
          <div className="flex items-start gap-3">
            <AlertCircle size={20} className="text-amber-400 shrink-0" />
            <div>
              <p className="text-[14px] font-medium text-amber-400">
                Troubleshooting Tips
              </p>
              <ul className="text-[12px] text-zinc-400 mt-2 space-y-1.5">
                <li>• Check your <code className="text-zinc-300">.env</code> file for correct configuration</li>
                <li>• Ensure all required services are running</li>
                <li>• Verify API keys are valid and have proper permissions</li>
                <li>• Check network connectivity to external services</li>
                <li>• Review the Troubleshooting section below for common issues</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Skeleton Loader
// ============================================================================

export function TestRunnerSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="h-5 w-48 bg-zinc-800 rounded" />
        <div className="h-10 w-32 bg-zinc-800 rounded-lg" />
      </div>

      {/* Test Items */}
      <div className="space-y-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <TestItemSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

TestRunner.displayName = "TestRunner";
TestRunnerSkeleton.displayName = "TestRunnerSkeleton";

// ============================================================================
// Default Export
// ============================================================================

export default TestRunner;
