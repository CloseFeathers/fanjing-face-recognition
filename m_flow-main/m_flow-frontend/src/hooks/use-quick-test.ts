"use client";

/**
 * Quick Test State Management Hook
 *
 * Manages the state for the Quick Test panel in the Setup page.
 * Handles running connectivity tests for all system components.
 *
 * Features:
 * - Run all tests at once
 * - Individual test status tracking
 * - Test duration measurement
 * - Error handling and retry logic
 * - Integration with health check API
 *
 * @example
 * // Basic usage
 * const test = useQuickTest();
 *
 * // Run all tests
 * await test.runAllTests();
 *
 * // Check results
 * console.log(test.results);
 * console.log(test.summary);
 */

import { useCallback, useMemo, useState, useEffect } from "react";
import { useDetailedHealth } from "./use-api";
import type { ProbeKey, ProbeResult, DetailedHealthResponse } from "@/types";
import type { TestResult, TestStatus, QuickTestState } from "@/types/setup";
import { PROBE_METADATA, PROBE_ORDER } from "@/lib/utils/health";

// ============================================================================
// Types
// ============================================================================

/** Test summary statistics */
export interface TestSummary {
  /** Total number of tests */
  total: number;
  /** Number of passed tests */
  passed: number;
  /** Number of failed tests */
  failed: number;
  /** Number of skipped tests */
  skipped: number;
  /** Number of tests still running */
  running: number;
  /** Whether all tests have completed */
  isComplete: boolean;
  /** Overall pass rate (0-100) */
  passRate: number;
  /** Total duration in milliseconds */
  totalDurationMs: number;
}

/** Hook return type */
export interface UseQuickTestReturn {
  // State
  results: TestResult[];
  isRunning: boolean;
  lastRunAt: Date | null;
  currentTestIndex: number | null;
  error: string | null;

  // Derived state
  summary: TestSummary;
  hasRun: boolean;
  isAllPassed: boolean;
  isAnyFailed: boolean;

  // Actions
  runAllTests: () => Promise<void>;
  runSingleTest: (testId: ProbeKey) => Promise<void>;
  reset: () => void;
  clearError: () => void;

  // Health data (direct access)
  healthData: DetailedHealthResponse | undefined;
  isLoadingHealth: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert health probe verdict to test status
 */
function verdictToTestStatus(verdict: "up" | "warn" | "down"): TestStatus {
  switch (verdict) {
    case "up":
      return "passed";
    case "warn":
      return "passed"; // Warn is still operational
    case "down":
      return "failed";
    default:
      return "failed";
  }
}

/**
 * Create initial test results from probe metadata
 */
function createInitialResults(): TestResult[] {
  return PROBE_ORDER.map((key) => ({
    id: key,
    name: PROBE_METADATA[key].displayName,
    status: "idle" as TestStatus,
    durationMs: null,
    message: null,
    backend: null,
  }));
}

/**
 * Convert health probe result to test result
 */
function probeToTestResult(key: ProbeKey, probe: ProbeResult): TestResult {
  const metadata = PROBE_METADATA[key];
  const status = verdictToTestStatus(probe.verdict);

  let message: string;
  if (probe.verdict === "up") {
    message = probe.backend !== "unknown"
      ? `Connected to ${probe.backend}`
      : "Service is healthy";
  } else if (probe.verdict === "warn") {
    message = probe.note || "Service is degraded but operational";
  } else {
    message = probe.note || "Connection failed";
  }

  return {
    id: key,
    name: metadata.displayName,
    status,
    durationMs: probe.latency_ms > 0 ? probe.latency_ms : null,
    message,
    backend: probe.backend !== "unknown" ? probe.backend : null,
  };
}

/**
 * Calculate test summary from results
 */
function calculateSummary(results: TestResult[]): TestSummary {
  const total = results.length;
  const passed = results.filter((r) => r.status === "passed").length;
  const failed = results.filter((r) => r.status === "failed").length;
  const skipped = results.filter((r) => r.status === "skipped").length;
  const running = results.filter((r) => r.status === "running").length;
  const isComplete = running === 0 && results.every((r) => r.status !== "idle");
  const passRate = total > 0 ? Math.round((passed / total) * 100) : 0;
  const totalDurationMs = results.reduce(
    (sum, r) => sum + (r.durationMs || 0),
    0
  );

  return {
    total,
    passed,
    failed,
    skipped,
    running,
    isComplete,
    passRate,
    totalDurationMs,
  };
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useQuickTest(): UseQuickTestReturn {
  // -------------------------------------------------------------------------
  // API Hook
  // -------------------------------------------------------------------------
  const {
    data: healthData,
    isLoading: isLoadingHealth,
    refetch: refetchHealth,
    error: healthError,
  } = useDetailedHealth({
    refetchInterval: false, // Manual refresh only
    enabled: true,
  });

  // -------------------------------------------------------------------------
  // Local State
  // -------------------------------------------------------------------------
  const [results, setResults] = useState<TestResult[]>(createInitialResults());
  const [isRunning, setIsRunning] = useState(false);
  const [lastRunAt, setLastRunAt] = useState<Date | null>(null);
  const [currentTestIndex, setCurrentTestIndex] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // -------------------------------------------------------------------------
  // Derived State
  // -------------------------------------------------------------------------
  const summary = useMemo(() => calculateSummary(results), [results]);
  const hasRun = lastRunAt !== null;
  const isAllPassed = summary.isComplete && summary.failed === 0;
  const isAnyFailed = summary.failed > 0;

  // -------------------------------------------------------------------------
  // Run All Tests
  // -------------------------------------------------------------------------
  const runAllTests = useCallback(async () => {
    setIsRunning(true);
    setError(null);

    // Set all tests to running state
    setResults((prev) =>
      prev.map((r) => ({ ...r, status: "running" as TestStatus }))
    );

    try {
      // Simulate sequential test execution with visual feedback
      for (let i = 0; i < PROBE_ORDER.length; i++) {
        setCurrentTestIndex(i);
        // Small delay for visual effect
        await new Promise((resolve) => setTimeout(resolve, 100));
      }

      // Fetch health data
      const result = await refetchHealth();

      if (result.data) {
        // Update results from health data
        setResults(
          PROBE_ORDER.map((key) =>
            probeToTestResult(key, result.data!.probes[key])
          )
        );
      } else {
        // Mark all as failed if no data
        setResults((prev) =>
          prev.map((r) => ({
            ...r,
            status: "failed" as TestStatus,
            message: "Failed to fetch health data",
          }))
        );
        setError("Failed to connect to the server");
      }
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : "Test execution failed";
      setError(errorMessage);
      setResults((prev) =>
        prev.map((r) => ({
          ...r,
          status: "failed" as TestStatus,
          message: errorMessage,
        }))
      );
    } finally {
      setIsRunning(false);
      setLastRunAt(new Date());
      setCurrentTestIndex(null);
    }
  }, [refetchHealth]);

  // -------------------------------------------------------------------------
  // Run Single Test
  // -------------------------------------------------------------------------
  const runSingleTest = useCallback(
    async (testId: ProbeKey) => {
      // Update single test to running
      setResults((prev) =>
        prev.map((r) =>
          r.id === testId ? { ...r, status: "running" as TestStatus } : r
        )
      );

      const testIndex = PROBE_ORDER.indexOf(testId);
      setCurrentTestIndex(testIndex);

      try {
        const result = await refetchHealth();

        if (result.data) {
          const probe = result.data.probes[testId];
          setResults((prev) =>
            prev.map((r) =>
              r.id === testId ? probeToTestResult(testId, probe) : r
            )
          );
        } else {
          setResults((prev) =>
            prev.map((r) =>
              r.id === testId
                ? {
                    ...r,
                    status: "failed" as TestStatus,
                    message: "Failed to fetch health data",
                  }
                : r
            )
          );
        }
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : "Test failed";
        setResults((prev) =>
          prev.map((r) =>
            r.id === testId
              ? { ...r, status: "failed" as TestStatus, message: errorMessage }
              : r
          )
        );
      } finally {
        setCurrentTestIndex(null);
      }
    },
    [refetchHealth]
  );

  // -------------------------------------------------------------------------
  // Reset
  // -------------------------------------------------------------------------
  const reset = useCallback(() => {
    setResults(createInitialResults());
    setIsRunning(false);
    setLastRunAt(null);
    setCurrentTestIndex(null);
    setError(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // -------------------------------------------------------------------------
  // Auto-populate from health data if available
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (healthData && !hasRun && !isRunning) {
      // Pre-populate results from existing health data
      setResults(
        PROBE_ORDER.map((key) =>
          probeToTestResult(key, healthData.probes[key])
        )
      );
    }
  }, [healthData, hasRun, isRunning]);

  // -------------------------------------------------------------------------
  // Return
  // -------------------------------------------------------------------------
  return useMemo(
    () => ({
      // State
      results,
      isRunning,
      lastRunAt,
      currentTestIndex,
      error,

      // Derived state
      summary,
      hasRun,
      isAllPassed,
      isAnyFailed,

      // Actions
      runAllTests,
      runSingleTest,
      reset,
      clearError,

      // Health data
      healthData,
      isLoadingHealth,
    }),
    [
      results,
      isRunning,
      lastRunAt,
      currentTestIndex,
      error,
      summary,
      hasRun,
      isAllPassed,
      isAnyFailed,
      runAllTests,
      runSingleTest,
      reset,
      clearError,
      healthData,
      isLoadingHealth,
    ]
  );
}

// ============================================================================
// Display Name
// ============================================================================

useQuickTest.displayName = "useQuickTest";

// ============================================================================
// Default Export
// ============================================================================

export default useQuickTest;
