"use client";

/**
 * ReviewStep Component
 *
 * Final step of the configuration wizard.
 * Displays a summary of all configurations and allows running connectivity tests.
 *
 * Features:
 * - Configuration summary for all services
 * - Run all connectivity tests
 * - Show test results with pass/fail status
 * - Next steps guidance
 *
 * @example
 * <ReviewStep
 *   healthData={wizard.healthData}
 *   onRunTests={wizard.runTests}
 *   isTesting={wizard.isTesting}
 * />
 */

import React from "react";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  AlertCircle,
  XCircle,
  Play,
  Loader2,
  ExternalLink,
  Rocket,
} from "lucide-react";
import type { DetailedHealthResponse, ProbeKey, ProbeResult } from "@/types";
import { PROBE_METADATA, PROBE_ORDER, formatLatency } from "@/lib/utils/health";

// ============================================================================
// Types
// ============================================================================

export interface ReviewStepProps {
  /** Health check data for status display */
  healthData?: DetailedHealthResponse;
  /** Whether health data is loading */
  isLoadingHealth?: boolean;
  /** Handler to run tests */
  onRunTests?: () => void;
  /** Whether tests are currently running */
  isTesting?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Status Summary Component
// ============================================================================

interface StatusSummaryProps {
  healthData?: DetailedHealthResponse;
}

function StatusSummary({ healthData }: StatusSummaryProps) {
  if (!healthData) {
    return null;
  }

  const probes = Object.entries(healthData.probes) as [ProbeKey, ProbeResult][];
  const upCount = probes.filter(([, p]) => p.verdict === "up").length;
  const warnCount = probes.filter(([, p]) => p.verdict === "warn").length;
  const downCount = probes.filter(([, p]) => p.verdict === "down").length;

  const overallStatus = healthData.verdict;

  return (
    <div
      className={cn(
        "p-4 rounded-lg border",
        overallStatus === "up" && "border-emerald-900/30 bg-emerald-950/10",
        overallStatus === "warn" && "border-amber-900/30 bg-amber-950/10",
        overallStatus === "down" && "border-red-900/30 bg-red-950/10"
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          {overallStatus === "up" ? (
            <CheckCircle2 size={24} className="text-emerald-400" />
          ) : overallStatus === "warn" ? (
            <AlertCircle size={24} className="text-amber-400" />
          ) : (
            <XCircle size={24} className="text-red-400" />
          )}
          <div>
            <p
              className={cn(
                "text-[16px] font-semibold",
                overallStatus === "up" && "text-emerald-400",
                overallStatus === "warn" && "text-amber-400",
                overallStatus === "down" && "text-red-400"
              )}
            >
              {overallStatus === "up"
                ? "All Systems Operational"
                : overallStatus === "warn"
                ? "Some Services Degraded"
                : "System Issues Detected"}
            </p>
            <p className="text-[12px] text-zinc-500 mt-0.5">
              {upCount} healthy, {warnCount} warnings, {downCount} errors
            </p>
          </div>
        </div>

        {/* Uptime */}
        <div className="text-right">
          <p className="text-[11px] text-zinc-500">Uptime</p>
          <p className="text-[13px] text-zinc-300 font-mono">
            {formatUptime(healthData.alive_seconds)}
          </p>
        </div>
      </div>
    </div>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

// ============================================================================
// Service List Component
// ============================================================================

interface ServiceListProps {
  healthData?: DetailedHealthResponse;
}

function ServiceList({ healthData }: ServiceListProps) {
  if (!healthData) {
    return (
      <div className="space-y-2">
        {PROBE_ORDER.map((key) => (
          <div
            key={key}
            className="flex items-center gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800"
          >
            <div className="w-4 h-4 rounded-full bg-zinc-700 animate-pulse" />
            <div className="h-4 w-32 bg-zinc-700 rounded animate-pulse" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {PROBE_ORDER.map((key) => {
        const probe = healthData.probes[key as ProbeKey];
        const metadata = PROBE_METADATA[key as ProbeKey];

        return (
          <div
            key={key}
            className={cn(
              "flex items-center justify-between p-3 rounded-lg border",
              probe.verdict === "up" && "border-zinc-800 bg-zinc-900/30",
              probe.verdict === "warn" && "border-amber-900/30 bg-amber-950/10",
              probe.verdict === "down" && "border-red-900/30 bg-red-950/10"
            )}
          >
            <div className="flex items-center gap-3">
              {/* Status Icon */}
              {probe.verdict === "up" ? (
                <CheckCircle2 size={16} className="text-emerald-400" />
              ) : probe.verdict === "warn" ? (
                <AlertCircle size={16} className="text-amber-400" />
              ) : (
                <XCircle size={16} className="text-red-400" />
              )}

              {/* Service Info */}
              <div>
                <p className="text-[13px] font-medium text-zinc-200">
                  {metadata.displayName}
                </p>
                <p className="text-[11px] text-zinc-500">
                  {probe.backend !== "unknown" ? probe.backend : "Not configured"}
                </p>
              </div>
            </div>

            {/* Status & Latency */}
            <div className="flex items-center gap-3">
              {probe.latency_ms > 0 && (
                <span className="text-[11px] text-zinc-600 font-mono">
                  {formatLatency(probe.latency_ms)}
                </span>
              )}
              <span
                className={cn(
                  "px-2 py-0.5 rounded text-[10px] font-medium",
                  probe.verdict === "up" && "bg-emerald-500/20 text-emerald-400",
                  probe.verdict === "warn" && "bg-amber-500/20 text-amber-400",
                  probe.verdict === "down" && "bg-red-500/20 text-red-400"
                )}
              >
                {probe.verdict.toUpperCase()}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// Test Button Component
// ============================================================================

interface TestButtonProps {
  onClick?: () => void;
  isLoading?: boolean;
}

function TestButton({ onClick, isLoading = false }: TestButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={isLoading}
      className={cn(
        "flex items-center justify-center gap-2 w-full py-3 px-4 rounded-lg",
        "text-[14px] font-medium transition-all",
        "bg-zinc-800 text-zinc-200 border border-zinc-700",
        "hover:bg-zinc-700 hover:border-zinc-600",
        "focus:outline-none focus:ring-2 focus:ring-zinc-500 focus:ring-offset-2 focus:ring-offset-zinc-900",
        "disabled:opacity-50 disabled:cursor-not-allowed"
      )}
    >
      {isLoading ? (
        <>
          <Loader2 size={18} className="animate-spin" />
          Running Tests...
        </>
      ) : (
        <>
          <Play size={18} />
          Run All Tests
        </>
      )}
    </button>
  );
}

// ============================================================================
// Next Steps Component
// ============================================================================

function NextSteps() {
  const steps = [
    {
      title: "Ingest your first document",
      description: "Use the Memorize page to add documents to your knowledge graph",
      link: "/memorize",
    },
    {
      title: "Explore the knowledge graph",
      description: "View and interact with your memories in the Memories page",
      link: "/memories",
    },
    {
      title: "Search your knowledge",
      description: "Use the Retrieve page to search across all your memories",
      link: "/retrieve",
    },
  ];

  return (
    <div className="space-y-3">
      <h4 className="text-[14px] font-medium text-zinc-200 flex items-center gap-2">
        <Rocket size={16} className="text-emerald-400" />
        Next Steps
      </h4>
      <div className="grid gap-3">
        {steps.map((step, idx) => (
          <a
            key={idx}
            href={step.link}
            className={cn(
              "flex items-center justify-between p-3 rounded-lg",
              "bg-zinc-900/30 border border-zinc-800",
              "hover:bg-zinc-900/50 hover:border-zinc-700 transition-colors group"
            )}
          >
            <div>
              <p className="text-[13px] font-medium text-zinc-200 group-hover:text-zinc-100">
                {step.title}
              </p>
              <p className="text-[11px] text-zinc-500">{step.description}</p>
            </div>
            <ExternalLink size={14} className="text-zinc-600 group-hover:text-zinc-400" />
          </a>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ReviewStep({
  healthData,
  isLoadingHealth = false,
  onRunTests,
  isTesting = false,
  className,
}: ReviewStepProps) {
  const isAllHealthy = healthData?.verdict === "up";

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div>
        <h3 className="text-[16px] font-semibold text-zinc-100">
          Review & Test Configuration
        </h3>
        <p className="text-[13px] text-zinc-500 mt-1">
          Verify all services are configured correctly and run connectivity tests.
        </p>
      </div>

      {/* Overall Status */}
      <StatusSummary healthData={healthData} />

      {/* Test Button */}
      <TestButton onClick={onRunTests} isLoading={isTesting || isLoadingHealth} />

      {/* Service List */}
      <div className="space-y-3">
        <h4 className="text-[14px] font-medium text-zinc-200">Service Status</h4>
        <ServiceList healthData={healthData} />
      </div>

      {/* Build Info */}
      {healthData && (
        <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-900/30 border border-zinc-800">
          <span className="text-[11px] text-zinc-500">Build Version</span>
          <span className="text-[11px] text-zinc-400 font-mono">
            {healthData.build || "unknown"}
          </span>
        </div>
      )}

      {/* Next Steps (show when all healthy) */}
      {isAllHealthy && <NextSteps />}

      {/* Troubleshooting link (show when issues) */}
      {!isAllHealthy && healthData && (
        <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-950/10 border border-amber-900/30">
          <AlertCircle size={16} className="text-amber-400 shrink-0 mt-0.5" />
          <div className="text-[12px]">
            <p className="text-amber-400 font-medium">Configuration Issues Detected</p>
            <p className="text-zinc-500 mt-1">
              Review the services above and check your <code className="text-zinc-400">.env</code>{" "}
              configuration. Restart the server after making changes.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Display Name
// ============================================================================

ReviewStep.displayName = "ReviewStep";

// ============================================================================
// Default Export
// ============================================================================

export default ReviewStep;
