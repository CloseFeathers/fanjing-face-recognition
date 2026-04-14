"use client";

/**
 * ActiveOperations Component
 *
 * Displays currently running pipeline operations with real-time progress.
 * Shows progress bars, current step, and elapsed time.
 */

import React, { useMemo, useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { useActivePipelines } from "@/hooks/use-active-pipelines";
import { useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { ActivePipeline } from "@/types";
import { Clock, Database, AlertTriangle, X } from "lucide-react";
import { toast } from "sonner";
import { useConfirm } from "@/components/ui/confirm-dialog";

// ============================================================================
// Constants
// ============================================================================

// Base threshold for stale detection (minutes)
// This is the minimum time before considering a pipeline stale
const BASE_STALE_THRESHOLD_MINUTES = 15;

// Estimated processing time per item (minutes)
// Used to calculate dynamic threshold based on task size
const ESTIMATED_MINUTES_PER_ITEM = 3;

// Maximum stale threshold to prevent unreasonably long waits (minutes)
const MAX_STALE_THRESHOLD_MINUTES = 120;

// ============================================================================
// Helper Functions
// ============================================================================

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

function formatStepName(step: string | null): string {
  if (!step) return "Processing";
  return step
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

// ============================================================================
// OperationCard Component
// ============================================================================

interface OperationCardProps {
  pipeline: ActivePipeline;
  onDismiss: (id: string, name: string) => void;
  isDismissing: boolean;
}

function OperationCard({ pipeline, onDismiss, isDismissing }: OperationCardProps) {
  const total = pipeline.totalItems ?? null;
  const processed = pipeline.processedItems ?? 0;
  const hasProgress = total !== null && total !== undefined && total > 0;
  const progressPercent = hasProgress ? Math.round((processed / total) * 100) : 0;
  
  // Track previous processed count to detect progress changes.
  // Initialize lastProgressChange from the server timestamp so that
  // a pipeline already stale on page load is detected immediately.
  const serverLastUpdate = pipeline.updatedAt || pipeline.startedAt || pipeline.createdAt;
  const serverLastUpdateMs = serverLastUpdate ? new Date(serverLastUpdate).getTime() : Date.now();
  const prevProcessedRef = useRef<number>(processed);
  const lastProgressChangeRef = useRef<number>(serverLastUpdateMs);
  
  useEffect(() => {
    if (processed !== prevProcessedRef.current) {
      lastProgressChangeRef.current = Date.now();
      prevProcessedRef.current = processed;
    }
  }, [processed]);
  
  const isStale = useMemo(() => {
    const remainingItems = total ? Math.max(0, total - processed) : 0;
    const dynamicThreshold = Math.min(
      MAX_STALE_THRESHOLD_MINUTES,
      BASE_STALE_THRESHOLD_MINUTES + (remainingItems * ESTIMATED_MINUTES_PER_ITEM / 20)
    );
    
    const minutesSinceProgressChange = (Date.now() - lastProgressChangeRef.current) / 60000;
    
    let minutesSinceServerUpdate = 0;
    if (serverLastUpdate) {
      minutesSinceServerUpdate = (Date.now() - serverLastUpdateMs) / 60000;
    }
    
    // Server-only check is sufficient: if the server timestamp is stale,
    // the pipeline is stale regardless of client-side ref timing.
    return minutesSinceServerUpdate > dynamicThreshold;
  }, [pipeline.updatedAt, pipeline.startedAt, pipeline.createdAt, serverLastUpdate, serverLastUpdateMs, total, processed]);

  return (
    <div
      className={cn(
        "p-4 rounded-lg border transition-colors group",
        "border-[var(--border-subtle)]",
        "bg-[var(--bg-surface)]"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {/* Subtle pulsing dot indicator */}
          <div className="relative w-2 h-2">
            <div className={cn(
              "absolute inset-0 rounded-full",
              isStale ? "bg-amber-500/50" : "bg-blue-500/50"
            )} />
            <div className={cn(
              "absolute inset-0 rounded-full animate-pulse-subtle",
              isStale ? "bg-amber-500" : "bg-blue-500"
            )} />
          </div>
          <div>
            <div className="text-[13px] font-medium text-[var(--text-primary)]">
              {pipeline.pipelineName || "Pipeline"}
            </div>
            {pipeline.datasetName && (
              <div className="flex items-center gap-1 text-[11px] text-[var(--text-muted)]">
                <Database size={10} />
                {pipeline.datasetName}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isStale ? (
            <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] rounded text-amber-500 bg-amber-500/10">
              <AlertTriangle size={10} />
              May be stale
            </span>
          ) : (
            <span className="px-2 py-0.5 text-[10px] rounded text-[var(--text-muted)] bg-[var(--bg-elevated)]">
              Running
            </span>
          )}
          {/* Dismiss button - always visible */}
          <button
            onClick={() => onDismiss(pipeline.pipelineRunId || pipeline.workflow_run_id, pipeline.pipelineName || pipeline.workflow_name || "Pipeline")}
            disabled={isDismissing}
            className={cn(
              "p-1 rounded transition-colors",
              "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)]",
              isDismissing && "opacity-50 cursor-not-allowed"
            )}
            title="Dismiss this pipeline"
          >
            <X size={12} />
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-2">
        <div className="h-1 bg-[var(--bg-elevated)] rounded-full overflow-hidden relative">
          {hasProgress && progressPercent > 0 ? (
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                isStale ? "bg-amber-500/60" : "bg-blue-500/60"
              )}
              style={{ width: `${progressPercent}%` }}
            />
          ) : (
            <div 
              className={cn(
                "absolute h-full rounded-full animate-progress-subtle",
                isStale ? "bg-amber-500/50" : "bg-blue-500/60"
              )}
              style={{ width: "30%" }}
            />
          )}
        </div>
      </div>

      {/* Details */}
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-[var(--text-muted)]">
          {pipeline.currentStep ? formatStepName(pipeline.currentStep) : "Initializing"}
        </span>
        <span className="text-[var(--text-muted)]">
          {hasProgress ? `${processed} / ${total}` : ""}
        </span>
      </div>

      {/* Elapsed Time */}
      {(pipeline.startedAt || pipeline.createdAt) && (
        <div className="flex items-center gap-1 mt-2 text-[10px] text-[var(--text-muted)]">
          <Clock size={10} />
          <span>Started {formatRelativeTime(pipeline.startedAt || pipeline.createdAt!)}</span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface ActiveOperationsProps {
  className?: string;
}

export function ActiveOperations({ className }: ActiveOperationsProps) {
  const { data: pipelines, isLoading } = useActivePipelines();
  const queryClient = useQueryClient();
  const [dismissingId, setDismissingId] = useState<string | null>(null);
  const confirm = useConfirm();

  const handleDismiss = async (pipelineRunId: string, pipelineName: string) => {
    const confirmed = await confirm({
      title: "Dismiss Pipeline",
      message: `Are you sure you want to dismiss "${pipelineName}"? This will mark the pipeline as terminated and remove it from the active list.`,
      confirmText: "Dismiss",
      cancelText: "Cancel",
      variant: "warning",
    });

    if (!confirmed) return;

    setDismissingId(pipelineRunId);
    try {
      const result = await apiClient.dismissPipeline(pipelineRunId);
      if (result.success) {
        toast.success("Pipeline dismissed");
        queryClient.invalidateQueries({ queryKey: ["pipelines", "active"] });
      } else {
        toast.error(result.message);
      }
    } catch (error) {
      toast.error("Failed to dismiss pipeline");
    } finally {
      setDismissingId(null);
    }
  };

  if (isLoading) {
    return null;
  }

  if (!pipelines || pipelines.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center gap-3">
        <h2 className="text-[13px] font-medium text-[var(--text-primary)]">
          Active Operations
        </h2>
        <span className="text-[11px] text-[var(--text-muted)]">
          {pipelines.length}
        </span>
      </div>
      <div className="space-y-2">
        {pipelines.map((pipeline) => (
          <OperationCard
            key={pipeline.pipelineRunId || pipeline.workflow_run_id}
            pipeline={pipeline}
            onDismiss={handleDismiss}
            isDismissing={dismissingId === (pipeline.pipelineRunId || pipeline.workflow_run_id)}
          />
        ))}
      </div>
    </div>
  );
}

ActiveOperations.displayName = "ActiveOperations";

export default ActiveOperations;
