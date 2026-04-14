"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { 
  Loader2, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  Wifi, 
  WifiOff,
  RefreshCw,
  Sparkles
} from "lucide-react";
import type { RunStatus } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface ProgressTrackerProps {
  /** Pipeline run ID being tracked */
  pipelineRunId?: string | null;
  /** Current status */
  status: RunStatus | null;
  /** Connection state */
  connectionState: "idle" | "connecting" | "connected" | "disconnected" | "error" | "auth_failed";
  /** Error message */
  error?: Error | null;
  /** Dataset name being processed */
  datasetName?: string;
  /** Show compact version */
  compact?: boolean;
  /** Callback when retry is clicked */
  onRetry?: () => void;
  /** Callback when cancel is clicked */
  onCancel?: () => void;
  /** Additional class name */
  className?: string;
}

// ============================================================================
// Status Configuration
// ============================================================================

interface StatusConfig {
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
}

const STATUS_CONFIG: Record<RunStatus, StatusConfig & { progress?: number }> = {
  RunStarted: {
    label: "Starting",
    description: "Initializing pipeline...",
    icon: <Loader2 size={16} className="animate-spin" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    progress: 10,
  },
  RunYield: {
    label: "Processing",
    description: "Extracting entities and relationships...",
    icon: <Loader2 size={16} className="animate-spin" />,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    progress: 50,
  },
  RunCompleted: {
    label: "Completed",
    description: "Knowledge graph built successfully",
    icon: <CheckCircle2 size={16} />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    progress: 100,
  },
  RunAlreadyCompleted: {
    label: "Already Done",
    description: "Using cached results",
    icon: <CheckCircle2 size={16} />,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    progress: 100,
  },
  RunFailed: {
    label: "Failed",
    description: "Processing encountered an error",
    icon: <XCircle size={16} />,
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    progress: 0,
  },
};

// ============================================================================
// Connection State Display
// ============================================================================

function ConnectionIndicator({ 
  state, 
  compact 
}: { 
  state: ProgressTrackerProps["connectionState"];
  compact?: boolean;
}) {
  const config = {
    idle: { icon: <Clock size={12} />, label: "Waiting", color: "text-[var(--text-muted)]" },
    connecting: { icon: <Loader2 size={12} className="animate-spin" />, label: "Connecting", color: "text-[var(--text-muted)]" },
    connected: { icon: <Wifi size={12} />, label: "Connected", color: "text-green-400" },
    disconnected: { icon: <WifiOff size={12} />, label: "Disconnected", color: "text-[var(--text-muted)]" },
    error: { icon: <WifiOff size={12} />, label: "Error", color: "text-red-400" },
    auth_failed: { icon: <WifiOff size={12} />, label: "Auth Failed", color: "text-red-400" },
  };

  const c = config[state];

  if (compact) {
    return (
      <span className={cn("flex items-center gap-1", c.color)}>
        {c.icon}
      </span>
    );
  }

  return (
    <span className={cn("flex items-center gap-1.5 text-[10px]", c.color)}>
      {c.icon}
      <span>{c.label}</span>
    </span>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ProgressTracker({
  pipelineRunId,
  status,
  connectionState,
  error,
  datasetName,
  compact = false,
  onRetry,
  onCancel,
  className,
}: ProgressTrackerProps) {
  // Get status configuration
  const statusConfig = status ? STATUS_CONFIG[status] : null;
  
  // Is completed or errored?
  const isFinished = status === "RunCompleted" || 
                     status === "RunAlreadyCompleted" ||
                     status === "RunFailed";
  
  const hasError = status === "RunFailed" || connectionState === "error" || connectionState === "auth_failed";

  // Compact version
  if (compact) {
    if (!pipelineRunId && connectionState === "idle") {
      return null;
    }

    return (
      <div className={cn(
        "flex items-center gap-2 px-2 py-1 rounded text-xs",
        statusConfig?.bgColor || "bg-[var(--bg-surface)]",
        className
      )}>
        {statusConfig ? (
          <>
            <span className={statusConfig.color}>{statusConfig.icon}</span>
            <span className="text-[var(--text-secondary)]">{statusConfig.label}</span>
          </>
        ) : (
          <>
            <ConnectionIndicator state={connectionState} compact />
            <span className="text-[var(--text-muted)]">
              {connectionState === "connecting" ? "Connecting..." : "Waiting..."}
            </span>
          </>
        )}
      </div>
    );
  }

  // Full version
  return (
    <div className={cn(
      "p-4 rounded-lg border transition-colors",
      hasError 
        ? "border-red-500/30 bg-red-500/5" 
        : isFinished 
          ? "border-green-500/30 bg-green-500/5"
          : "border-[var(--border-subtle)] bg-[var(--bg-surface)]",
      className
    )}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {/* Status Icon */}
          <div className={cn(
            "w-8 h-8 rounded-full flex items-center justify-center",
            statusConfig?.bgColor || "bg-[var(--bg-elevated)]"
          )}>
            {statusConfig ? (
              <span className={statusConfig.color}>{statusConfig.icon}</span>
            ) : (
              <Loader2 size={16} className="text-[var(--text-muted)] animate-spin" />
            )}
          </div>

          {/* Status Text */}
          <div>
            <p className="text-sm font-medium text-[var(--text-primary)]">
              {statusConfig?.label || "Initializing"}
            </p>
            <p className="text-xs text-[var(--text-muted)]">
              {statusConfig?.description || "Connecting to server..."}
            </p>
          </div>
        </div>

        {/* Connection Indicator */}
        <ConnectionIndicator state={connectionState} />
      </div>

      {/* Dataset Info */}
      {datasetName && (
        <div className="mb-3 text-xs text-[var(--text-muted)]">
          Dataset: <span className="text-[var(--text-secondary)]">{datasetName}</span>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-2 mb-3 bg-red-500/10 border border-red-500/20 rounded text-xs text-red-400">
          {error.message}
        </div>
      )}

      {/* Progress Bar with Percentage */}
      {status && (
        <div className="space-y-1">
          {!isFinished && (
            <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)]">
              <span>Progress</span>
              <span>{statusConfig?.progress ?? 0}%</span>
            </div>
          )}
          <div className="h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
            {isFinished ? (
              <motion.div
                className={cn(
                  "h-full rounded-full",
                  hasError ? "bg-red-500" : "bg-green-500"
                )}
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ duration: 0.3 }}
              />
            ) : (
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 via-blue-400 to-blue-500 rounded-full"
                style={{ backgroundSize: "200% 100%", width: `${statusConfig?.progress ?? 0}%` }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1, backgroundPosition: ["0% 0%", "200% 0%"] }}
                transition={{
                  opacity: { duration: 0.2 },
                  backgroundPosition: {
                    repeat: Infinity,
                    duration: 1.5,
                    ease: "linear",
                  },
                }}
              />
            )}
          </div>
        </div>
      )}
      
      {/* Success animation */}
      <AnimatePresence>
        {status === "RunCompleted" && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-2 mt-2"
          >
            {[...Array(3)].map((_, i) => (
              <motion.div
                key={i}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: [0, 1.2, 1], opacity: [0, 1, 0.7] }}
                transition={{ delay: 0.1 + i * 0.15, duration: 0.4 }}
              >
                <Sparkles size={12} className="text-green-400" />
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Actions */}
      {(hasError || isFinished) && (onRetry || onCancel) && (
        <div className="flex items-center gap-2 mt-3 pt-3 border-t border-[var(--border-subtle)]">
          {hasError && onRetry && (
            <button
              onClick={onRetry}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[var(--text-secondary)] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded hover:border-[var(--text-muted)] transition-colors"
            >
              <RefreshCw size={12} />
              Retry
            </button>
          )}
          {onCancel && !isFinished && (
            <button
              onClick={onCancel}
              className="px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      )}

      {/* Pipeline Run ID (for debugging) */}
      {pipelineRunId && (
        <div className="mt-3 pt-3 border-t border-[var(--border-subtle)]">
          <p className="text-[10px] text-[var(--text-muted)] font-mono truncate">
            ID: {pipelineRunId}
          </p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Multi-Pipeline Progress Tracker
// ============================================================================

export interface MultiProgressTrackerProps {
  /** Array of pipeline progress states */
  pipelines: Array<{
    datasetName: string;
    pipelineRunId: string;
    status: RunStatus | null;
    isActive?: boolean;
  }>;
  /** Connection state */
  connectionState: "idle" | "connecting" | "connected" | "disconnected" | "error" | "auth_failed";
  /** Error message */
  error?: Error | null;
  /** Callback when retry is clicked */
  onRetry?: () => void;
  /** Callback when cancel is clicked */
  onCancel?: () => void;
  /** Additional class name */
  className?: string;
}

export function MultiProgressTracker({
  pipelines,
  connectionState,
  error,
  onRetry,
  onCancel,
  className,
}: MultiProgressTrackerProps) {
  const completedCount = pipelines.filter(
    p => p.status === "RunCompleted" || p.status === "RunAlreadyCompleted"
  ).length;

  const errorCount = pipelines.filter(
    p => p.status === "RunFailed"
  ).length;

  const hasError = errorCount > 0 || connectionState === "error" || connectionState === "auth_failed";
  const allCompleted = completedCount === pipelines.length;

  return (
    <div className={cn(
      "p-4 rounded-lg border transition-colors",
      hasError 
        ? "border-red-500/30 bg-red-500/5" 
        : allCompleted 
          ? "border-green-500/30 bg-green-500/5"
          : "border-[var(--border-subtle)] bg-[var(--bg-surface)]",
      className
    )}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-[var(--text-primary)]">
          Processing {pipelines.length} dataset(s)
        </span>
        <ConnectionIndicator state={connectionState} />
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-2 mb-3 bg-red-500/10 border border-red-500/20 rounded text-xs text-red-400">
          {error.message}
        </div>
      )}

      {/* Progress Items */}
      <div className="space-y-1">
        {pipelines.map((pipeline) => {
          const config = pipeline.status ? STATUS_CONFIG[pipeline.status] : null;
          
          return (
            <div 
              key={pipeline.pipelineRunId}
              className={cn(
                "flex items-center gap-2 p-2 rounded border transition-colors",
                pipeline.isActive 
                  ? "bg-[var(--bg-elevated)] border-[var(--border-default)]" 
                  : "bg-[var(--bg-surface)] border-[var(--border-subtle)]"
              )}
            >
              <span className={config?.color || "text-[var(--text-muted)]"}>
                {config?.icon || <Loader2 size={12} className="animate-spin" />}
              </span>
              <span className="flex-1 text-xs text-[var(--text-secondary)] truncate">
                {pipeline.datasetName}
              </span>
              <span className="text-[10px] text-[var(--text-muted)]">
                {config?.label || "Waiting"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Summary Stats */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-[var(--border-subtle)]">
        <div className="flex items-center gap-4 text-[10px]">
          {completedCount > 0 && (
            <span className="text-green-400">
              {completedCount} completed
            </span>
          )}
          {errorCount > 0 && (
            <span className="text-red-400">
              {errorCount} failed
            </span>
          )}
          {pipelines.length - completedCount - errorCount > 0 && (
            <span className="text-[var(--text-muted)]">
              {pipelines.length - completedCount - errorCount} processing
            </span>
          )}
        </div>

        {/* Actions */}
        {(hasError || onCancel) && (
          <div className="flex items-center gap-2">
            {hasError && onRetry && (
              <button
                onClick={onRetry}
                className="flex items-center gap-1.5 px-2 py-1 text-[10px] text-[var(--text-secondary)] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded hover:border-[var(--text-muted)] transition-colors"
              >
                <RefreshCw size={10} />
                Retry
              </button>
            )}
            {onCancel && !allCompleted && (
              <button
                onClick={onCancel}
                className="px-2 py-1 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
