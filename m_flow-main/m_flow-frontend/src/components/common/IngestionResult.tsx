"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { 
  CheckCircle2, 
  XCircle, 
  Loader2, 
  Database, 
  Clock, 
  FileText,
  ChevronRight,
  RotateCcw
} from "lucide-react";
import type { RunStatus } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface IngestionResultProps {
  status: "success" | "error" | "processing" | "idle";
  datasetName?: string;
  filesProcessed?: number;
  timeElapsed?: number;
  errorMessage?: string;
  onRetry?: () => void;
  onViewGraph?: () => void;
  className?: string;
}

export interface PipelineStatusProps {
  status: RunStatus;
  datasetName: string;
  className?: string;
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

function getStatusConfig(status: IngestionResultProps["status"]) {
  switch (status) {
    case "success":
      return {
        icon: CheckCircle2,
        color: "text-[var(--success)]",
        bg: "bg-[var(--success)]/10",
        border: "border-[var(--success)]/20",
        title: "Ingestion Complete",
        message: "Your data has been successfully processed and added to the knowledge graph.",
      };
    case "error":
      return {
        icon: XCircle,
        color: "text-[var(--error)]",
        bg: "bg-[var(--error)]/10",
        border: "border-[var(--error)]/20",
        title: "Ingestion Failed",
        message: "An error occurred during processing. Please check your data and try again.",
      };
    case "processing":
      return {
        icon: Loader2,
        color: "text-[var(--text-primary)]",
        bg: "bg-[var(--text-primary)]/10",
        border: "border-[var(--text-primary)]/20",
        title: "Processing...",
        message: "Building your knowledge graph. This may take a few minutes for large datasets.",
      };
    default:
      return null;
  }
}

// ============================================================================
// Components
// ============================================================================

export function IngestionResult({
  status,
  datasetName,
  filesProcessed,
  timeElapsed,
  errorMessage,
  onRetry,
  onViewGraph,
  className,
}: IngestionResultProps) {
  const config = getStatusConfig(status);
  
  if (!config || status === "idle") return null;

  const Icon = config.icon;
  const isProcessing = status === "processing";

  return (
    <div className={cn(
      "p-4 rounded-lg border",
      config.bg,
      config.border,
      className
    )}>
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className={cn(
          "p-2 rounded-full",
          config.bg
        )}>
          <Icon 
            size={20} 
            className={cn(
              config.color,
              isProcessing && "animate-spin"
            )} 
          />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className={cn(
            "text-sm font-medium",
            config.color
          )}>
            {config.title}
          </h4>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            {errorMessage || config.message}
          </p>
        </div>
      </div>

      {/* Stats */}
      {(datasetName || filesProcessed || timeElapsed) && (
        <div className="mt-3 pt-3 border-t border-[var(--border-subtle)] flex flex-wrap gap-4">
          {datasetName && (
            <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
              <Database size={12} className="text-[var(--text-muted)]" />
              <span>{datasetName}</span>
            </div>
          )}
          {filesProcessed !== undefined && filesProcessed > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
              <FileText size={12} className="text-[var(--text-muted)]" />
              <span>{filesProcessed} file(s)</span>
            </div>
          )}
          {timeElapsed !== undefined && (
            <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
              <Clock size={12} className="text-[var(--text-muted)]" />
              <span>{formatTime(timeElapsed)}</span>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      {(onRetry || onViewGraph) && status !== "processing" && (
        <div className="mt-3 pt-3 border-t border-[var(--border-subtle)] flex gap-2">
          {status === "error" && onRetry && (
            <button
              onClick={onRetry}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[var(--text-secondary)] bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded hover:border-[var(--text-muted)] transition-colors"
            >
              <RotateCcw size={12} />
              Retry
            </button>
          )}
          {status === "success" && onViewGraph && (
            <button
              onClick={onViewGraph}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[var(--bg-base)] bg-[var(--text-primary)] rounded hover:opacity-90 transition-opacity"
            >
              View Knowledge Graph
              <ChevronRight size={12} />
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Compact pipeline status indicator
 */
export function PipelineStatusBadge({ 
  status, 
  datasetName,
  className 
}: PipelineStatusProps) {
  const getStatusStyle = () => {
    switch (status) {
      case "RunCompleted":
      case "RunAlreadyCompleted":
        return {
          bg: "bg-[var(--success)]/10",
          border: "border-[var(--success)]/20",
          text: "text-[var(--success)]",
          icon: CheckCircle2,
          label: "Completed",
        };
      case "RunFailed":
        return {
          bg: "bg-[var(--error)]/10",
          border: "border-[var(--error)]/20",
          text: "text-[var(--error)]",
          icon: XCircle,
          label: "Failed",
        };
      case "RunStarted":
      case "RunYield":
      default:
        return {
          bg: "bg-[var(--text-primary)]/10",
          border: "border-[var(--text-primary)]/20",
          text: "text-[var(--text-primary)]",
          icon: Loader2,
          label: "Processing",
          animate: true,
        };
    }
  };

  const style = getStatusStyle();
  const Icon = style.icon;

  return (
    <div className={cn(
      "inline-flex items-center gap-2 px-3 py-1.5 rounded border",
      style.bg,
      style.border,
      className
    )}>
      <Icon 
        size={14} 
        className={cn(
          style.text,
          "animate" in style && style.animate && "animate-spin"
        )} 
      />
      <span className="text-xs text-[var(--text-secondary)]">{datasetName}</span>
      <span className={cn("text-[10px] font-medium uppercase", style.text)}>
        {style.label}
      </span>
    </div>
  );
}

// ============================================================================
// Error Message Helpers
// ============================================================================

export function getActionableErrorMessage(error: Error | string): string {
  const message = typeof error === "string" ? error : error.message;
  const lowerMessage = message.toLowerCase();

  if (lowerMessage.includes("timeout") || lowerMessage.includes("connection")) {
    return "Connection timeout. Please check your network and try again.";
  }

  if (lowerMessage.includes("401") || lowerMessage.includes("unauthorized")) {
    return "Authentication failed. Please login again.";
  }

  if (lowerMessage.includes("403") || lowerMessage.includes("forbidden")) {
    return "Access denied. You don't have permission for this operation.";
  }

  if (lowerMessage.includes("404") || lowerMessage.includes("not found")) {
    return "Resource not found. The dataset may have been deleted.";
  }

  if (lowerMessage.includes("413") || lowerMessage.includes("too large")) {
    return "File too large. Please reduce file size or split into smaller parts.";
  }

  if (lowerMessage.includes("500") || lowerMessage.includes("internal server")) {
    return "Server error. Please try again later or contact support.";
  }

  if (lowerMessage.includes("llm") || lowerMessage.includes("api key")) {
    return "LLM service unavailable. Please check your API key configuration.";
  }

  if (lowerMessage.includes("graph") || lowerMessage.includes("neo4j")) {
    return "Graph database connection failed. Please verify database settings.";
  }

  if (lowerMessage.includes("vector") || lowerMessage.includes("embedding")) {
    return "Vector database error. Please check embedding service configuration.";
  }

  return message || "An unexpected error occurred. Please try again.";
}
