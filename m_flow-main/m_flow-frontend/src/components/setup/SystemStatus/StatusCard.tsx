"use client";

/**
 * StatusCard Component
 * 
 * Displays the health status of a single system component.
 * Compact card that can be clicked to show full details in a modal.
 */

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { 
  PROBE_METADATA, 
  getVerdictDisplayConfig, 
  formatLatency,
} from "@/lib/utils/health";
import { StatusIndicator } from "./StatusIndicator";
import { X, Clock, Server, Cpu, Database, HardDrive, Brain, Sparkles } from "lucide-react";
import type { ProbeKey, ProbeResult } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface StatusCardProps {
  /** Service key from the probes object */
  serviceKey: ProbeKey;
  /** Probe result from health check */
  probe: ProbeResult;
  /** Whether to show expanded details */
  showDetails?: boolean;
  /** Whether the card is currently being tested */
  isTesting?: boolean;
  /** Click handler */
  onClick?: () => void;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Component
// ============================================================================

export function StatusCard({
  serviceKey,
  probe,
  showDetails = false,
  isTesting = false,
  onClick,
  className,
}: StatusCardProps) {
  // Get metadata for this service
  const metadata = PROBE_METADATA[serviceKey];
  const displayConfig = getVerdictDisplayConfig(probe.verdict);

  // Determine if this is a clickable card
  const isClickable = !!onClick;

  return (
    <div
      className={cn(
        // Base styles - compact single-row layout
        "relative px-3 py-2.5 rounded-lg border transition-all duration-200",
        // Border color based on verdict
        probe.verdict === "up" && "border-zinc-800 hover:border-zinc-700",
        probe.verdict === "warn" && "border-amber-900/50 bg-amber-950/10",
        probe.verdict === "down" && "border-red-900/50 bg-red-950/10",
        // Clickable styles
        isClickable && "cursor-pointer hover:bg-zinc-900/50",
        // Additional classes
        className
      )}
      onClick={onClick}
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
      onKeyDown={
        isClickable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
    >
      {/* Single Row Layout: Name | Status | Latency */}
      <div className="flex items-center justify-between gap-2">
        {/* Left: Service name with indicator */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <StatusIndicator
            verdict={probe.verdict}
            size="sm"
            animate={probe.verdict === "up"}
            loading={isTesting}
          />
          <span className="text-[12px] font-medium text-zinc-200 truncate">
            {metadata.displayName}
          </span>
        </div>

        {/* Right: Status and latency inline */}
        <div className="flex items-center gap-2 shrink-0">
          <span
            className={cn(
              "text-[11px] font-medium",
              probe.verdict === "up" && "text-emerald-400",
              probe.verdict === "warn" && "text-amber-400",
              probe.verdict === "down" && "text-red-400"
            )}
          >
            {displayConfig.label}
          </span>
          {probe.latency_ms > 0 && (
            <span className="text-[10px] text-zinc-500">
              {formatLatency(probe.latency_ms)}
            </span>
          )}
        </div>
      </div>

      {/* Backend info - only show if available and not "unknown" */}
      {probe.backend && probe.backend !== "unknown" && (
        <p className="text-[10px] text-zinc-500 mt-1 ml-5 truncate">
          {probe.backend}
        </p>
      )}

      {/* Details Row (optional) */}
      {showDetails && probe.note && (
        <div className="mt-2 pt-2 border-t border-zinc-800">
          <p className="text-[10px] text-zinc-500 leading-relaxed">
            {probe.note.length > 80
              ? `${probe.note.slice(0, 80)}...`
              : probe.note}
          </p>
        </div>
      )}

      {/* Critical badge for critical services */}
      {metadata.isCritical && probe.verdict === "down" && (
        <div className="absolute -top-1 -right-1">
          <span className="flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
          </span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Skeleton Component for Loading State
// ============================================================================

export function StatusCardSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "px-3 py-2.5 rounded-lg border border-zinc-800 animate-pulse",
        className
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {/* Indicator skeleton */}
          <div className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
          {/* Name skeleton */}
          <div className="h-3 w-24 bg-zinc-700 rounded" />
        </div>
        <div className="flex items-center gap-2">
          {/* Status skeleton */}
          <div className="h-3 w-14 bg-zinc-700 rounded" />
          {/* Latency skeleton */}
          <div className="h-2.5 w-8 bg-zinc-800 rounded" />
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Service Icons - Compact size for modal
// ============================================================================

const SERVICE_ICONS: Record<ProbeKey, React.ReactNode> = {
  relational_db: <Database size={16} />,
  vector_db: <Cpu size={16} />,
  graph_db: <Server size={16} />,
  file_storage: <HardDrive size={16} />,
  llm_provider: <Brain size={16} />,
  embedding_service: <Sparkles size={16} />,
};

// ============================================================================
// Status Detail Modal Component - M-Flow Design System
// ============================================================================

export interface StatusDetailModalProps {
  serviceKey: ProbeKey;
  probe: ProbeResult;
  isOpen: boolean;
  onClose: () => void;
}

export function StatusDetailModal({
  serviceKey,
  probe,
  isOpen,
  onClose,
}: StatusDetailModalProps) {
  const metadata = PROBE_METADATA[serviceKey];
  const displayConfig = getVerdictDisplayConfig(probe.verdict);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
          onClick={onClose}
        >
          {/* Backdrop - pure black with opacity */}
          <div className="absolute inset-0 bg-[var(--bg-base)]/80 backdrop-blur-sm" />
          
          {/* Modal - M-Flow style */}
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.15 }}
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-xs mx-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg shadow-2xl overflow-hidden"
          >
            {/* Header - Minimal */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-subtle)]">
              <div className="flex items-center gap-2.5">
                <div
                  className={cn(
                    "flex items-center justify-center w-7 h-7 rounded",
                    probe.verdict === "up" && "bg-[var(--success)]/10 text-[var(--success)]",
                    probe.verdict === "warn" && "bg-[var(--warning)]/10 text-[var(--warning)]",
                    probe.verdict === "down" && "bg-[var(--error)]/10 text-[var(--error)]"
                  )}
                >
                  {SERVICE_ICONS[serviceKey]}
                </div>
                <div>
                  <h3 className="text-[13px] font-medium text-[var(--text-primary)]">
                    {metadata.displayName}
                  </h3>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-1 rounded hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            {/* Content - Compact rows */}
            <div className="px-4 py-3 space-y-2.5">
              {/* Description */}
              <p className="text-[11px] text-[var(--text-tertiary)] leading-relaxed">
                {metadata.description}
              </p>

              {/* Divider */}
              <div className="border-t border-[var(--border-subtle)]" />

              {/* Status Row */}
              <div className="flex items-center justify-between py-1">
                <span className="text-[11px] text-[var(--text-muted)]">Status</span>
                <div className="flex items-center gap-1.5">
                  <StatusIndicator verdict={probe.verdict} size="sm" animate />
                  <span
                    className={cn(
                      "text-[11px] font-medium",
                      probe.verdict === "up" && "text-[var(--success)]",
                      probe.verdict === "warn" && "text-[var(--warning)]",
                      probe.verdict === "down" && "text-[var(--error)]"
                    )}
                  >
                    {displayConfig.label}
                  </span>
                </div>
              </div>

              {/* Backend Row */}
              <div className="flex items-center justify-between py-1">
                <span className="text-[11px] text-[var(--text-muted)]">Backend</span>
                <span className="text-[11px] text-[var(--text-secondary)]">
                  {probe.backend !== "unknown" ? probe.backend : "—"}
                </span>
              </div>

              {/* Latency Row */}
              {probe.latency_ms > 0 && (
                <div className="flex items-center justify-between py-1">
                  <span className="text-[11px] text-[var(--text-muted)]">Latency</span>
                  <span className="text-[11px] text-[var(--text-secondary)]">
                    {formatLatency(probe.latency_ms)}
                  </span>
                </div>
              )}

              {/* Critical Badge */}
              {metadata.isCritical && (
                <div className="flex items-center justify-between py-1">
                  <span className="text-[11px] text-[var(--text-muted)]">Priority</span>
                  <span className="px-1.5 py-0.5 text-[9px] font-medium bg-[var(--warning)]/10 text-[var(--warning)] rounded">
                    Critical
                  </span>
                </div>
              )}

              {/* Note */}
              {probe.note && (
                <>
                  <div className="border-t border-[var(--border-subtle)]" />
                  <div className="py-1">
                    <p className="text-[10px] text-[var(--text-muted)] mb-1">Note</p>
                    <p className="text-[11px] text-[var(--text-tertiary)] leading-relaxed">
                      {probe.note}
                    </p>
                  </div>
                </>
              )}
            </div>

            {/* Footer - Minimal */}
            <div className="px-4 py-2.5 border-t border-[var(--border-subtle)] bg-[var(--bg-base)]">
              <button
                onClick={onClose}
                className="w-full py-1.5 text-[11px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] border border-[var(--border-subtle)] rounded transition-colors"
              >
                Close
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ============================================================================
// Display Names for DevTools
// ============================================================================

StatusCard.displayName = "StatusCard";
StatusCardSkeleton.displayName = "StatusCardSkeleton";
StatusDetailModal.displayName = "StatusDetailModal";

// ============================================================================
// Default Export
// ============================================================================

export default StatusCard;
