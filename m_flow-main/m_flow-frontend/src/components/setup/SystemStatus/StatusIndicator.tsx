"use client";

/**
 * StatusIndicator Component
 * 
 * A small visual indicator showing health status.
 * Used within StatusCard and other components to show service state.
 * 
 * @example
 * // Simple usage
 * <StatusIndicator verdict="up" />
 * 
 * @example
 * // With animation
 * <StatusIndicator verdict="up" animate />
 * 
 * @example
 * // Different sizes
 * <StatusIndicator verdict="warn" size="lg" />
 */

import React from "react";
import { cn } from "@/lib/utils";
import type { HealthVerdict } from "@/types";

// ============================================================================
// Types
// ============================================================================

export type IndicatorSize = "sm" | "md" | "lg";

export interface StatusIndicatorProps {
  /** Health verdict to display */
  verdict: HealthVerdict;
  /** Size of the indicator */
  size?: IndicatorSize;
  /** Whether to show pulse animation for 'up' status */
  animate?: boolean;
  /** Whether currently loading/testing */
  loading?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Constants
// ============================================================================

/** Size mappings for the indicator */
const SIZE_CLASSES: Record<IndicatorSize, string> = {
  sm: "w-2 h-2",
  md: "w-3 h-3",
  lg: "w-4 h-4",
};

/** Color mappings for each verdict */
const VERDICT_COLORS: Record<HealthVerdict, string> = {
  up: "bg-emerald-500",
  warn: "bg-amber-500",
  down: "bg-red-500",
};

/** Glow effect for each verdict */
const VERDICT_GLOW: Record<HealthVerdict, string> = {
  up: "shadow-emerald-500/50",
  warn: "shadow-amber-500/50",
  down: "shadow-red-500/50",
};

// ============================================================================
// Component
// ============================================================================

export function StatusIndicator({
  verdict,
  size = "md",
  animate = false,
  loading = false,
  className,
}: StatusIndicatorProps) {
  // Loading state - show spinner
  if (loading) {
    return (
      <div
        className={cn(
          "rounded-full border-2 border-zinc-600 border-t-zinc-300 animate-spin",
          SIZE_CLASSES[size],
          className
        )}
        role="status"
        aria-label="Loading"
      />
    );
  }

  return (
    <div className={cn("relative", className)}>
      {/* Main indicator dot */}
      <div
        className={cn(
          "rounded-full shadow-sm",
          SIZE_CLASSES[size],
          VERDICT_COLORS[verdict],
          animate && verdict === "up" && "shadow-lg",
          animate && verdict === "up" && VERDICT_GLOW[verdict]
        )}
        role="status"
        aria-label={`Status: ${verdict}`}
      />
      
      {/* Pulse animation for healthy status */}
      {animate && verdict === "up" && (
        <div
          className={cn(
            "absolute inset-0 rounded-full animate-ping",
            VERDICT_COLORS[verdict],
            "opacity-75"
          )}
          style={{ animationDuration: "2s" }}
        />
      )}
    </div>
  );
}

// ============================================================================
// Display Name for DevTools
// ============================================================================

StatusIndicator.displayName = "StatusIndicator";

// ============================================================================
// Default Export
// ============================================================================

export default StatusIndicator;
