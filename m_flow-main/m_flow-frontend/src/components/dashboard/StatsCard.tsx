"use client";

/**
 * StatsCard Component
 *
 * Displays a single statistic with optional trend indicator.
 * Used in Dashboard for key metrics display.
 *
 * Features:
 * - Large numeric display
 * - Trend indicator (up/down/neutral)
 * - Loading skeleton
 * - Optional click handler
 */

import React from "react";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface StatsCardProps {
  label: string;
  value: number | string;
  trend?: {
    value: number;
    direction: "up" | "down" | "neutral";
    label?: string;
  };
  icon?: React.ReactNode;
  loading?: boolean;
  onClick?: () => void;
  className?: string;
}

// ============================================================================
// Trend Indicator Component
// ============================================================================

interface TrendIndicatorProps {
  value: number;
  direction: "up" | "down" | "neutral";
  label?: string;
}

function TrendIndicator({ value, direction, label }: TrendIndicatorProps) {
  const isPositive = direction === "up";
  const isNegative = direction === "down";
  const isNeutral = direction === "neutral";

  return (
    <div
      className={cn(
        "flex items-center gap-1 text-[11px] font-medium",
        isPositive && "text-emerald-400",
        isNegative && "text-red-400",
        isNeutral && "text-[var(--text-muted)]"
      )}
    >
      {isPositive && <TrendingUp size={12} />}
      {isNegative && <TrendingDown size={12} />}
      {isNeutral && <Minus size={12} />}
      <span>
        {isPositive && "+"}
        {value}
        {label && ` ${label}`}
      </span>
    </div>
  );
}

// ============================================================================
// Loading Skeleton
// ============================================================================

function StatsCardSkeleton() {
  return (
    <div className="animate-pulse">
      <div className="h-3 w-16 rounded bg-[var(--bg-elevated)] mb-2" />
      <div className="h-8 w-20 rounded bg-[var(--bg-elevated)] mb-1" />
      <div className="h-3 w-12 rounded bg-[var(--bg-elevated)]" />
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function StatsCard({
  label,
  value,
  trend,
  icon,
  loading = false,
  onClick,
  className,
}: StatsCardProps) {
  const isClickable = !!onClick;

  if (loading) {
    return (
      <div className={cn("", className)}>
        <StatsCardSkeleton />
      </div>
    );
  }

  const content = (
    <>
      {/* Label */}
      <div className="flex items-center gap-2 mb-1">
        {icon && (
          <span className="text-[var(--text-muted)]">{icon}</span>
        )}
        <p className="text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-medium">
          {label}
        </p>
      </div>

      {/* Value */}
      <p className="text-[28px] font-semibold text-[var(--text-primary)] tracking-tight leading-none mb-1">
        {typeof value === "number" ? value.toLocaleString() : value}
      </p>

      {/* Trend */}
      {trend && (
        <TrendIndicator
          value={trend.value}
          direction={trend.direction}
          label={trend.label}
        />
      )}
    </>
  );

  if (isClickable) {
    return (
      <button
        onClick={onClick}
        className={cn(
          "text-left transition-colors rounded-lg p-3 -m-3",
          "hover:bg-[var(--bg-hover)]",
          "focus:outline-none focus:ring-1 focus:ring-[var(--border-default)]",
          className
        )}
      >
        {content}
      </button>
    );
  }

  return <div className={cn("", className)}>{content}</div>;
}

StatsCard.displayName = "StatsCard";

// ============================================================================
// Stats Grid Component
// ============================================================================

export interface StatsGridProps {
  children: React.ReactNode;
  columns?: 2 | 3 | 4;
  className?: string;
}

export function StatsGrid({ children, columns = 4, className }: StatsGridProps) {
  return (
    <div
      className={cn(
        "grid gap-6",
        columns === 2 && "grid-cols-2",
        columns === 3 && "grid-cols-3",
        columns === 4 && "grid-cols-2 sm:grid-cols-4",
        className
      )}
    >
      {children}
    </div>
  );
}

StatsGrid.displayName = "StatsGrid";

export default StatsCard;
