"use client";

/**
 * Skeleton Component Library
 *
 * Provides skeleton loading states for various UI elements.
 * Used to improve perceived performance during data loading.
 *
 * Features:
 * - Multiple skeleton variants (text, card, avatar, button)
 * - Configurable animation
 * - Compound components for complex layouts
 * - Accessible loading states
 *
 * @example
 * <Skeleton.Card />
 * <Skeleton.Text lines={3} />
 * <Skeleton.Avatar size="lg" />
 */

import React from "react";
import { cn } from "@/lib/utils";

// ============================================================================
// Base Skeleton Component
// ============================================================================

export interface SkeletonProps {
  /** Width of the skeleton */
  width?: string | number;
  /** Height of the skeleton */
  height?: string | number;
  /** Border radius */
  rounded?: "none" | "sm" | "md" | "lg" | "xl" | "full";
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function Skeleton({
  width,
  height,
  rounded = "md",
  animate = true,
  className,
}: SkeletonProps) {
  const roundedClass = {
    none: "rounded-none",
    sm: "rounded-sm",
    md: "rounded-md",
    lg: "rounded-lg",
    xl: "rounded-xl",
    full: "rounded-full",
  };

  return (
    <div
      className={cn(
        "bg-zinc-800",
        animate && "animate-pulse",
        roundedClass[rounded],
        className
      )}
      style={{
        width: typeof width === "number" ? `${width}px` : width,
        height: typeof height === "number" ? `${height}px` : height,
      }}
      role="status"
      aria-label="Loading"
    />
  );
}

// ============================================================================
// Text Skeleton
// ============================================================================

export interface SkeletonTextProps {
  /** Number of lines */
  lines?: number;
  /** Line height */
  lineHeight?: number;
  /** Gap between lines */
  gap?: number;
  /** Whether last line is shorter */
  lastLineWidth?: string;
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonText({
  lines = 1,
  lineHeight = 16,
  gap = 8,
  lastLineWidth = "60%",
  animate = true,
  className,
}: SkeletonTextProps) {
  return (
    <div
      className={cn("space-y-2", className)}
      style={{ gap: `${gap}px` }}
      role="status"
      aria-label="Loading text"
    >
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton
          key={i}
          height={lineHeight}
          width={i === lines - 1 && lines > 1 ? lastLineWidth : "100%"}
          rounded="sm"
          animate={animate}
        />
      ))}
    </div>
  );
}

// ============================================================================
// Avatar Skeleton
// ============================================================================

export interface SkeletonAvatarProps {
  /** Size of the avatar */
  size?: "sm" | "md" | "lg" | "xl";
  /** Shape */
  shape?: "circle" | "square";
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonAvatar({
  size = "md",
  shape = "circle",
  animate = true,
  className,
}: SkeletonAvatarProps) {
  const sizeClass = {
    sm: "w-8 h-8",
    md: "w-10 h-10",
    lg: "w-12 h-12",
    xl: "w-16 h-16",
  };

  return (
    <Skeleton
      rounded={shape === "circle" ? "full" : "lg"}
      animate={animate}
      className={cn(sizeClass[size], className)}
    />
  );
}

// ============================================================================
// Button Skeleton
// ============================================================================

export interface SkeletonButtonProps {
  /** Size of the button */
  size?: "sm" | "md" | "lg";
  /** Width of the button */
  width?: string | number;
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonButton({
  size = "md",
  width = "auto",
  animate = true,
  className,
}: SkeletonButtonProps) {
  const sizeClass = {
    sm: "h-8 min-w-[64px]",
    md: "h-10 min-w-[80px]",
    lg: "h-12 min-w-[96px]",
  };

  return (
    <Skeleton
      width={width === "auto" ? undefined : width}
      rounded="lg"
      animate={animate}
      className={cn(sizeClass[size], className)}
    />
  );
}

// ============================================================================
// Card Skeleton
// ============================================================================

export interface SkeletonCardProps {
  /** Whether to show avatar */
  showAvatar?: boolean;
  /** Number of text lines */
  lines?: number;
  /** Whether to show action button */
  showAction?: boolean;
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonCard({
  showAvatar = false,
  lines = 2,
  showAction = false,
  animate = true,
  className,
}: SkeletonCardProps) {
  return (
    <div
      className={cn(
        "p-4 rounded-xl border border-zinc-800 bg-zinc-900/30",
        className
      )}
      role="status"
      aria-label="Loading card"
    >
      <div className="flex items-start gap-3">
        {showAvatar && <SkeletonAvatar size="md" animate={animate} />}
        <div className="flex-1 space-y-3">
          <SkeletonText lines={lines} animate={animate} />
          {showAction && (
            <div className="flex gap-2 mt-4">
              <SkeletonButton size="sm" width={80} animate={animate} />
              <SkeletonButton size="sm" width={60} animate={animate} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Status Card Skeleton (for Setup page)
// ============================================================================

export interface SkeletonStatusCardProps {
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonStatusCard({
  animate = true,
  className,
}: SkeletonStatusCardProps) {
  return (
    <div
      className={cn(
        "p-4 rounded-xl border border-zinc-800 bg-zinc-900/30",
        className
      )}
      role="status"
      aria-label="Loading status"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <Skeleton
            width={32}
            height={32}
            rounded="full"
            animate={animate}
          />
          <div className="space-y-2">
            <Skeleton width={100} height={14} rounded="sm" animate={animate} />
            <Skeleton width={80} height={10} rounded="sm" animate={animate} />
          </div>
        </div>
        <div className="text-right space-y-1">
          <Skeleton width={40} height={12} rounded="sm" animate={animate} />
          <Skeleton width={30} height={10} rounded="sm" animate={animate} />
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Grid Skeleton
// ============================================================================

export interface SkeletonGridProps {
  /** Number of items */
  count?: number;
  /** Number of columns */
  columns?: 1 | 2 | 3 | 4;
  /** Card variant */
  cardVariant?: "default" | "status";
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonGrid({
  count = 6,
  columns = 3,
  cardVariant = "default",
  animate = true,
  className,
}: SkeletonGridProps) {
  const gridClass = {
    1: "grid-cols-1",
    2: "grid-cols-1 sm:grid-cols-2",
    3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
  };

  return (
    <div className={cn("grid gap-3", gridClass[columns], className)}>
      {Array.from({ length: count }, (_, i) =>
        cardVariant === "status" ? (
          <SkeletonStatusCard key={i} animate={animate} />
        ) : (
          <SkeletonCard key={i} animate={animate} />
        )
      )}
    </div>
  );
}

// ============================================================================
// List Skeleton
// ============================================================================

export interface SkeletonListProps {
  /** Number of items */
  count?: number;
  /** Whether to show avatar */
  showAvatar?: boolean;
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonList({
  count = 5,
  showAvatar = false,
  animate = true,
  className,
}: SkeletonListProps) {
  return (
    <div className={cn("space-y-3", className)}>
      {Array.from({ length: count }, (_, i) => (
        <div
          key={i}
          className="flex items-center gap-3 p-3 rounded-lg border border-zinc-800"
        >
          {showAvatar && <SkeletonAvatar size="sm" animate={animate} />}
          <div className="flex-1 space-y-2">
            <Skeleton width="70%" height={14} rounded="sm" animate={animate} />
            <Skeleton width="50%" height={10} rounded="sm" animate={animate} />
          </div>
          <Skeleton width={60} height={24} rounded="md" animate={animate} />
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Table Skeleton
// ============================================================================

export interface SkeletonTableProps {
  /** Number of rows */
  rows?: number;
  /** Number of columns */
  columns?: number;
  /** Whether to show header */
  showHeader?: boolean;
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonTable({
  rows = 5,
  columns = 4,
  showHeader = true,
  animate = true,
  className,
}: SkeletonTableProps) {
  return (
    <div
      className={cn(
        "border border-zinc-800 rounded-xl overflow-hidden",
        className
      )}
    >
      {showHeader && (
        <div className="flex gap-4 p-4 bg-zinc-900/50 border-b border-zinc-800">
          {Array.from({ length: columns }, (_, i) => (
            <Skeleton
              key={i}
              width={`${100 / columns}%`}
              height={12}
              rounded="sm"
              animate={animate}
            />
          ))}
        </div>
      )}
      <div className="divide-y divide-zinc-800">
        {Array.from({ length: rows }, (_, rowIdx) => (
          <div key={rowIdx} className="flex gap-4 p-4">
            {Array.from({ length: columns }, (_, colIdx) => (
              <Skeleton
                key={colIdx}
                width={`${100 / columns}%`}
                height={14}
                rounded="sm"
                animate={animate}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Section Skeleton (for Setup page sections)
// ============================================================================

export interface SkeletonSectionProps {
  /** Whether to animate */
  animate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function SkeletonSection({
  animate = true,
  className,
}: SkeletonSectionProps) {
  return (
    <div className={cn("space-y-4", className)}>
      {/* Header */}
      <div className="space-y-2 pb-4 border-b border-zinc-800">
        <Skeleton width={200} height={20} rounded="sm" animate={animate} />
        <Skeleton width={300} height={14} rounded="sm" animate={animate} />
      </div>

      {/* Content */}
      <SkeletonGrid count={6} columns={3} cardVariant="status" animate={animate} />
    </div>
  );
}

// ============================================================================
// Compound Component
// ============================================================================

export const SkeletonCompound = Object.assign(Skeleton, {
  Text: SkeletonText,
  Avatar: SkeletonAvatar,
  Button: SkeletonButton,
  Card: SkeletonCard,
  StatusCard: SkeletonStatusCard,
  Grid: SkeletonGrid,
  List: SkeletonList,
  Table: SkeletonTable,
  Section: SkeletonSection,
});

// ============================================================================
// Display Names
// ============================================================================

Skeleton.displayName = "Skeleton";
SkeletonText.displayName = "Skeleton.Text";
SkeletonAvatar.displayName = "Skeleton.Avatar";
SkeletonButton.displayName = "Skeleton.Button";
SkeletonCard.displayName = "Skeleton.Card";
SkeletonStatusCard.displayName = "Skeleton.StatusCard";
SkeletonGrid.displayName = "Skeleton.Grid";
SkeletonList.displayName = "Skeleton.List";
SkeletonTable.displayName = "Skeleton.Table";
SkeletonSection.displayName = "Skeleton.Section";

// ============================================================================
// Default Export
// ============================================================================

export default SkeletonCompound;
