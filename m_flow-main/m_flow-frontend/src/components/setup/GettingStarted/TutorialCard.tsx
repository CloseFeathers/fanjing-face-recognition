"use client";

/**
 * TutorialCard Component
 *
 * Displays a tutorial or guide card with icon, title, description,
 * and metadata like duration and difficulty level.
 *
 * Features:
 * - Dynamic icon rendering
 * - Difficulty badge
 * - Duration display
 * - Interactive/external link handling
 * - Hover animations
 *
 * @example
 * <TutorialCard
 *   tutorial={tutorialData}
 *   onClick={() => handleSelect(tutorial.id)}
 * />
 */

import React from "react";
import { cn } from "@/lib/utils";
import {
  Clock,
  ExternalLink,
  Sparkles,
  Search,
  Code,
  Globe,
  Settings,
  Database,
  GitBranch,
  Layers,
  Zap,
  GraduationCap,
  ChevronRight,
  Play,
  type LucideIcon,
} from "lucide-react";
import type { TutorialOption } from "@/types/setup";

// ============================================================================
// Types
// ============================================================================

export interface TutorialCardProps {
  /** Tutorial data */
  tutorial: TutorialOption;
  /** Click handler */
  onClick?: () => void;
  /** Whether this card is selected */
  isSelected?: boolean;
  /** Card variant */
  variant?: "default" | "compact";
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Icon Mapping
// ============================================================================

const ICON_MAP: Record<string, LucideIcon> = {
  Sparkles,
  Search,
  Code,
  Globe,
  Settings,
  Database,
  GitBranch,
  Layers,
  Zap,
  GraduationCap,
};

function getIcon(iconName?: string) {
  if (!iconName) return Code;
  return ICON_MAP[iconName] || Code;
}

// ============================================================================
// Difficulty Badge Component
// ============================================================================

interface DifficultyBadgeProps {
  difficulty: TutorialOption["difficulty"];
}

function DifficultyBadge({ difficulty }: DifficultyBadgeProps) {
  const config = {
    beginner: {
      label: "Beginner",
      bgClass: "bg-emerald-500/10",
      textClass: "text-emerald-400",
      borderClass: "border-emerald-500/30",
    },
    intermediate: {
      label: "Intermediate",
      bgClass: "bg-amber-500/10",
      textClass: "text-amber-400",
      borderClass: "border-amber-500/30",
    },
    advanced: {
      label: "Advanced",
      bgClass: "bg-purple-500/10",
      textClass: "text-purple-400",
      borderClass: "border-purple-500/30",
    },
  };

  const c = config[difficulty];

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium border",
        c.bgClass,
        c.textClass,
        c.borderClass
      )}
    >
      {c.label}
    </span>
  );
}

// ============================================================================
// Duration Display Component
// ============================================================================

interface DurationDisplayProps {
  minutes: number;
}

function DurationDisplay({ minutes }: DurationDisplayProps) {
  return (
    <span className="inline-flex items-center gap-1 text-[11px] text-zinc-500">
      <Clock size={12} />
      {minutes} min
    </span>
  );
}

// ============================================================================
// Main Component (Default Variant)
// ============================================================================

function TutorialCardDefault({
  tutorial,
  onClick,
  isSelected,
  className,
}: TutorialCardProps) {
  const Icon = getIcon(tutorial.icon);
  const isExternal = !!tutorial.docLink?.startsWith("http");

  return (
    <div
      className={cn(
        "group relative p-4 rounded-xl border transition-all duration-200",
        "bg-zinc-900/30 hover:bg-zinc-900/50",
        isSelected
          ? "border-zinc-100 ring-1 ring-zinc-100"
          : "border-zinc-800 hover:border-zinc-700",
        (onClick || isExternal) && "cursor-pointer",
        className
      )}
      onClick={() => {
        if (onClick) {
          onClick();
        } else if (isExternal && tutorial.docLink) {
          window.open(tutorial.docLink, "_blank", "noopener,noreferrer");
        }
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (onClick) {
            onClick();
          } else if (isExternal && tutorial.docLink) {
            window.open(tutorial.docLink, "_blank", "noopener,noreferrer");
          }
        }
      }}
    >
      {/* Icon */}
      <div
        className={cn(
          "w-10 h-10 rounded-lg flex items-center justify-center mb-3",
          "bg-zinc-800 group-hover:bg-zinc-700 transition-colors"
        )}
      >
        <Icon size={20} className="text-zinc-400 group-hover:text-zinc-200" />
      </div>

      {/* Title */}
      <h3 className="text-[14px] font-semibold text-zinc-100 mb-1 group-hover:text-white">
        {tutorial.title}
        {isExternal && (
          <ExternalLink size={12} className="inline ml-1.5 text-zinc-500" />
        )}
      </h3>

      {/* Description */}
      <p className="text-[12px] text-zinc-500 mb-3 line-clamp-2">
        {tutorial.description}
      </p>

      {/* Footer: Metadata */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <DifficultyBadge difficulty={tutorial.difficulty} />
          <DurationDisplay minutes={tutorial.durationMinutes} />
        </div>

        {/* Interactive indicator */}
        {tutorial.isInteractive && (
          <span className="flex items-center gap-1 text-[10px] text-emerald-400">
            <Play size={10} />
            Interactive
          </span>
        )}
      </div>

      {/* Hover arrow */}
      <div
        className={cn(
          "absolute right-3 top-1/2 -translate-y-1/2",
          "opacity-0 group-hover:opacity-100 transition-opacity"
        )}
      >
        <ChevronRight size={16} className="text-zinc-500" />
      </div>
    </div>
  );
}

// ============================================================================
// Compact Variant
// ============================================================================

function TutorialCardCompact({
  tutorial,
  onClick,
  isSelected,
  className,
}: TutorialCardProps) {
  const Icon = getIcon(tutorial.icon);
  const isExternal = !!tutorial.docLink?.startsWith("http");

  return (
    <div
      className={cn(
        "group flex items-center gap-3 p-3 rounded-lg border transition-all duration-200",
        "bg-zinc-900/30 hover:bg-zinc-900/50",
        isSelected
          ? "border-zinc-100"
          : "border-zinc-800 hover:border-zinc-700",
        (onClick || isExternal) && "cursor-pointer",
        className
      )}
      onClick={() => {
        if (onClick) {
          onClick();
        } else if (isExternal && tutorial.docLink) {
          window.open(tutorial.docLink, "_blank", "noopener,noreferrer");
        }
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (onClick) {
            onClick();
          } else if (isExternal && tutorial.docLink) {
            window.open(tutorial.docLink, "_blank", "noopener,noreferrer");
          }
        }
      }}
    >
      {/* Icon */}
      <div
        className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
          "bg-zinc-800 group-hover:bg-zinc-700 transition-colors"
        )}
      >
        <Icon size={16} className="text-zinc-400 group-hover:text-zinc-200" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <h3 className="text-[13px] font-medium text-zinc-200 truncate group-hover:text-white">
          {tutorial.title}
          {isExternal && (
            <ExternalLink size={10} className="inline ml-1 text-zinc-500" />
          )}
        </h3>
        <p className="text-[11px] text-zinc-500 truncate">
          {tutorial.description}
        </p>
      </div>

      {/* Metadata */}
      <div className="flex items-center gap-2 shrink-0">
        <DurationDisplay minutes={tutorial.durationMinutes} />
        <ChevronRight
          size={14}
          className="text-zinc-600 group-hover:text-zinc-400 transition-colors"
        />
      </div>
    </div>
  );
}

// ============================================================================
// Main Export
// ============================================================================

export function TutorialCard(props: TutorialCardProps) {
  if (props.variant === "compact") {
    return <TutorialCardCompact {...props} />;
  }
  return <TutorialCardDefault {...props} />;
}

// ============================================================================
// Tutorial Grid Component
// ============================================================================

export interface TutorialGridProps {
  tutorials: TutorialOption[];
  onSelect?: (tutorial: TutorialOption) => void;
  selectedId?: string;
  variant?: "default" | "compact";
  columns?: 1 | 2 | 3;
  className?: string;
}

export function TutorialGrid({
  tutorials,
  onSelect,
  selectedId,
  variant = "default",
  columns = 2,
  className,
}: TutorialGridProps) {
  const gridClass =
    columns === 1
      ? "grid-cols-1"
      : columns === 2
      ? "grid-cols-1 sm:grid-cols-2"
      : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3";

  return (
    <div className={cn("grid gap-3", gridClass, className)}>
      {tutorials.map((tutorial) => (
        <TutorialCard
          key={tutorial.id}
          tutorial={tutorial}
          variant={variant}
          isSelected={selectedId === tutorial.id}
          onClick={onSelect ? () => onSelect(tutorial) : undefined}
        />
      ))}
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

TutorialCard.displayName = "TutorialCard";
TutorialGrid.displayName = "TutorialGrid";

// ============================================================================
// Default Export
// ============================================================================

export default TutorialCard;
