"use client";

/**
 * QuickActions Component
 *
 * Displays quick action cards for common tasks and navigation.
 * Used in the Getting Started section for easy access to features.
 *
 * Features:
 * - Action cards with icons
 * - Internal and external link support
 * - Hover animations
 * - Responsive grid layout
 *
 * @example
 * <QuickActions actions={quickActions} />
 */

import React from "react";
import { cn } from "@/lib/utils";
import {
  Upload,
  Search,
  GitBranch,
  Book,
  Github,
  MessageCircle,
  ExternalLink,
  ChevronRight,
  LucideIcon,
} from "lucide-react";
import { useUIStore } from "@/lib/store";
import type { View } from "@/lib/store/ui";
import type { QuickAction } from "@/types/setup";

// ============================================================================
// Types
// ============================================================================

export interface QuickActionsProps {
  /** Quick actions to display */
  actions: QuickAction[];
  /** Layout variant */
  variant?: "grid" | "list";
  /** Additional CSS classes */
  className?: string;
}

export interface ActionCardProps {
  /** Action data */
  action: QuickAction;
  /** Variant */
  variant?: "grid" | "list";
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Icon Mapping
// ============================================================================

const ICON_MAP: Record<string, LucideIcon> = {
  Upload,
  Search,
  GitBranch,
  Book,
  Github,
  MessageCircle,
};

function getIcon(iconName: string): LucideIcon {
  return ICON_MAP[iconName] || Book;
}

// ============================================================================
// Action Card Component
// ============================================================================

const VIEW_MAP: Record<string, string> = {
  "/memorize": "memorize-ingest",
  "/retrieve": "retrieve-episodic",
  "/memories": "memories",
};

export function ActionCard({ action, variant = "grid", className }: ActionCardProps) {
  const Icon = getIcon(action.icon);
  const isExternal = action.external;
  const { setCurrentView } = useUIStore();

  const CardContent = () => (
    <>
      {/* Icon */}
      <div
        className={cn(
          "rounded-lg flex items-center justify-center shrink-0",
          "bg-zinc-800 group-hover:bg-zinc-700 transition-colors",
          variant === "grid" ? "w-10 h-10 mb-3" : "w-9 h-9"
        )}
      >
        <Icon
          size={variant === "grid" ? 20 : 18}
          className="text-zinc-400 group-hover:text-zinc-200"
        />
      </div>

      {/* Content */}
      <div className={cn(variant === "list" && "flex-1 min-w-0")}>
        <h4
          className={cn(
            "font-medium text-zinc-200 group-hover:text-white",
            variant === "grid" ? "text-[14px] mb-1" : "text-[13px]"
          )}
        >
          {action.title}
          {isExternal && (
            <ExternalLink
              size={10}
              className="inline ml-1.5 text-zinc-500"
            />
          )}
        </h4>
        <p
          className={cn(
            "text-zinc-500",
            variant === "grid" ? "text-[12px]" : "text-[11px] truncate"
          )}
        >
          {action.description}
        </p>
      </div>

      {/* Arrow (list variant) */}
      {variant === "list" && (
        <ChevronRight
          size={14}
          className="text-zinc-600 group-hover:text-zinc-400 shrink-0"
        />
      )}
    </>
  );

  const cardClasses = cn(
    "group relative rounded-xl border transition-all duration-200",
    "bg-zinc-900/30 hover:bg-zinc-900/50",
    "border-zinc-800 hover:border-zinc-700",
    variant === "grid" ? "p-4" : "p-3 flex items-center gap-3",
    className
  );

  if (isExternal) {
    return (
      <a
        href={action.href}
        target="_blank"
        rel="noopener noreferrer"
        className={cardClasses}
      >
        <CardContent />
      </a>
    );
  }

  const viewName = (VIEW_MAP[action.href] || action.href.replace("/", "")) as View;

  return (
    <button
      type="button"
      onClick={() => setCurrentView(viewName)}
      className={cn(cardClasses, "text-left w-full")}
    >
      <CardContent />
    </button>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function QuickActions({
  actions,
  variant = "grid",
  className,
}: QuickActionsProps) {
  const gridClass =
    variant === "grid"
      ? "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3"
      : "space-y-2";

  return (
    <div className={cn(gridClass, className)}>
      {actions.map((action) => (
        <ActionCard key={action.id} action={action} variant={variant} />
      ))}
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

ActionCard.displayName = "ActionCard";
QuickActions.displayName = "QuickActions";

// ============================================================================
// Default Export
// ============================================================================

export default QuickActions;
