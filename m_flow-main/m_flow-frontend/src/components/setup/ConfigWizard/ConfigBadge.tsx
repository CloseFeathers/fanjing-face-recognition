"use client";

/**
 * ConfigBadge Component
 *
 * Visual badge indicating whether a configuration setting is:
 * - CONFIGURABLE: Can be changed via the API/UI
 * - ENV ONLY: Requires editing .env file and restart
 * - SESSION: Stored in browser only (not persisted to server)
 *
 * @example
 * // API-configurable setting
 * <ConfigBadge type="configurable" />
 *
 * @example
 * // Environment variable only
 * <ConfigBadge type="env-only" />
 *
 * @example
 * // Session/local storage only
 * <ConfigBadge type="session" />
 */

import React from "react";
import { cn } from "@/lib/utils";

// ============================================================================
// Types
// ============================================================================

export type BadgeType = "configurable" | "env-only" | "session";

export interface ConfigBadgeProps {
  /** Badge type */
  type: BadgeType;
  /** Size variant */
  size?: "sm" | "md";
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Badge Configuration
// ============================================================================

interface BadgeConfig {
  label: string;
  description: string;
  bgClass: string;
  textClass: string;
  borderClass: string;
}

const BADGE_CONFIG: Record<BadgeType, BadgeConfig> = {
  configurable: {
    label: "Configurable",
    description: "Can be changed via the UI",
    bgClass: "bg-emerald-500/10",
    textClass: "text-emerald-400",
    borderClass: "border-emerald-500/30",
  },
  "env-only": {
    label: "Requires .env",
    description: "Edit .env file and restart to change",
    bgClass: "bg-amber-500/10",
    textClass: "text-amber-400",
    borderClass: "border-amber-500/30",
  },
  session: {
    label: "Session Only",
    description: "Stored in browser, not persisted to server",
    bgClass: "bg-zinc-500/10",
    textClass: "text-zinc-400",
    borderClass: "border-zinc-500/30",
  },
};

// ============================================================================
// Component
// ============================================================================

export function ConfigBadge({
  type,
  size = "sm",
  className,
}: ConfigBadgeProps) {
  const config = BADGE_CONFIG[type];

  return (
    <span
      className={cn(
        // Base styles
        "inline-flex items-center rounded-full border font-medium",
        // Size variants
        size === "sm" && "px-2 py-0.5 text-[10px]",
        size === "md" && "px-2.5 py-1 text-[11px]",
        // Color from config
        config.bgClass,
        config.textClass,
        config.borderClass,
        // Custom classes
        className
      )}
      title={config.description}
    >
      {config.label}
    </span>
  );
}

// ============================================================================
// Inline Badge (for use within text)
// ============================================================================

export function ConfigBadgeInline({ type }: { type: BadgeType }) {
  const config = BADGE_CONFIG[type];

  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium ml-2",
        config.bgClass,
        config.textClass
      )}
    >
      {type === "env-only" && (
        <svg
          className="w-3 h-3 mr-1"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
          />
        </svg>
      )}
      {config.label}
    </span>
  );
}

// ============================================================================
// Display Names
// ============================================================================

ConfigBadge.displayName = "ConfigBadge";
ConfigBadgeInline.displayName = "ConfigBadgeInline";

// ============================================================================
// Default Export
// ============================================================================

export default ConfigBadge;
