"use client";

/**
 * ActivityFeed Component
 *
 * Displays recent system activity in a compact timeline format.
 * Shows ingestion, search, and system events.
 *
 * Features:
 * - Real-time updates via polling
 * - Grouped by time (just now, minutes ago, hours ago)
 * - Activity type icons
 * - Click to view details
 */

import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  Upload,
  Search,
  Database,
  Trash2,
  Settings,
  AlertCircle,
  Clock,
  RefreshCw,
} from "lucide-react";
import { useActivities } from "@/hooks/use-api";
import type { ActivityItem as ApiActivityItem } from "@/types";

// ============================================================================
// Types
// ============================================================================

interface DisplayActivityItem {
  id: string;
  type: "ingest" | "search" | "create" | "delete" | "config" | "error";
  title: string;
  description?: string;
  timestamp: Date;
  status?: "success" | "error" | "pending";
}

export interface ActivityFeedProps {
  limit?: number;
  onActivityClick?: (activity: DisplayActivityItem) => void;
  className?: string;
}

// ============================================================================
// Activity Type Configuration
// ============================================================================

const ACTIVITY_CONFIG: Record<
  DisplayActivityItem["type"],
  { icon: React.ReactNode; color: string }
> = {
  ingest: {
    icon: <Upload size={12} />,
    color: "text-blue-400",
  },
  search: {
    icon: <Search size={12} />,
    color: "text-violet-400",
  },
  create: {
    icon: <Database size={12} />,
    color: "text-emerald-400",
  },
  delete: {
    icon: <Trash2 size={12} />,
    color: "text-amber-400",
  },
  config: {
    icon: <Settings size={12} />,
    color: "text-zinc-400",
  },
  error: {
    icon: <AlertCircle size={12} />,
    color: "text-red-400",
  },
};

// ============================================================================
// Time Formatting
// ============================================================================

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHour < 24) return `${diffHour}h ago`;
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}

function RelativeTime({ date }: { date: Date }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return <>{formatRelativeTime(date)}</>;
}

// ============================================================================
// Transform API data to display format
// ============================================================================

function transformActivityItem(item: ApiActivityItem): DisplayActivityItem {
  const typeMap: Record<string, DisplayActivityItem["type"]> = {
    search: "search",
    ingest: "ingest",
    create: "create",
    delete: "delete",
    config: "config",
  };

  return {
    id: item.id,
    type: typeMap[item.type] || "config",
    title: item.title,
    description: item.description,
    timestamp: new Date(item.createdAt),
    status: item.status as DisplayActivityItem["status"],
  };
}

// ============================================================================
// Activity Item Component
// ============================================================================

interface ActivityItemRowProps {
  activity: DisplayActivityItem;
  onClick?: () => void;
}

function ActivityItemRow({ activity, onClick }: ActivityItemRowProps) {
  const config = ACTIVITY_CONFIG[activity.type];

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-start gap-3 p-2 -mx-2 rounded-md",
        "text-left transition-colors",
        "hover:bg-[var(--bg-hover)]",
        "focus:outline-none focus:ring-1 focus:ring-[var(--border-default)]"
      )}
    >
      {/* Icon */}
      <div
        className={cn(
          "flex items-center justify-center w-6 h-6 rounded-md mt-0.5",
          "bg-[var(--bg-elevated)]",
          config.color
        )}
      >
        {config.icon}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[13px] text-[var(--text-primary)] truncate">
            {activity.title}
          </span>
          {activity.status === "error" && (
            <AlertCircle size={12} className="text-red-400 shrink-0" />
          )}
          {activity.status === "pending" && (
            <Clock size={12} className="text-amber-400 shrink-0 animate-pulse" />
          )}
        </div>
        {activity.description && (
          <p className="text-[11px] text-[var(--text-muted)] truncate mt-0.5">
            {activity.description}
          </p>
        )}
      </div>

      {/* Timestamp - rendered only on client to avoid hydration mismatch */}
      <span className="text-[10px] text-[var(--text-muted)] shrink-0 mt-0.5">
        <RelativeTime date={activity.timestamp} />
      </span>
    </button>
  );
}

// ============================================================================
// Empty State
// ============================================================================

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <div className="w-10 h-10 rounded-full bg-[var(--bg-elevated)] flex items-center justify-center mb-3">
        <Clock size={18} className="text-[var(--text-muted)]" />
      </div>
      <p className="text-[13px] text-[var(--text-secondary)]">No recent activity</p>
      <p className="text-[11px] text-[var(--text-muted)] mt-1">
        Activity will appear here as you use the system
      </p>
    </div>
  );
}

// ============================================================================
// Loading Skeleton
// ============================================================================

function ActivitySkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="flex items-start gap-3 animate-pulse">
          <div className="w-6 h-6 rounded-md bg-[var(--bg-elevated)]" />
          <div className="flex-1 space-y-2">
            <div className="h-3 w-3/4 rounded bg-[var(--bg-elevated)]" />
            <div className="h-2 w-1/2 rounded bg-[var(--bg-elevated)]" />
          </div>
          <div className="h-2 w-10 rounded bg-[var(--bg-elevated)]" />
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Error State
// ============================================================================

function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center mb-3">
        <AlertCircle size={18} className="text-red-400" />
      </div>
      <p className="text-[13px] text-[var(--text-secondary)]">Failed to load activities</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 flex items-center gap-1 text-[11px] text-[var(--accent-primary)] hover:underline"
        >
          <RefreshCw size={10} />
          Retry
        </button>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ActivityFeed({
  limit = 10,
  onActivityClick,
  className,
}: ActivityFeedProps) {
  const { data: activities, isLoading, isError, refetch } = useActivities(limit);

  const displayActivities = (activities || []).map(transformActivityItem);

  return (
    <div className={cn("", className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[13px] font-medium text-[var(--text-primary)]">
          Recent Activity
        </h3>
      </div>

      {/* Content */}
      <div className="space-y-1">
        {isLoading ? (
          <ActivitySkeleton />
        ) : isError ? (
          <ErrorState onRetry={() => refetch()} />
        ) : displayActivities.length === 0 ? (
          <EmptyState />
        ) : (
          displayActivities.map((activity) => (
            <ActivityItemRow
              key={activity.id}
              activity={activity}
              onClick={() => onActivityClick?.(activity)}
            />
          ))
        )}
      </div>
    </div>
  );
}

ActivityFeed.displayName = "ActivityFeed";

export default ActivityFeed;
