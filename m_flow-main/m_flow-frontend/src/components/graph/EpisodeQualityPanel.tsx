"use client";

import React, { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  ChevronDown,
  ChevronUp,
  Play,
  AlertCircle,
  Circle,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { apiClient } from "@/lib/api/client";
import type { EpisodeQualityItem } from "@/types";

interface EpisodeQualityPanelProps {
  datasetId?: string;
}

export function EpisodeQualityPanel({ datasetId }: EpisodeQualityPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showAllEpisodes, setShowAllEpisodes] = useState(false);

  const queryClient = useQueryClient();

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["episodeQuality", datasetId],
    queryFn: () => apiClient.getEpisodeQuality(datasetId),
    refetchInterval: 30000,
  });

  const sizeCheckMutation = useMutation({
    mutationFn: (episodeIds: string[]) =>
      apiClient.runEpisodeSizeCheck(episodeIds),
    onSuccess: (result) => {
      toast.success(
        `Checked ${result.summary.checked} episodes: ${result.summary.split} split, ${result.summary.kept} kept`
      );
      queryClient.invalidateQueries({ queryKey: ["episodeQuality"] });
      queryClient.invalidateQueries({ queryKey: ["episodesOverview"] });
      queryClient.invalidateQueries({ queryKey: ["graph"] });
      setSelectedIds(new Set());
    },
    onError: (error: Error & { message?: string }) => {
      const message = error.message || "Unknown error";
      if (message.includes("pipeline") || message.includes("409")) {
        toast.error("Cannot run Size Check: ingestion pipeline is active");
      } else {
        toast.error(`Size Check failed: ${message}`);
      }
    },
  });

  const handleToggleSelect = (id: string, canSizeCheck: boolean) => {
    if (!canSizeCheck) return;

    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  const handleSelectAllProblematic = () => {
    if (!data?.problematic_episodes) return;
    const selectableIds = data.problematic_episodes
      .filter((e) => e.can_size_check)
      .map((e) => e.id);
    setSelectedIds(new Set(selectableIds));
  };

  const handleClearSelection = () => {
    setSelectedIds(new Set());
  };

  const handleRunSizeCheck = () => {
    if (selectedIds.size === 0) {
      toast.warning("Please select episodes to check");
      return;
    }
    sizeCheckMutation.mutate(Array.from(selectedIds));
  };

  const stats = data?.stats;
  const problematicEpisodes = data?.problematic_episodes || [];
  const allEpisodes = data?.all_episodes || [];

  const normalEpisodes = useMemo(() => {
    return allEpisodes.filter((e) => !e.issue_type && e.can_size_check);
  }, [allEpisodes]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-[var(--text-muted)] text-[12px] py-2">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span>Loading quality stats...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-[var(--error)] text-[12px] py-2">
        <AlertCircle className="w-3 h-3" />
        <span>Failed to load quality stats</span>
      </div>
    );
  }

  const hasProblems = problematicEpisodes.length > 0;

  return (
    <div className="mb-3">
      {/* Collapsed header bar */}
      <div
        className={cn(
          "flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-colors",
          "bg-[var(--bg-surface)] border border-[var(--border-subtle)]",
          "hover:border-[var(--border-default)]"
        )}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          {hasProblems ? (
            <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
          )}
          <span className="text-[12px] font-medium text-[var(--text-primary)]">
            Quality
          </span>
          {hasProblems && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-500">
              {problematicEpisodes.length}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Mini stats */}
          <div className="flex items-center gap-3 text-[11px] text-[var(--text-muted)]">
            <span>{stats?.total_episodes || 0} total</span>
            {(stats?.empty_count || 0) > 0 && (
              <span className="text-red-400">{stats?.empty_count} empty</span>
            )}
            {(stats?.oversized_count || 0) > 0 && (
              <span className="text-amber-400">{stats?.oversized_count} oversized</span>
            )}
          </div>

          <button
            className="p-1 rounded hover:bg-[var(--bg-hover)] transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              refetch();
            }}
          >
            <RefreshCw className="w-3 h-3 text-[var(--text-muted)]" />
          </button>

          {isExpanded ? (
            <ChevronUp className="w-3.5 h-3.5 text-[var(--text-muted)]" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-[var(--text-muted)]" />
          )}
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="mt-2 p-3 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
          {/* Problematic Episodes */}
          {problematicEpisodes.length > 0 && (
            <div className="mb-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-medium text-[var(--text-muted)]">
                  Issues ({problematicEpisodes.length})
                </span>
                <div className="flex gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-[10px]"
                    onClick={handleSelectAllProblematic}
                    disabled={sizeCheckMutation.isPending}
                  >
                    Select All
                  </Button>
                  {selectedIds.size > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-[10px]"
                      onClick={handleClearSelection}
                    >
                      Clear
                    </Button>
                  )}
                  <Button
                    size="sm"
                    className="h-6 px-2 text-[10px]"
                    onClick={handleRunSizeCheck}
                    disabled={selectedIds.size === 0 || sizeCheckMutation.isPending}
                  >
                    {sizeCheckMutation.isPending ? (
                      <Loader2 className="w-3 h-3 animate-spin mr-1" />
                    ) : (
                      <Play className="w-3 h-3 mr-1" />
                    )}
                    Check ({selectedIds.size})
                  </Button>
                </div>
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {problematicEpisodes.map((ep) => (
                  <EpisodeRow
                    key={ep.id}
                    episode={ep}
                    isSelected={selectedIds.has(ep.id)}
                    onToggle={() => handleToggleSelect(ep.id, ep.can_size_check)}
                    disabled={sizeCheckMutation.isPending}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Normal Episodes */}
          {normalEpisodes.length > 0 && (
            <div>
              <div
                className="flex items-center justify-between cursor-pointer py-1"
                onClick={() => setShowAllEpisodes(!showAllEpisodes)}
              >
                <span className="text-[11px] text-[var(--text-muted)]">
                  Normal ({normalEpisodes.length})
                </span>
                {showAllEpisodes ? (
                  <ChevronUp className="w-3 h-3 text-[var(--text-muted)]" />
                ) : (
                  <ChevronDown className="w-3 h-3 text-[var(--text-muted)]" />
                )}
              </div>
              {showAllEpisodes && (
                <div className="mt-1 space-y-1 max-h-32 overflow-y-auto">
                  {normalEpisodes.slice(0, 50).map((ep) => (
                    <EpisodeRow
                      key={ep.id}
                      episode={ep}
                      isSelected={selectedIds.has(ep.id)}
                      onToggle={() => handleToggleSelect(ep.id, ep.can_size_check)}
                      disabled={sizeCheckMutation.isPending}
                    />
                  ))}
                  {normalEpisodes.length > 50 && (
                    <div className="text-[10px] text-[var(--text-muted)] text-center py-1">
                      +{normalEpisodes.length - 50} more
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Empty state */}
          {!hasProblems && stats?.total_episodes === 0 && (
            <div className="text-center text-[11px] text-[var(--text-muted)] py-2">
              No episodes
            </div>
          )}
          {!hasProblems && (stats?.total_episodes || 0) > 0 && normalEpisodes.length === 0 && (
            <div className="flex items-center justify-center gap-1 text-[11px] text-emerald-500 py-2">
              <CheckCircle2 className="w-3 h-3" />
              All healthy
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EpisodeRow({
  episode,
  isSelected,
  onToggle,
  disabled,
}: {
  episode: EpisodeQualityItem;
  isSelected: boolean;
  onToggle: () => void;
  disabled: boolean;
}) {
  const isDisabled = !episode.can_size_check || disabled;

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-2 py-1.5 rounded text-[11px] transition-colors",
        isDisabled
          ? "opacity-50 cursor-not-allowed"
          : "cursor-pointer hover:bg-[var(--bg-hover)]",
        isSelected && "bg-[var(--bg-elevated)]"
      )}
      onClick={() => !isDisabled && onToggle()}
    >
      {/* Checkbox */}
      <div
        className={cn(
          "w-3 h-3 rounded border flex items-center justify-center flex-shrink-0",
          isSelected
            ? "bg-[var(--text-primary)] border-[var(--text-primary)]"
            : "border-[var(--border-default)]",
          isDisabled && "opacity-50"
        )}
      >
        {isSelected && <CheckCircle2 className="w-2 h-2 text-[var(--bg-base)]" />}
      </div>

      {/* Issue icon */}
      {episode.issue_type === "empty" && (
        <Circle className="w-2.5 h-2.5 text-red-500 fill-red-500 flex-shrink-0" />
      )}
      {episode.issue_type === "oversized" && (
        <AlertTriangle
          className={cn(
            "w-2.5 h-2.5 flex-shrink-0",
            episode.severity === "high"
              ? "text-red-500"
              : episode.severity === "medium"
                ? "text-amber-500"
                : "text-yellow-500"
          )}
        />
      )}
      {!episode.issue_type && (
        <CheckCircle2 className="w-2.5 h-2.5 text-emerald-500 flex-shrink-0" />
      )}

      {/* Name */}
      <span className="flex-1 truncate text-[var(--text-secondary)]">
        {episode.name}
      </span>

      {/* Facet count */}
      <span className="text-[var(--text-muted)] flex-shrink-0">
        {episode.facet_count}
      </span>

      {/* Issue badge */}
      {episode.issue_type && (
        <span
          className={cn(
            "text-[9px] px-1 py-0.5 rounded flex-shrink-0",
            episode.issue_type === "empty"
              ? "bg-red-500/20 text-red-400"
              : "bg-amber-500/20 text-amber-400"
          )}
        >
          {episode.issue_type}
        </span>
      )}
    </div>
  );
}
