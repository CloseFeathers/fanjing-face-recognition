"use client";

import React, { useState, useCallback, useMemo, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Loader2,
  AlertCircle,
  FileText,
  Users,
  Calendar,
  ChevronDown,
  Search,
  SortAsc,
  SortDesc,
} from "lucide-react";
import { useEpisodesOverview } from "@/hooks/use-api";
import { EpisodeOverview as EpisodeOverviewType } from "@/types";
import { Button } from "@/components/ui/button";

// Extract preview text from summary (first section or truncated text)
function getSummaryPreview(summary: string | undefined, maxLength: number = 100): string {
  if (!summary) return "";
  
  // Try to extract first section content
  const sectionMatch = summary.match(/【[^】]+】([^【]*)/);
  if (sectionMatch && sectionMatch[1]) {
    const content = sectionMatch[1].trim();
    return content.length > maxLength ? content.slice(0, maxLength) + "..." : content;
  }
  
  // Fallback: just truncate
  return summary.length > maxLength ? summary.slice(0, maxLength) + "..." : summary;
}

interface EpisodeOverviewProps {
  datasetId?: string;
  onSelectEpisode: (episode: EpisodeOverviewType) => void;
}

// Animation variants for staggered card entrance
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.04,
      delayChildren: 0.1,
    },
  },
};

const cardVariants = {
  hidden: { 
    opacity: 0, 
    y: 20,
    scale: 0.95,
  },
  visible: { 
    opacity: 1, 
    y: 0,
    scale: 1,
    transition: {
      type: "spring" as const,
      stiffness: 300,
      damping: 24,
    },
  },
  hover: {
    scale: 1.02,
    y: -4,
    transition: {
      type: "spring" as const,
      stiffness: 400,
      damping: 20,
    },
  },
  tap: {
    scale: 0.98,
  },
};

// Skeleton loader for cards
function EpisodeCardSkeleton() {
  return (
    <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg animate-pulse">
      <div className="h-4 bg-[var(--bg-hover)] rounded w-3/4 mb-3" />
      <div className="h-3 bg-[var(--bg-hover)] rounded w-full mb-2" />
      <div className="h-3 bg-[var(--bg-hover)] rounded w-2/3 mb-4" />
      <div className="flex gap-4">
        <div className="h-3 bg-[var(--bg-hover)] rounded w-16" />
        <div className="h-3 bg-[var(--bg-hover)] rounded w-16" />
      </div>
    </div>
  );
}

// Individual episode card with optimized rendering
const EpisodeCard = React.memo(function EpisodeCard({
  episode,
  onSelect,
}: {
  episode: EpisodeOverviewType;
  onSelect: (episode: EpisodeOverviewType) => void;
}) {
  return (
    <motion.button
      variants={cardVariants}
      whileHover="hover"
      whileTap="tap"
      onClick={() => onSelect(episode)}
      className="
        group text-left p-4 w-full
        bg-[var(--bg-surface)] border border-[var(--border-subtle)] 
        rounded-lg transition-colors duration-200
        hover:border-[var(--border-default)] hover:bg-[var(--bg-hover)]
        focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-offset-2 focus:ring-offset-[var(--bg-base)]
      "
    >
      {/* Episode Name */}
      <h3 className="text-[14px] font-medium text-[var(--text-primary)] mb-2 line-clamp-2 group-hover:text-white transition-colors">
        {episode.name || `Episode ${episode.id.slice(0, 8)}`}
      </h3>

      {/* Summary preview */}
      {episode.summary && (
        <div className="mb-3">
          {/* Preview text */}
          <p className="text-[12px] text-[var(--text-muted)] line-clamp-2 relative">
            {getSummaryPreview(episode.summary, 120)}
            <span className="absolute bottom-0 right-0 w-12 h-full bg-gradient-to-l from-[var(--bg-surface)] group-hover:from-[var(--bg-hover)] to-transparent pointer-events-none" />
          </p>
        </div>
      )}

      {/* Stats with icons */}
      <div className="flex items-center gap-4 text-[11px] text-[var(--text-muted)]">
        <motion.div 
          className="flex items-center gap-1"
          whileHover={{ scale: 1.05 }}
        >
          <FileText size={12} className="opacity-60" />
          <span>{episode.facetCount} facets</span>
        </motion.div>
        <motion.div 
          className="flex items-center gap-1"
          whileHover={{ scale: 1.05 }}
        >
          <Users size={12} className="opacity-60" />
          <span>{episode.entityCount} entities</span>
        </motion.div>
      </div>

      {/* Timestamp */}
      {episode.createdAt && (
        <div className="flex items-center gap-1 mt-2 text-[10px] text-[var(--text-muted)] opacity-60">
          <Calendar size={10} />
          <span>{new Date(parseInt(episode.createdAt)).toLocaleDateString()}</span>
        </div>
      )}

      {/* Hover indicator */}
      <motion.div
        className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
        initial={{ scale: 0.8 }}
        whileHover={{ scale: 1 }}
      >
        <div className="text-[10px] text-[var(--text-muted)] flex items-center gap-1">
          <span>Explore</span>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
            <path d="M4.5 2L8.5 6L4.5 10" stroke="currentColor" strokeWidth="1.5" fill="none" />
          </svg>
        </div>
      </motion.div>
    </motion.button>
  );
});

/**
 * Layer 0: Episodes Overview
 * Displays all episodes as cards with aggregated information.
 * Optimized with virtual scrolling hints and pagination.
 */
export function EpisodeOverview({
  datasetId,
  onSelectEpisode,
}: EpisodeOverviewProps) {
  const [displayLimit, setDisplayLimit] = useState(20);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Request all episodes without limit
  const { data, isLoading, error } = useEpisodesOverview({ datasetId, limit: 100000 });

  // Filter and sort episodes
  const filteredEpisodes = useMemo(() => {
    if (!data?.episodes) return [];
    
    let episodes = [...data.episodes];
    
    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      episodes = episodes.filter(
        (ep) =>
          ep.name?.toLowerCase().includes(query) ||
          ep.summary?.toLowerCase().includes(query)
      );
    }
    
    // Sort by createdAt or name
    episodes.sort((a, b) => {
      const aVal = a.createdAt || a.name || "";
      const bVal = b.createdAt || b.name || "";
      return sortOrder === "desc" 
        ? bVal.localeCompare(aVal)
        : aVal.localeCompare(bVal);
    });
    
    return episodes;
  }, [data?.episodes, searchQuery, sortOrder]);

  // Paginated episodes for display
  const displayedEpisodes = useMemo(
    () => filteredEpisodes.slice(0, displayLimit),
    [filteredEpisodes, displayLimit]
  );

  const hasMore = displayLimit < filteredEpisodes.length;

  // Intersection Observer for infinite scroll
  useEffect(() => {
    if (!loadMoreRef.current || !hasMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setDisplayLimit((prev) => prev + 20);
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(loadMoreRef.current);
    return () => observer.disconnect();
  }, [hasMore]);

  // Reset display limit when search changes
  useEffect(() => {
    setDisplayLimit(20);
  }, [searchQuery]);

  const handleLoadMore = useCallback(() => {
    setDisplayLimit((prev) => prev + 20);
  }, []);

  // Loading state with skeleton
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="h-10 bg-[var(--bg-surface)] rounded-lg animate-pulse" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <EpisodeCardSkeleton key={i} />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <motion.div 
        className="flex flex-col items-center justify-center py-20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <AlertCircle size={24} className="text-[var(--error)] mb-3" />
        <p className="text-[14px] text-[var(--text-primary)] mb-1">
          Failed to load episodes
        </p>
        <p className="text-[13px] text-[var(--text-muted)]">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
      </motion.div>
    );
  }

  const episodes = data?.episodes || [];

  if (episodes.length === 0) {
    return (
      <motion.div 
        className="flex flex-col items-center justify-center py-20"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
      >
        <motion.div
          animate={{ 
            y: [0, -8, 0],
          }}
          transition={{ 
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <FileText size={32} className="text-[var(--text-muted)] mb-3" />
        </motion.div>
        <p className="text-[14px] text-[var(--text-primary)] mb-1">
          No episodes found
        </p>
        <p className="text-[13px] text-[var(--text-muted)]">
          Episodes will appear here after data ingestion
        </p>
      </motion.div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with search and sort */}
      <motion.div 
        className="flex items-center justify-between gap-4 flex-wrap"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        {/* Stats */}
        <div className="text-[13px] text-[var(--text-muted)]">
          {filteredEpisodes.length === data?.total
            ? `${data?.total || episodes.length} episodes`
            : `${filteredEpisodes.length} of ${data?.total || episodes.length} episodes`}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          {/* Search input */}
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
            <input
              type="text"
              placeholder="Search episodes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="
                pl-8 pr-3 py-1.5 text-[12px]
                bg-[var(--bg-surface)] border border-[var(--border-subtle)]
                rounded-md text-[var(--text-primary)]
                placeholder:text-[var(--text-muted)]
                focus:outline-none focus:border-[var(--border-default)]
                transition-colors w-40
              "
            />
          </div>

          {/* Sort toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
            className="h-8 px-2"
          >
            {sortOrder === "desc" ? (
              <SortDesc size={14} className="text-[var(--text-muted)]" />
            ) : (
              <SortAsc size={14} className="text-[var(--text-muted)]" />
            )}
          </Button>
        </div>
      </motion.div>

      {/* Episode Grid - key forces re-render on search */}
      <div
        key={searchQuery}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
      >
        {displayedEpisodes.map((episode, index) => (
          <EpisodeCard
            key={`${episode.id}-${episode.datasetId || index}`}
            episode={episode}
            onSelect={onSelectEpisode}
          />
        ))}
      </div>

      {/* Load more trigger / button */}
      {hasMore && (
        <motion.div
          ref={loadMoreRef}
          className="flex justify-center pt-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={handleLoadMore}
            className="gap-2"
          >
            <ChevronDown size={14} />
            Load more ({filteredEpisodes.length - displayLimit} remaining)
          </Button>
        </motion.div>
      )}

      {/* Loading indicator for infinite scroll */}
      {hasMore && displayLimit > 20 && (
        <div className="flex justify-center py-4">
          <Loader2 size={16} className="animate-spin text-[var(--text-muted)]" />
        </div>
      )}
    </div>
  );
}
