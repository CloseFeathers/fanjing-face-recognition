"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, AlertCircle, FileText, Hash, Users, Link2 } from "lucide-react";
import { useEntityNetwork } from "@/hooks/use-api";
import { EntityNetworkNode } from "@/types";

interface EntityNetworkProps {
  entityId: string;
  entityName?: string;
  datasetId?: string;
  onSelectEpisode: (episodeId: string, episodeName?: string) => void;
  onSelectFacet: (facetId: string, facetName?: string) => void;
}

/**
 * Layer 3: Entity Network
 * Shows all Episodes and Facets connected to a specific Entity.
 * Uses a card-based layout instead of force-directed graph.
 */
export function EntityNetwork({
  entityId,
  entityName,
  datasetId,
  onSelectEpisode,
  onSelectFacet,
}: EntityNetworkProps) {
  const { data, isLoading, error } = useEntityNetwork(entityId, datasetId);
  const [activeTab, setActiveTab] = useState<"episodes" | "facets" | "sameAs">("episodes");

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <Loader2
          size={20}
          className="text-[var(--text-muted)] animate-spin mb-3"
        />
        <p className="text-[13px] text-[var(--text-muted)]">
          Loading entity network...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <AlertCircle size={24} className="text-[var(--error)] mb-3" />
        <p className="text-[14px] text-[var(--text-primary)] mb-1">
          Failed to load entity network
        </p>
        <p className="text-[13px] text-[var(--text-muted)]">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
      </div>
    );
  }

  const episodes: EntityNetworkNode[] = data?.connectedEpisodes || [];
  const facets: EntityNetworkNode[] = data?.connectedFacets || [];
  const sameEntities: EntityNetworkNode[] = data?.sameEntities || [];
  const displayName = data?.entityName || entityName || entityId;
  const entityType = data?.entityType || "Entity";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
    >
      {/* Entity Header */}
      <div className="mb-4 p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-[#666666] flex items-center justify-center">
            <Users size={20} className="text-white" />
          </div>
          <div>
            <h2 className="text-[16px] font-medium text-[var(--text-primary)]">
              {displayName}
            </h2>
            <p className="text-[12px] text-[var(--text-muted)]">
              {entityType} · {episodes.length} episodes · {facets.length} facets
              {sameEntities.length > 0 && ` · ${sameEntities.length} aliases`}
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 p-1 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg w-fit">
        <button
          onClick={() => setActiveTab("episodes")}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] transition-colors
            ${
              activeTab === "episodes"
                ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
            }
          `}
        >
          <FileText size={14} />
          Episodes ({episodes.length})
        </button>
        <button
          onClick={() => setActiveTab("facets")}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] transition-colors
            ${
              activeTab === "facets"
                ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
            }
          `}
        >
          <Hash size={14} />
          Facets ({facets.length})
        </button>
        {sameEntities.length > 0 && (
          <button
            onClick={() => setActiveTab("sameAs")}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] transition-colors
              ${
                activeTab === "sameAs"
                  ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }
            `}
          >
            <Link2 size={14} />
            Same As ({sameEntities.length})
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        <AnimatePresence mode="wait">
          {activeTab === "episodes" && (
            <motion.div
              key="episodes"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.15 }}
            >
              {episodes.length === 0 ? (
                <div className="text-center py-8 text-[13px] text-[var(--text-muted)]">
                  No connected episodes
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {episodes.map((episode, index) => (
                    <ConnectionCard
                      key={episode.id}
                      node={episode}
                      index={index}
                      icon={<FileText size={14} />}
                      onClick={() => onSelectEpisode(episode.id, episode.name)}
                    />
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === "facets" && (
            <motion.div
              key="facets"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.15 }}
            >
              {facets.length === 0 ? (
                <div className="text-center py-8 text-[13px] text-[var(--text-muted)]">
                  No connected facets
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {facets.map((facet, index) => (
                    <ConnectionCard
                      key={facet.id}
                      node={facet}
                      index={index}
                      icon={<Hash size={14} />}
                      onClick={() => onSelectFacet(facet.id, facet.name)}
                    />
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === "sameAs" && (
            <motion.div
              key="sameAs"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.15 }}
            >
              {sameEntities.length === 0 ? (
                <div className="text-center py-8 text-[13px] text-[var(--text-muted)]">
                  No equivalent entities
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {sameEntities.map((entity, index) => (
                    <ConnectionCard
                      key={entity.id}
                      node={entity}
                      index={index}
                      icon={<Link2 size={14} />}
                      onClick={() => {}}
                    />
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

/**
 * Card component for displaying connected nodes
 */
function ConnectionCard({
  node,
  index,
  icon,
  onClick,
}: {
  node: EntityNetworkNode;
  index: number;
  icon: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <motion.button
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.03 }}
      onClick={onClick}
      className="
        group text-left p-3
        bg-[var(--bg-surface)] border border-[var(--border-subtle)]
        rounded-lg transition-all duration-200
        hover:border-[var(--border-default)] hover:bg-[var(--bg-hover)]
      "
    >
      <div className="flex items-start gap-2">
        <div className="mt-0.5 text-[var(--text-muted)] group-hover:text-[var(--text-secondary)] transition-colors">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[13px] font-medium text-[var(--text-primary)] group-hover:text-white truncate transition-colors">
            {node.name || `${node.type}_${node.id.slice(0, 8)}`}
          </p>
          <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
            via {node.relationship}
          </p>
        </div>
      </div>
    </motion.button>
  );
}
