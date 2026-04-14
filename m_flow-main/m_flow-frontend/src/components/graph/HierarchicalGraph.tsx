"use client";

import React, { useState, useCallback, useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence, LayoutGroup } from "framer-motion";
import { ArrowLeft, Network, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { BreadcrumbNav } from "./BreadcrumbNav";
import { EpisodeOverview } from "./EpisodeOverview";
import { EpisodeQualityPanel } from "./EpisodeQualityPanel";
import { EpisodeSubgraph } from "./EpisodeSubgraph";
import { FacetSubgraph } from "./FacetSubgraph";
import { EntityNetwork } from "./EntityNetwork";
import { ProcedureSubgraph } from "./ProcedureSubgraph";
import { AlertTriangle } from "lucide-react";
import {
  NavigationState,
  NavigationLayer,
  EpisodeOverview as EpisodeOverviewType,
  GraphNode,
} from "@/types";

function RemoveProceduralButton({ datasetId }: { datasetId?: string }) {
  const [confirming, setConfirming] = React.useState(false);
  const [removing, setRemoving] = React.useState(false);
  const [result, setResult] = React.useState<string | null>(null);
  const queryClient = useQueryClient();

  const handleRemove = async () => {
    setRemoving(true);
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = typeof window !== "undefined" ? localStorage.getItem("mflow_token") || "" : "";
      const resp = await fetch(`${baseUrl}/api/v1/prune/procedural`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify(datasetId ? { dataset_id: datasetId } : {}),
      });
      const data = await resp.json();
      setResult(data.message || "Done");
      queryClient.invalidateQueries({ queryKey: ["procedures-overview"] });
    } catch {
      setResult("Failed");
    } finally {
      setRemoving(false);
      setConfirming(false);
    }
  };

  return (
    <div className="mt-3 pt-3 border-t border-[var(--border-subtle)]">
      {result && <p className="text-[10px] text-green-400 mb-2">{result}</p>}
      {!confirming ? (
        <button
          onClick={() => setConfirming(true)}
          className="text-[10px] text-red-400/60 hover:text-red-400 transition-colors"
        >
          Remove All Procedural Data...
        </button>
      ) : (
        <div className="flex items-center gap-2">
          <button
            onClick={handleRemove}
            disabled={removing}
            className="px-2 py-1 text-[10px] bg-red-500/15 text-red-400 rounded hover:bg-red-500/25 transition-colors disabled:opacity-50"
          >
            {removing ? "Removing..." : "Confirm Remove"}
          </button>
          <button
            onClick={() => setConfirming(false)}
            className="px-2 py-1 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

interface HierarchicalGraphProps {
  datasetId?: string;
}

function ProcedureOverviewList({ datasetId, onSelect }: { datasetId?: string; onSelect: (id: string, name?: string) => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ["procedures-overview", datasetId],
    queryFn: async () => {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const params = datasetId ? `?dataset_id=${datasetId}` : "";
      const token = typeof window !== "undefined" ? localStorage.getItem("mflow_token") || "" : "";
      const resp = await fetch(`${baseUrl}/api/v1/graph/procedures${params}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!resp.ok) return { procedures: [], total: 0 };
      return resp.json();
    },
  });

  if (isLoading) {
    return <div className="text-xs text-[var(--text-muted)] py-4 text-center">Loading procedures...</div>;
  }

  const procedures = data?.procedures || [];
  if (procedures.length === 0) {
    return <div className="text-xs text-[var(--text-muted)] py-4 text-center">No procedures found. Use Extract Procedures to generate them.</div>;
  }

  return (
    <div className="space-y-1.5">
      {procedures.map((p: any) => (
        <button
          key={p.id}
          onClick={() => onSelect(p.id, p.name)}
          className="w-full text-left px-3 py-2.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] hover:border-[var(--border-default)] transition-colors group"
        >
          <div className="flex items-center justify-between">
            <span className="text-xs text-[var(--text-primary)] group-hover:text-[var(--accent)] transition-colors truncate">
              {p.name}
            </span>
            <div className="flex items-center gap-2 flex-shrink-0">
              {p.confidence === "low" && (
                <span className="text-[9px] text-amber-400">low</span>
              )}
              <span className="text-[9px] text-[var(--text-muted)]">
                v{p.version} · {p.point_count} points
              </span>
            </div>
          </div>
          {p.search_text && (
            <p className="text-[10px] text-[var(--text-muted)] mt-0.5 truncate">{p.search_text}</p>
          )}
        </button>
      ))}
    </div>
  );
}

// Enhanced transition variants for layer switching
const layerVariants = {
  // Enter from right (drilling down)
  enterFromRight: {
    initial: { opacity: 0, x: 60, scale: 0.96 },
    animate: { 
      opacity: 1, 
      x: 0, 
      scale: 1,
      transition: {
        type: "spring" as const,
        stiffness: 300,
        damping: 30,
        mass: 0.8,
      },
    },
    exit: { 
      opacity: 0, 
      x: -40, 
      scale: 0.98,
      transition: {
        duration: 0.2,
        ease: "easeIn" as const,
      },
    },
  },
  // Enter from left (going back)
  enterFromLeft: {
    initial: { opacity: 0, x: -60, scale: 0.96 },
    animate: { 
      opacity: 1, 
      x: 0, 
      scale: 1,
      transition: {
        type: "spring" as const,
        stiffness: 300,
        damping: 30,
        mass: 0.8,
      },
    },
    exit: { 
      opacity: 0, 
      x: 40, 
      scale: 0.98,
      transition: {
        duration: 0.2,
        ease: "easeIn" as const,
      },
    },
  },
  // Fade for overview
  fade: {
    initial: { opacity: 0, scale: 0.98 },
    animate: { 
      opacity: 1, 
      scale: 1,
      transition: {
        type: "spring" as const,
        stiffness: 400,
        damping: 30,
      },
    },
    exit: { 
      opacity: 0, 
      scale: 1.02,
      transition: {
        duration: 0.15,
      },
    },
  },
};

// Back button animation
const backButtonVariants = {
  hidden: { opacity: 0, x: -20, scale: 0.8 },
  visible: { 
    opacity: 1, 
    x: 0, 
    scale: 1,
    transition: {
      type: "spring" as const,
      stiffness: 500,
      damping: 25,
    },
  },
  exit: { 
    opacity: 0, 
    x: -20, 
    scale: 0.8,
    transition: { duration: 0.15 },
  },
  hover: {
    scale: 1.1,
    x: -2,
  },
  tap: {
    scale: 0.95,
  },
};

// Keyboard hint animation
const hintVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { delay: 0.5 },
  },
  exit: { 
    opacity: 0, 
    y: 10,
    transition: { duration: 0.1 },
  },
};

/**
 * Main component for hierarchical graph navigation.
 * Implements progressive disclosure pattern with polished animations.
 */
export function HierarchicalGraph({ datasetId }: HierarchicalGraphProps) {
  // Navigation state
  const [navigation, setNavigation] = useState<NavigationState>({
    layer: "overview",
    datasetId,
  });

  // Navigation history for back button
  const [history, setHistory] = useState<NavigationState[]>([]);
  
  // Track navigation direction for animations
  const [navigationDirection, setNavigationDirection] = useState<"forward" | "back">("forward");
  
  // Procedural memory toggle (experimental)
  const [showProcedural, setShowProcedural] = useState(false);

  // Reset navigation when datasetId changes (dataset selector in header)
  React.useEffect(() => {
    if (datasetId !== navigation.datasetId) {
      setHistory([]);
      setNavigation({
        layer: "overview",
        datasetId,
      });
    }
  }, [datasetId, navigation.datasetId]);

  // Determine layer depth for animation direction
  const getLayerDepth = useCallback((layer: NavigationLayer): number => {
    const depths: Record<NavigationLayer, number> = {
      overview: 0,
      episode: 1,
      procedure: 1,
      facet: 2,
      entity: 3,
    };
    return depths[layer];
  }, []);

  // Navigate to a specific layer
  const navigateTo = useCallback(
    (
      layer: NavigationLayer,
      options?: {
        episodeId?: string;
        episodeName?: string;
        facetId?: string;
        facetName?: string;
        entityId?: string;
        entityName?: string;
        procedureId?: string;
        procedureName?: string;
      }
    ) => {
      const currentDepth = getLayerDepth(navigation.layer);
      const newDepth = getLayerDepth(layer);
      setNavigationDirection(newDepth > currentDepth ? "forward" : "back");

      // Save current state to history
      setHistory((prev) => [...prev, navigation]);

      // Build new navigation state
      const newState: NavigationState = {
        layer,
        datasetId: navigation.datasetId,
      };

      // Preserve parent context based on target layer
      if (layer === "episode" || layer === "facet" || layer === "entity") {
        newState.episodeId = options?.episodeId || navigation.episodeId;
        newState.episodeName = options?.episodeName || navigation.episodeName;
      }

      if (layer === "facet" || layer === "entity") {
        newState.facetId = options?.facetId || navigation.facetId;
        newState.facetName = options?.facetName || navigation.facetName;
      }

      if (layer === "entity") {
        newState.entityId = options?.entityId;
        newState.entityName = options?.entityName;
      }

      if (layer === "procedure") {
        newState.procedureId = options?.procedureId;
        newState.procedureName = options?.procedureName;
      }

      setNavigation(newState);
    },
    [navigation, getLayerDepth]
  );

  // Handle breadcrumb navigation
  const handleBreadcrumbNavigate = useCallback(
    (layer: NavigationLayer) => {
      if (layer === navigation.layer) return;

      const currentDepth = getLayerDepth(navigation.layer);
      const newDepth = getLayerDepth(layer);
      setNavigationDirection(newDepth > currentDepth ? "forward" : "back");

      const newState: NavigationState = {
        layer,
        datasetId: navigation.datasetId,
      };

      if (layer === "overview") {
        setHistory([]);
        setNavigation(newState);
        return;
      }

      if (layer === "episode") {
        newState.episodeId = navigation.episodeId;
        newState.episodeName = navigation.episodeName;
      }

      if (layer === "facet") {
        newState.episodeId = navigation.episodeId;
        newState.episodeName = navigation.episodeName;
        newState.facetId = navigation.facetId;
        newState.facetName = navigation.facetName;
      }

      setHistory((prev) => [...prev, navigation]);
      setNavigation(newState);
    },
    [navigation, getLayerDepth]
  );

  // Go back
  const goBack = useCallback(() => {
    setNavigationDirection("back");
    
    if (history.length === 0) {
      setNavigation({ layer: "overview", datasetId: navigation.datasetId });
      return;
    }

    const previousState = history[history.length - 1];
    setHistory((prev) => prev.slice(0, -1));
    setNavigation(previousState);
  }, [history, navigation.datasetId]);

  // Handle episode selection from overview
  const handleSelectEpisode = useCallback(
    (episode: EpisodeOverviewType) => {
      navigateTo("episode", {
        episodeId: episode.id,
        episodeName: episode.name,
      });
    },
    [navigateTo]
  );

  // Handle facet selection
  const handleSelectFacet = useCallback(
    (facet: GraphNode) => {
      navigateTo("facet", {
        facetId: facet.id,
        facetName: facet.name,
      });
    },
    [navigateTo]
  );

  // Handle entity selection
  const handleSelectEntity = useCallback(
    (entity: GraphNode) => {
      navigateTo("entity", {
        entityId: entity.id,
        entityName: entity.name,
      });
    },
    [navigateTo]
  );

  // Handle navigation from EntityNetwork to Episode
  const handleEntityToEpisode = useCallback(
    (episodeId: string, episodeName?: string) => {
      navigateTo("episode", {
        episodeId,
        episodeName,
      });
    },
    [navigateTo]
  );

  // Handle navigation from EntityNetwork to Facet
  const handleEntityToFacet = useCallback(
    (facetId: string, facetName?: string) => {
      navigateTo("facet", {
        facetId,
        facetName,
      });
    },
    [navigateTo]
  );

  // Handle procedure selection from overview
  const handleSelectProcedure = useCallback(
    (procedureId: string, procedureName?: string) => {
      navigateTo("procedure", { procedureId, procedureName });
    },
    [navigateTo]
  );

  // Get animation variant based on navigation direction
  const getAnimationVariant = useMemo(() => {
    if (navigation.layer === "overview") {
      return layerVariants.fade;
    }
    return navigationDirection === "forward"
      ? layerVariants.enterFromRight
      : layerVariants.enterFromLeft;
  }, [navigation.layer, navigationDirection]);

  // Keyboard navigation
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && navigation.layer !== "overview") {
        goBack();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigation.layer, goBack]);

  // Layer depth indicator
  const layerDepth = getLayerDepth(navigation.layer);

  return (
    <LayoutGroup>
      <div className="h-[calc(100vh-8rem)] flex flex-col">
        {/* Header */}
        <motion.div 
          className="flex items-center justify-between mb-4"
          layout
        >
          <div className="flex items-center gap-3">
            {/* Back button with animation */}
            <AnimatePresence mode="wait">
              {navigation.layer !== "overview" && (
                <motion.div
                  variants={backButtonVariants}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  whileHover="hover"
                  whileTap="tap"
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={goBack}
                    className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                  >
                    <ArrowLeft size={16} />
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Title */}
            <div>
              <motion.h1 
                className="text-lg font-medium text-[var(--text-primary)]"
                layout
              >
                Knowledge Graph
              </motion.h1>
              <BreadcrumbNav
                navigation={navigation}
                onNavigate={handleBreadcrumbNavigate}
              />
            </div>
          </div>

          {/* Layer depth indicator */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              {[0, 1, 2, 3].map((depth) => (
                <motion.div
                  key={depth}
                  className={`w-1.5 h-1.5 rounded-full transition-colors ${
                    depth <= layerDepth
                      ? "bg-[var(--text-primary)]"
                      : "bg-[var(--border-subtle)]"
                  }`}
                  initial={{ scale: 0.8 }}
                  animate={{ 
                    scale: depth === layerDepth ? 1.2 : 1,
                    opacity: depth <= layerDepth ? 1 : 0.3,
                  }}
                  transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
              ))}
            </div>
            <Layers size={16} className="text-[var(--text-muted)] ml-1" />
          </div>
        </motion.div>

        {/* Content Area with enhanced transitions */}
        <div className="flex-1 overflow-hidden relative">
          <AnimatePresence mode="wait" initial={false}>
            {/* Layer 0: Episodes Overview */}
            {navigation.layer === "overview" && (
              <motion.div
                key="overview"
                {...getAnimationVariant}
                className="h-full overflow-auto absolute inset-0 p-4"
              >
                <EpisodeQualityPanel datasetId={navigation.datasetId} />
                <EpisodeOverview
                  datasetId={navigation.datasetId}
                  onSelectEpisode={handleSelectEpisode}
                />

                {/* Procedural Memory Toggle */}
                <div className="mt-6 pt-4 border-t border-[var(--border-subtle)]">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider">
                        Procedural Memory
                      </label>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 font-medium">
                        Experimental
                      </span>
                    </div>
                    <button
                      onClick={() => setShowProcedural(!showProcedural)}
                      className={`relative w-9 h-5 rounded-full transition-colors ${
                        showProcedural ? "bg-[var(--accent)]" : "bg-[var(--bg-elevated)]"
                      }`}
                    >
                      <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform shadow-sm ${
                        showProcedural ? "translate-x-4" : ""
                      }`} />
                    </button>
                  </div>

                  {showProcedural && (
                    <>
                      <div className="mb-3 p-2 rounded bg-amber-500/5 border border-amber-500/15">
                        <div className="flex items-start gap-1.5">
                          <AlertTriangle size={11} className="text-amber-400 mt-0.5 flex-shrink-0" />
                          <p className="text-[10px] text-amber-200/70 leading-relaxed">
                            Procedural retrieval is in testing. Understand parsing and usage patterns before production use.
                          </p>
                        </div>
                      </div>
                      <ProcedureOverviewList
                        datasetId={navigation.datasetId}
                        onSelect={handleSelectProcedure}
                      />
                      <RemoveProceduralButton datasetId={navigation.datasetId} />
                    </>
                  )}
                </div>
              </motion.div>
            )}

            {/* Layer 1: Episode Subgraph */}
            {navigation.layer === "episode" && navigation.episodeId && (
              <motion.div
                key={`episode-${navigation.episodeId}`}
                {...getAnimationVariant}
                className="h-full absolute inset-0"
              >
                <EpisodeSubgraph
                  episodeId={navigation.episodeId}
                  episodeName={navigation.episodeName}
                  datasetId={navigation.datasetId}
                  onSelectFacet={handleSelectFacet}
                  onSelectEntity={handleSelectEntity}
                  onDeleted={() => {
                    setNavigationDirection("back");
                    setHistory([]);
                    setNavigation({ layer: "overview", datasetId: navigation.datasetId });
                  }}
                />
              </motion.div>
            )}

            {/* Layer 2: Facet Subgraph */}
            {navigation.layer === "facet" && navigation.facetId && (
              <motion.div
                key={`facet-${navigation.facetId}`}
                {...getAnimationVariant}
                className="h-full absolute inset-0"
              >
                <FacetSubgraph
                  facetId={navigation.facetId}
                  facetName={navigation.facetName}
                  datasetId={navigation.datasetId}
                  onSelectEntity={handleSelectEntity}
                />
              </motion.div>
            )}

            {/* Layer: Procedure Subgraph */}
            {navigation.layer === "procedure" && navigation.procedureId && (
              <motion.div
                key={`procedure-${navigation.procedureId}`}
                {...getAnimationVariant}
                className="h-full absolute inset-0"
              >
                <ProcedureSubgraph
                  procedureId={navigation.procedureId}
                  datasetId={navigation.datasetId}
                  onBack={goBack}
                />
              </motion.div>
            )}

            {/* Layer 3: Entity Network */}
            {navigation.layer === "entity" && navigation.entityId && (
              <motion.div
                key={`entity-${navigation.entityId}`}
                {...getAnimationVariant}
                className="h-full absolute inset-0"
              >
                <EntityNetwork
                  entityId={navigation.entityId}
                  entityName={navigation.entityName}
                  datasetId={navigation.datasetId}
                  onSelectEpisode={handleEntityToEpisode}
                  onSelectFacet={handleEntityToFacet}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Keyboard hint with animation */}
        <AnimatePresence>
          {navigation.layer !== "overview" && (
            <motion.div
              variants={hintVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              className="mt-2 text-[11px] text-[var(--text-muted)] text-center"
            >
              Press{" "}
              <kbd className="px-1.5 py-0.5 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] font-mono">
                ESC
              </kbd>{" "}
              to go back
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </LayoutGroup>
  );
}
