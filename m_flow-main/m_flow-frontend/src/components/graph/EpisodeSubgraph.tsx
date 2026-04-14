"use client";

import React, { useEffect, useRef, useCallback, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, AlertCircle, ZoomIn, ZoomOut, Maximize2, ChevronDown, FileText, Trash2 } from "lucide-react";
import * as d3 from "d3";
import { useEpisodeGraph } from "@/hooks/use-api";
import { GraphNode, GraphLink } from "@/types";
import { Button } from "@/components/ui/button";
import { DeleteConfirmDialog } from "./DeleteConfirmDialog";

// Parse episode summary into sections
interface Section {
  heading: string;
  content: string;
}

function parseSummaryIntoSections(summary: string): Section[] {
  if (!summary) return [];
  
  // Match pattern: 【heading】content
  const sectionRegex = /【([^】]+)】([^【]*)/g;
  const sections: Section[] = [];
  let match;
  
  while ((match = sectionRegex.exec(summary)) !== null) {
    const heading = match[1].trim();
    const content = match[2].trim();
    if (heading || content) {
      sections.push({ heading, content });
    }
  }
  
  // If no sections found, return the whole summary as a single section
  if (sections.length === 0 && summary.trim()) {
    return [{ heading: "", content: summary.trim() }];
  }
  
  return sections;
}

// Formatted summary component with section structure
function FormattedSummary({ summary }: { summary: string }) {
  const sections = useMemo(() => parseSummaryIntoSections(summary), [summary]);
  
  if (sections.length === 0) {
    return (
      <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed">
        {summary}
      </p>
    );
  }
  
  // Single section without heading - just show content
  if (sections.length === 1 && !sections[0].heading) {
    return (
      <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
        {sections[0].content}
      </p>
    );
  }
  
  return (
    <div className="space-y-2.5">
      {sections.map((section, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: index * 0.03 }}
        >
          {/* Section heading */}
          {section.heading && (
            <h4 className="text-[11px] font-medium text-[var(--text-secondary)] mb-1">
              {section.heading}
            </h4>
          )}
          
          {/* Section content */}
          {section.content && (
            <p className="text-[11px] text-[var(--text-muted)] leading-relaxed whitespace-pre-wrap">
              {section.content}
            </p>
          )}
        </motion.div>
      ))}
    </div>
  );
}

interface SimulationNode extends GraphNode, d3.SimulationNodeDatum {
  fx?: number | null;
  fy?: number | null;
}

interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
  relationship: string;
}

interface EpisodeSubgraphProps {
  episodeId: string;
  episodeName?: string;
  datasetId?: string;
  onSelectFacet: (facet: GraphNode) => void;
  onSelectEntity: (entity: GraphNode) => void;
  onDeleted?: () => void;
}

// Tooltip component for node hover
function NodeTooltip({ 
  node, 
  position 
}: { 
  node: GraphNode | null; 
  position: { x: number; y: number } | null;
}) {
  if (!node || !position) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 5 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 5 }}
      transition={{ duration: 0.15 }}
      className="absolute pointer-events-none z-50 px-3 py-2 rounded-lg shadow-lg
        bg-[var(--bg-elevated)] border border-[var(--border-default)]
        max-w-[200px]"
      style={{
        left: position.x,
        top: position.y - 50,
        transform: "translateX(-50%)",
      }}
    >
      <p className="text-[12px] font-medium text-[var(--text-primary)] truncate">
        {node.name || node.id.slice(0, 16)}
      </p>
      <p className="text-[10px] text-[var(--text-muted)]">
        {node.type} • Click to explore
      </p>
    </motion.div>
  );
}

/**
 * Layer 1: Episode Subgraph
 * Displays the episode at center with connected Facets and Entities.
 * Optimized with D3 rendering and interactive hover effects.
 */
export function EpisodeSubgraph({
  episodeId,
  episodeName,
  datasetId,
  onSelectFacet,
  onSelectEntity,
  onDeleted,
}: EpisodeSubgraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimulationNode, SimulationLink> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [isSimulationStable, setIsSimulationStable] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const { data: graph, isLoading, error } = useEpisodeGraph(episodeId, datasetId);

  // Extract episode display content from the center node
  // Prefers display_only (full content) over summary (indexed abstract)
  const episodeSummary = useMemo(() => {
    if (!graph?.nodes) return null;
    const episodeNode = graph.nodes.find(n => n.id === episodeId);
    if (!episodeNode) return null;
    
    let props = episodeNode.properties;
    if (typeof props === "string") {
      try {
        props = JSON.parse(props);
      } catch {
        props = {};
      }
    }
    const p = props as Record<string, unknown>;
    return (p?.display_only as string) || (p?.summary as string) || null;
  }, [graph, episodeId]);

  // Node color by type with enhanced palette
  const getNodeColor = useCallback((type: string, isHovered: boolean): string => {
    const colors: Record<string, { base: string; hover: string }> = {
      Episode: { base: "#ffffff", hover: "#ffffff" },
      Facet: { base: "#a0a0a0", hover: "#c0c0c0" },
      Entity: { base: "#666666", hover: "#888888" },
    };
    const color = colors[type] || { base: "#404040", hover: "#606060" };
    return isHovered ? color.hover : color.base;
  }, []);

  // Get node size based on type
  const getNodeSize = useCallback((type: string, id: string): number => {
    if (id === episodeId) return 18;
    if (type === "Facet") return 12;
    return 10;
  }, [episodeId]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 1.4);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 0.7);
    }
  }, []);

  const handleFitView = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(500)
        .call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  // Memoize graph data processing
  const processedData = useMemo(() => {
    if (!graph) return null;
    
    const nodes: SimulationNode[] = graph.nodes.map((n) => ({ ...n }));
    const links: SimulationLink[] = graph.edges.map((e) => ({
      source: typeof e.source === "string" ? e.source : e.source.id,
      target: typeof e.target === "string" ? e.target : e.target.id,
      relationship: e.relationship,
    }));
    
    return { nodes, links };
  }, [graph]);

  // Render graph with optimizations
  useEffect(() => {
    if (!processedData || !svgRef.current || !containerRef.current) return;

    const { nodes, links } = processedData;
    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Clear previous
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);
    setIsSimulationStable(false);

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    const g = svg.append("g");

    // Find center node (the episode)
    const centerNode = nodes.find((n) => n.id === episodeId);
    if (centerNode) {
      centerNode.fx = width / 2;
      centerNode.fy = height / 2;
    }

    // Force simulation with optimized parameters
    const simulation = d3.forceSimulation<SimulationNode>(nodes)
      .force(
        "link",
        d3.forceLink<SimulationNode, SimulationLink>(links)
          .id((d) => d.id)
          .distance(100)
          .strength(0.8)
      )
      .force("charge", d3.forceManyBody().strength(-350).distanceMax(300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(35))
      .alphaDecay(0.02)
      .velocityDecay(0.3);

    simulationRef.current = simulation;

    // Draw links with gradient
    const linkGroup = g.append("g").attr("class", "links");
    
    const link = linkGroup
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "rgba(255, 255, 255, 0.1)")
      .attr("stroke-width", 1.5)
      .attr("stroke-linecap", "round");

    // Link labels (hidden by default, shown on hover)
    const linkLabel = g
      .append("g")
      .attr("class", "link-labels")
      .selectAll("text")
      .data(links)
      .enter()
      .append("text")
      .text((d) => d.relationship || "")
      .attr("font-size", 9)
      .attr("fill", "#404040")
      .attr("text-anchor", "middle")
      .attr("dy", -4)
      .style("opacity", 0.5)
      .style("pointer-events", "none");

    // Draw nodes with glow effect
    const nodeGroup = g.append("g").attr("class", "nodes");

    // Node glow filter
    const defs = svg.append("defs");
    const filter = defs.append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
    filter.append("feGaussianBlur")
      .attr("stdDeviation", "3")
      .attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Node circles
    const node = nodeGroup
      .selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("r", (d) => getNodeSize(d.type, d.id))
      .attr("fill", (d) => getNodeColor(d.type, false))
      .attr("stroke", (d) =>
        d.id === episodeId ? "#ffffff" : "rgba(255, 255, 255, 0.15)"
      )
      .attr("stroke-width", (d) => (d.id === episodeId ? 3 : 1.5))
      .style("cursor", (d) => (d.id === episodeId ? "default" : "pointer"))
      .style("filter", (d) => (d.id === episodeId ? "url(#glow)" : "none"))
      .on("mouseenter", function (event, d) {
        if (d.id === episodeId) return;
        
        // Update visual
        d3.select(this)
          .transition()
          .duration(150)
          .attr("r", getNodeSize(d.type, d.id) * 1.3)
          .attr("fill", getNodeColor(d.type, true))
          .style("filter", "url(#glow)");
        
        // Show tooltip
        const rect = containerRef.current?.getBoundingClientRect();
        if (rect) {
          setHoveredNode(d);
          setTooltipPosition({
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
          });
        }

        // Highlight connected links
        link
          .transition()
          .duration(150)
          .attr("stroke", (l) => {
            const sourceId = typeof l.source === "string" ? l.source : (l.source as SimulationNode).id;
            const targetId = typeof l.target === "string" ? l.target : (l.target as SimulationNode).id;
            return sourceId === d.id || targetId === d.id
              ? "rgba(255, 255, 255, 0.4)"
              : "rgba(255, 255, 255, 0.05)";
          })
          .attr("stroke-width", (l) => {
            const sourceId = typeof l.source === "string" ? l.source : (l.source as SimulationNode).id;
            const targetId = typeof l.target === "string" ? l.target : (l.target as SimulationNode).id;
            return sourceId === d.id || targetId === d.id ? 2.5 : 1;
          });
      })
      .on("mouseleave", function (event, d) {
        if (d.id === episodeId) return;
        
        d3.select(this)
          .transition()
          .duration(150)
          .attr("r", getNodeSize(d.type, d.id))
          .attr("fill", getNodeColor(d.type, false))
          .style("filter", "none");
        
        setHoveredNode(null);
        setTooltipPosition(null);

        // Reset links
        link
          .transition()
          .duration(150)
          .attr("stroke", "rgba(255, 255, 255, 0.1)")
          .attr("stroke-width", 1.5);
      })
      .on("click", (event, d) => {
        event.stopPropagation();
        if (d.id === episodeId) return;
        
        // Click animation
        d3.select(event.currentTarget)
          .transition()
          .duration(100)
          .attr("r", getNodeSize(d.type, d.id) * 0.8)
          .transition()
          .duration(100)
          .attr("r", getNodeSize(d.type, d.id));

        if (d.type === "Facet") {
          onSelectFacet(d);
        } else if (d.type === "Entity" || d.type === "Entity") {
          onSelectEntity(d);
        }
      })
      .call(
        d3
          .drag<SVGCircleElement, SimulationNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            if (d.id !== episodeId) {
              d.fx = d.x;
              d.fy = d.y;
            }
          })
          .on("drag", (event, d) => {
            if (d.id !== episodeId) {
              d.fx = event.x;
              d.fy = event.y;
            }
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            if (d.id !== episodeId) {
              d.fx = null;
              d.fy = null;
            }
          })
      );

    // Node labels with better readability
    const nodeLabel = g
      .append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(nodes)
      .enter()
      .append("text")
      .text((d) => {
        const name = d.name || d.id;
        return name.length > 14 ? name.slice(0, 14) + "…" : name;
      })
      .attr("font-size", (d) => (d.id === episodeId ? 12 : 10))
      .attr("font-weight", (d) => (d.id === episodeId ? 600 : 400))
      .attr("fill", (d) => (d.id === episodeId ? "#ffffff" : "#808080"))
      .attr("text-anchor", "middle")
      .attr("dy", (d) => (d.id === episodeId ? 34 : 26))
      .style("pointer-events", "none")
      .style("text-shadow", "0 1px 2px rgba(0,0,0,0.5)");

    // Update positions with requestAnimationFrame optimization
    let frameId: number;
    simulation.on("tick", () => {
      cancelAnimationFrame(frameId);
      frameId = requestAnimationFrame(() => {
        link
          .attr("x1", (d) => (d.source as SimulationNode).x!)
          .attr("y1", (d) => (d.source as SimulationNode).y!)
          .attr("x2", (d) => (d.target as SimulationNode).x!)
          .attr("y2", (d) => (d.target as SimulationNode).y!);

        linkLabel
          .attr("x", (d) =>
            ((d.source as SimulationNode).x! + (d.target as SimulationNode).x!) / 2
          )
          .attr("y", (d) =>
            ((d.source as SimulationNode).y! + (d.target as SimulationNode).y!) / 2
          );

        node.attr("cx", (d) => d.x!).attr("cy", (d) => d.y!);
        nodeLabel.attr("x", (d) => d.x!).attr("y", (d) => d.y!);
      });
    });

    // Mark simulation as stable after it settles
    simulation.on("end", () => {
      setIsSimulationStable(true);
    });

    return () => {
      simulation.stop();
      cancelAnimationFrame(frameId);
    };
  }, [processedData, episodeId, getNodeColor, getNodeSize, onSelectFacet, onSelectEntity]);

  if (isLoading) {
    return (
      <motion.div 
        className="flex flex-col items-center justify-center h-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <Loader2 size={24} className="text-[var(--text-muted)]" />
        </motion.div>
        <motion.p 
          className="text-[13px] text-[var(--text-muted)] mt-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Loading episode graph...
        </motion.p>
      </motion.div>
    );
  }

  if (error) {
    return (
      <motion.div 
        className="flex flex-col items-center justify-center h-full"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <AlertCircle size={24} className="text-[var(--error)] mb-3" />
        <p className="text-[14px] text-[var(--text-primary)] mb-1">
          Failed to load episode
        </p>
        <p className="text-[13px] text-[var(--text-muted)]">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
      </motion.div>
    );
  }

  const nodeCount = graph?.nodes.length || 0;
  const linkCount = graph?.edges.length || 0;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
    >
      {/* Episode Summary Panel - Compact toggle style */}
      {episodeSummary && (
        <div className="mb-2">
          <button
            onClick={() => setShowSummary(!showSummary)}
            className="inline-flex items-center gap-1.5 px-2 py-1 text-[11px] text-[var(--text-secondary)] 
                       hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] 
                       rounded transition-colors"
          >
            <FileText size={12} />
            <span>Summary</span>
            <ChevronDown 
              size={12} 
              className={`transition-transform duration-200 ${showSummary ? 'rotate-180' : ''}`} 
            />
          </button>
          <AnimatePresence>
            {showSummary && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="overflow-hidden"
              >
                <div className="mt-2 pl-2 border-l-2 border-[var(--border-subtle)] max-h-48 overflow-auto">
                  <FormattedSummary summary={episodeSummary} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Header with stats and controls */}
      <div className="mb-2 flex items-center justify-between">
        <div className="text-[12px] text-[var(--text-muted)]">
          {nodeCount} nodes · {linkCount} edges
          {!isSimulationStable && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="ml-2"
            >
              (stabilizing...)
            </motion.span>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleZoomIn}
            className="h-7 w-7 p-0"
            title="Zoom in"
          >
            <ZoomIn size={14} className="text-[var(--text-muted)]" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleZoomOut}
            className="h-7 w-7 p-0"
            title="Zoom out"
          >
            <ZoomOut size={14} className="text-[var(--text-muted)]" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleFitView}
            className="h-7 w-7 p-0"
            title="Fit view"
          >
            <Maximize2 size={14} className="text-[var(--text-muted)]" />
          </Button>
          {datasetId && (
            <>
              <div className="w-px h-4 bg-[var(--border-subtle)] mx-1" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDeleteDialog(true)}
                className="h-7 w-7 p-0 hover:bg-red-500/10"
                title="Delete Episode"
              >
                <Trash2 size={14} className="text-[var(--text-muted)] hover:text-red-500" />
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Graph Container */}
      <div
        ref={containerRef}
        className="flex-1 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded-lg overflow-hidden relative"
      >
        <svg ref={svgRef} className="w-full h-full" />
        
        {/* Tooltip */}
        <AnimatePresence>
          {hoveredNode && (
            <NodeTooltip node={hoveredNode} position={tooltipPosition} />
          )}
        </AnimatePresence>
      </div>

      {/* Legend with animation */}
      <motion.div 
        className="mt-3 flex items-center gap-4 text-[11px] text-[var(--text-muted)]"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-3.5 h-3.5 rounded-full border-2 border-white/30"
            style={{ backgroundColor: "#ffffff" }}
            whileHover={{ scale: 1.2 }}
          />
          <span>Episode</span>
        </div>
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: "#a0a0a0" }}
            whileHover={{ scale: 1.2 }}
          />
          <span>Facet</span>
        </div>
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: "#666666" }}
            whileHover={{ scale: 1.2 }}
          />
          <span>Entity</span>
        </div>
      </motion.div>

      {/* Delete Confirmation Dialog */}
      {datasetId && (
        <DeleteConfirmDialog
          isOpen={showDeleteDialog}
          onClose={() => setShowDeleteDialog(false)}
          nodeId={episodeId}
          nodeType="Episode"
          nodeName={episodeName}
          datasetId={datasetId}
          onSuccess={() => {
            setShowDeleteDialog(false);
            onDeleted?.();
          }}
        />
      )}
    </motion.div>
  );
}
