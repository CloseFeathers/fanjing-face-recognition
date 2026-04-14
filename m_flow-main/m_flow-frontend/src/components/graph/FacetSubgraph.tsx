"use client";

import React, { useEffect, useRef, useCallback, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, AlertCircle, ZoomIn, ZoomOut, Maximize2, ChevronDown, ChevronUp, Hash } from "lucide-react";
import * as d3 from "d3";
import { useFacetGraph } from "@/hooks/use-api";
import { GraphNode, GraphLink } from "@/types";
import { NodeDetailPanel } from "./NodeDetailPanel";
import { Button } from "@/components/ui/button";

interface SimulationNode extends GraphNode, d3.SimulationNodeDatum {
  fx?: number | null;
  fy?: number | null;
}

interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
  relationship: string;
}

interface FacetSubgraphProps {
  facetId: string;
  facetName?: string;
  datasetId?: string;
  onSelectEntity: (entity: GraphNode) => void;
}

// Tooltip for node hover
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
        {node.type} • Click to {node.type === "FacetPoint" ? "view details" : "explore"}
      </p>
    </motion.div>
  );
}

/**
 * Layer 2: Facet Subgraph
 * Displays the facet at center with connected FacetPoints and Entities.
 * Optimized with consistent animations and hover effects.
 */
export function FacetSubgraph({
  facetId,
  facetName,
  datasetId,
  onSelectEntity,
}: FacetSubgraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimulationNode, SimulationLink> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const [isSimulationStable, setIsSimulationStable] = useState(false);
  const [showDescription, setShowDescription] = useState(true);

  const { data: graph, isLoading, error } = useFacetGraph(facetId, datasetId);

  // Extract facet display content from the center node
  // Prefers display_only (full content) over description/anchor_text
  const facetDescription = useMemo(() => {
    if (!graph?.nodes) return null;
    const facetNode = graph.nodes.find(n => n.id === facetId);
    if (!facetNode) return null;
    
    let props = facetNode.properties;
    if (typeof props === "string") {
      try {
        props = JSON.parse(props);
      } catch {
        props = {};
      }
    }
    const p = props as Record<string, unknown>;
    return (p?.display_only as string) || (p?.description as string) || (p?.anchor_text as string) || null;
  }, [graph, facetId]);

  // Node color by type with hover states
  const getNodeColor = useCallback((type: string, isHovered: boolean): string => {
    const colors: Record<string, { base: string; hover: string }> = {
      Facet: { base: "#ffffff", hover: "#ffffff" },
      FacetPoint: { base: "#b0b0b0", hover: "#d0d0d0" },
      Entity: { base: "#666666", hover: "#888888" },
      ContentFragment: { base: "#e89040", hover: "#f0a860" },
      MemorySpace: { base: "#9b6dd7", hover: "#b48be8" },
      EntityType: { base: "#d96ba0", hover: "#e88dba" },
    };
    const color = colors[type] || { base: "#404040", hover: "#606060" };
    return isHovered ? color.hover : color.base;
  }, []);

  // Get node size based on type
  const getNodeSize = useCallback((type: string, id: string): number => {
    if (id === facetId) return 16;
    if (type === "FacetPoint") return 9;
    return 11;
  }, [facetId]);

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

  // Memoize graph data
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

  // Render graph
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

    // Find center node (the facet)
    const centerNode = nodes.find((n) => n.id === facetId);
    if (centerNode) {
      centerNode.fx = width / 2;
      centerNode.fy = height / 2;
    }

    // Force simulation
    const simulation = d3.forceSimulation<SimulationNode>(nodes)
      .force(
        "link",
        d3.forceLink<SimulationNode, SimulationLink>(links)
          .id((d) => d.id)
          .distance(90)
          .strength(0.8)
      )
      .force("charge", d3.forceManyBody().strength(-300).distanceMax(250))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30))
      .alphaDecay(0.02)
      .velocityDecay(0.3);

    simulationRef.current = simulation;

    // Glow filter
    const defs = svg.append("defs");
    const filter = defs.append("filter")
      .attr("id", "facet-glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
    filter.append("feGaussianBlur")
      .attr("stdDeviation", "2.5")
      .attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Draw links
    const link = g
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "rgba(255, 255, 255, 0.1)")
      .attr("stroke-width", 1.5)
      .attr("stroke-linecap", "round");

    // Link labels
    const linkLabel = g
      .append("g")
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

    // Draw nodes
    const node = g
      .append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("r", (d) => getNodeSize(d.type, d.id))
      .attr("fill", (d) => getNodeColor(d.type, false))
      .attr("stroke", (d) =>
        d.id === facetId ? "#ffffff" : "rgba(255, 255, 255, 0.15)"
      )
      .attr("stroke-width", (d) => (d.id === facetId ? 3 : 1.5))
      .style("cursor", (d) => (d.id === facetId ? "default" : "pointer"))
      .style("filter", (d) => (d.id === facetId ? "url(#facet-glow)" : "none"))
      .on("mouseenter", function (event, d) {
        if (d.id === facetId) return;
        
        d3.select(this)
          .transition()
          .duration(150)
          .attr("r", getNodeSize(d.type, d.id) * 1.3)
          .attr("fill", getNodeColor(d.type, true))
          .style("filter", "url(#facet-glow)");
        
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
              ? "rgba(255, 255, 255, 0.35)"
              : "rgba(255, 255, 255, 0.05)";
          });
      })
      .on("mouseleave", function (event, d) {
        if (d.id === facetId) return;
        
        d3.select(this)
          .transition()
          .duration(150)
          .attr("r", getNodeSize(d.type, d.id))
          .attr("fill", getNodeColor(d.type, false))
          .style("filter", "none");
        
        setHoveredNode(null);
        setTooltipPosition(null);

        link
          .transition()
          .duration(150)
          .attr("stroke", "rgba(255, 255, 255, 0.1)");
      })
      .on("click", (event, d) => {
        event.stopPropagation();
        if (d.id === facetId) return;
        
        // Click animation
        d3.select(event.currentTarget)
          .transition()
          .duration(100)
          .attr("r", getNodeSize(d.type, d.id) * 0.8)
          .transition()
          .duration(100)
          .attr("r", getNodeSize(d.type, d.id));

        if (d.type === "Entity" || d.type === "Entity") {
          onSelectEntity(d);
        } else {
          setSelectedNode(d);
        }
      })
      .call(
        d3
          .drag<SVGCircleElement, SimulationNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            if (d.id !== facetId) {
              d.fx = d.x;
              d.fy = d.y;
            }
          })
          .on("drag", (event, d) => {
            if (d.id !== facetId) {
              d.fx = event.x;
              d.fy = event.y;
            }
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            if (d.id !== facetId) {
              d.fx = null;
              d.fy = null;
            }
          })
      );

    // Node labels
    const nodeLabel = g
      .append("g")
      .selectAll("text")
      .data(nodes)
      .enter()
      .append("text")
      .text((d) => {
        const name = d.name || d.id;
        const maxLen = d.id === facetId ? 18 : 12;
        return name.length > maxLen ? name.slice(0, maxLen) + "…" : name;
      })
      .attr("font-size", (d) => (d.id === facetId ? 12 : 9))
      .attr("font-weight", (d) => (d.id === facetId ? 600 : 400))
      .attr("fill", (d) => (d.id === facetId ? "#ffffff" : "#707070"))
      .attr("text-anchor", "middle")
      .attr("dy", (d) => (d.id === facetId ? 30 : 22))
      .style("pointer-events", "none")
      .style("text-shadow", "0 1px 2px rgba(0,0,0,0.5)");

    // Update positions with RAF
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

    simulation.on("end", () => {
      setIsSimulationStable(true);
    });

    // Click on background to deselect
    svg.on("click", () => {
      setSelectedNode(null);
    });

    return () => {
      simulation.stop();
      cancelAnimationFrame(frameId);
    };
  }, [processedData, facetId, getNodeColor, getNodeSize, onSelectEntity]);

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
          Loading facet graph...
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
          Failed to load facet
        </p>
        <p className="text-[13px] text-[var(--text-muted)]">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
      </motion.div>
    );
  }

  const nodeCount = graph?.nodes.length || 0;
  const linkCount = graph?.edges.length || 0;
  const hasEntities = graph?.nodes.some((n) => n.type === "Entity" || n.type === "Entity");
  const hasFacetPoints = graph?.nodes.some((n) => n.type === "FacetPoint");

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
    >
      {/* Facet Description Panel */}
      {facetDescription && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
          className="mb-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg overflow-hidden"
        >
          <button
            onClick={() => setShowDescription(!showDescription)}
            className="w-full flex items-center justify-between px-4 py-2 hover:bg-[var(--bg-hover)] transition-colors"
          >
            <div className="flex items-center gap-2 text-[12px] text-[var(--text-muted)]">
              <Hash size={14} />
              <span>Facet Description</span>
            </div>
            {showDescription ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          <AnimatePresence>
            {showDescription && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-3 max-h-48 overflow-auto">
                  <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
                    {facetDescription}
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* Header */}
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

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={handleZoomIn} className="h-7 w-7 p-0" title="Zoom in">
            <ZoomIn size={14} className="text-[var(--text-muted)]" />
          </Button>
          <Button variant="ghost" size="sm" onClick={handleZoomOut} className="h-7 w-7 p-0" title="Zoom out">
            <ZoomOut size={14} className="text-[var(--text-muted)]" />
          </Button>
          <Button variant="ghost" size="sm" onClick={handleFitView} className="h-7 w-7 p-0" title="Fit view">
            <Maximize2 size={14} className="text-[var(--text-muted)]" />
          </Button>
        </div>
      </div>

      {/* Graph Container */}
      <div
        ref={containerRef}
        className="flex-1 relative bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded-lg overflow-hidden"
      >
        <svg ref={svgRef} className="w-full h-full" />

        {/* Tooltip */}
        <AnimatePresence>
          {hoveredNode && !selectedNode && (
            <NodeTooltip node={hoveredNode} position={tooltipPosition} />
          )}
        </AnimatePresence>

        {/* Node Detail Panel */}
        <AnimatePresence>
          {selectedNode && (
            <NodeDetailPanel
              node={selectedNode}
              onClose={() => setSelectedNode(null)}
              onNavigateToEntity={(entityId: string) => {
                const entity = processedData?.nodes.find(n => n.id === entityId);
                if (entity) onSelectEntity(entity);
              }}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Legend */}
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
          <span>Facet (center)</span>
        </div>
        {hasFacetPoints && (
          <div className="flex items-center gap-1.5">
            <motion.div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: "#b0b0b0" }}
              whileHover={{ scale: 1.2 }}
            />
            <span>FacetPoint (click for details)</span>
          </div>
        )}
        {hasEntities && (
          <div className="flex items-center gap-1.5">
            <motion.div
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: "#666666" }}
              whileHover={{ scale: 1.2 }}
            />
            <span>Entity (click to explore)</span>
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: "#e89040" }}
            whileHover={{ scale: 1.2 }}
          />
          <span>ContentFragment</span>
        </div>
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: "#9b6dd7" }}
            whileHover={{ scale: 1.2 }}
          />
          <span>MemorySpace</span>
        </div>
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: "#d96ba0" }}
            whileHover={{ scale: 1.2 }}
          />
          <span>EntityType</span>
        </div>
      </motion.div>
    </motion.div>
  );
}
