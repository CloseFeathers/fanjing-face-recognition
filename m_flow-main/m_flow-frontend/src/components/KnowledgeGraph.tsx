"use client";

import { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import type { GraphData, GraphNode, GraphLink } from "@/types";

interface Props {
  data: GraphData;
  onNodeClick?: (node: GraphNode) => void;
  selectedNode?: string | null;
  viewLevel: "episodes" | "episode" | "facet";
}

// Type for D3 simulation node
interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

// Type for D3 simulation link
interface SimLink {
  source: SimNode;
  target: SimNode;
  relationship: string;
}

export function ExtractedGraph({ data, onNodeClick, selectedNode, viewLevel }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  // Filter nodes based on view level
  const filterNodes = useCallback((nodes: GraphNode[], level: string): GraphNode[] => {
    if (level === "episodes") {
      // Outer layer: only Episode, Facet, Entity (no FacetPoint)
      return nodes.filter(n => n.type !== "facet_point");
    } else if (level === "episode") {
      // Episode subgraph: Facet, FacetPoint, Entity
      return nodes.filter(n => n.type !== "episode");
    } else if (level === "facet") {
      // Facet subgraph: FacetPoint, Entity
      return nodes.filter(n => n.type === "facet_point" || n.type === "entity" || n.type === "facet");
    }
    return nodes;
  }, []);

  // Filter links to only include those with valid nodes
  const filterLinks = useCallback((links: GraphLink[], nodeIds: Set<string>): GraphLink[] => {
    return links.filter(link => {
      const sourceId = typeof link.source === "string" ? link.source : link.source.id;
      const targetId = typeof link.target === "string" ? link.target : link.target.id;
      return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });
  }, []);

  const getNodeColor = useCallback((node: GraphNode, isSelected: boolean) => {
    if (isSelected) return "#ffffff";
    switch (node.type) {
      case "episode":
        return "#ffffff";
      case "facet":
        return "#aaaaaa";
      case "facet_point":
        return "#666666";
      case "entity":
        return "#444444";
      default:
        return "#444444";
    }
  }, []);

  const getNodeRadius = useCallback((node: GraphNode) => {
    switch (node.type) {
      case "episode":
        return 45;  // Largest
      case "facet":
        return 28;  // Medium
      case "facet_point":
        return 16;  // Small
      case "entity":
        return 8;   // Smallest
      default:
        return 12;
    }
  }, []);


  const renderGraph = useCallback(() => {
    if (!svgRef.current || !containerRef.current) return;

    // Filter nodes based on view level
    const filteredNodes = filterNodes(data.nodes, viewLevel);
    if (filteredNodes.length === 0) return;

    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredLinks = filterLinks(data.edges, nodeIds);

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg.attr("width", width).attr("height", height);

    // Create container group for zoom/pan
    const g = svg.append("g");

    // Add zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 3])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Initialize node positions - spread out in a grid pattern to avoid overlap
    const gridSize = Math.ceil(Math.sqrt(filteredNodes.length));
    const spacing = Math.min(width, height) / (gridSize + 1);
    
    const simNodes: SimNode[] = filteredNodes.map((node, i) => {
      const row = Math.floor(i / gridSize);
      const col = i % gridSize;
      return {
        ...node,
        x: width / 2 + (col - gridSize / 2) * spacing + (Math.random() - 0.5) * 20,
        y: height / 2 + (row - gridSize / 2) * spacing + (Math.random() - 0.5) * 20,
      };
    });

    // Create node lookup for links
    const nodeMap = new Map(simNodes.map(n => [n.id, n]));
    
    const simLinks: SimLink[] = filteredLinks.map(link => ({
      source: nodeMap.get(typeof link.source === "string" ? link.source : link.source.id)!,
      target: nodeMap.get(typeof link.target === "string" ? link.target : link.target.id)!,
      relationship: link.relationship,
    })).filter(l => l.source && l.target);

    // Stop any existing simulation
    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    // Create force simulation with stabilized parameters
    const simulation = d3
      .forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        d3.forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(100)
          .strength(0.3)
      )
      .force("charge", d3.forceManyBody().strength(-200).distanceMax(400))
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.1))
      .force("collision", d3.forceCollide<SimNode>().radius(d => getNodeRadius(d) + 15).strength(0.8))
      .force("x", d3.forceX(width / 2).strength(0.02))
      .force("y", d3.forceY(height / 2).strength(0.02))
      .alphaDecay(0.05)  // Faster decay for stability
      .velocityDecay(0.4);  // Higher damping to reduce shaking

    simulationRef.current = simulation;

    // Create links
    const link = g
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(simLinks)
      .enter()
      .append("line")
      .attr("stroke", "rgba(255, 255, 255, 0.08)")
      .attr("stroke-width", 0.5);

    // Track if we're dragging to distinguish from clicks
    let isDragging = false;

    // Drag behavior
    const dragBehavior = d3.drag<SVGGElement, SimNode>()
      .on("start", function(event, d) {
        isDragging = false;
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        d3.select(this).style("cursor", "grabbing");
      })
      .on("drag", function(event, d) {
        isDragging = true;
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", function(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
        d3.select(this).style("cursor", "grab");
        isDragging = false;
      });

    // Create node groups
    const node = g
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(simNodes)
      .enter()
      .append("g")
      .attr("class", d => `node node-${d.type}`)
      .style("cursor", "grab")
      .call(dragBehavior);

    // Node circles
    node
      .append("circle")
      .attr("r", d => getNodeRadius(d))
      .attr("fill", d => getNodeColor(d, d.id === selectedNode))
      .attr("stroke", d => {
        if (d.id === selectedNode) return "#ffffff";
        if (d.type === "episode") return "rgba(255, 255, 255, 0.3)";
        return "rgba(255, 255, 255, 0.15)";
      })
      .attr("stroke-width", d => d.type === "episode" ? 2 : 1);

    // Node type icons (only for larger nodes)
    node
      .filter(d => d.type !== "entity")
      .append("text")
      .attr("class", "node-icon")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("fill", d => {
        if (d.type === "episode") return "#000000";
        return "rgba(255, 255, 255, 0.9)";
      })
      .attr("font-size", d => {
        if (d.type === "episode") return "20px";
        if (d.type === "facet") return "14px";
        return "10px";
      })
      .attr("font-weight", "300")
      .text(d => {
        switch (d.type) {
          case "episode": return "◉";
          case "facet": return "◇";
          case "facet_point": return "·";
          default: return "";
        }
      });

    // Node labels - only for non-entity nodes by default
    node
      .append("text")
      .attr("class", "node-label")
      .attr("text-anchor", "middle")
      .attr("y", d => getNodeRadius(d) + 14)
      .attr("fill", d => {
        if (d.id === selectedNode) return "#ffffff";
        if (d.type === "entity") return "transparent";  // Hidden by default
        return "rgba(255, 255, 255, 0.65)";
      })
      .attr("font-size", d => {
        if (d.type === "episode") return "11px";
        if (d.type === "facet") return "10px";
        return "9px";
      })
      .attr("font-weight", d => d.type === "episode" ? "500" : "400")
      .attr("letter-spacing", "0.02em")
      .text(d => {
        const label = d.label ?? d.name ?? "";
        const maxLen = d.type === "episode" ? 16 : 14;
        return label.length > maxLen ? label.slice(0, maxLen) + "…" : label;
      });

    // Click handler - only trigger if not dragging
    node.on("click", (event, d) => {
      if (isDragging) return;
      event.stopPropagation();
      onNodeClick?.(d);
    });

    // Hover effects - pure D3, no React state
    node
      .on("mouseenter", function(_, d) {
        const nodeEl = d3.select(this);
        
        // Enlarge circle slightly
        nodeEl.select("circle")
          .transition()
          .duration(100)
          .attr("r", getNodeRadius(d) + 2)
          .attr("fill", "#ffffff")
          .attr("stroke", "#ffffff")
          .attr("stroke-width", 2);

        // Show label for entities (which are hidden by default)
        nodeEl.select(".node-label")
          .transition()
          .duration(100)
          .attr("fill", "#ffffff");
      })
      .on("mouseleave", function(_, d) {
        const nodeEl = d3.select(this);
        const isSelected = d.id === selectedNode;
        
        // Reset circle
        nodeEl.select("circle")
          .transition()
          .duration(100)
          .attr("r", getNodeRadius(d))
          .attr("fill", getNodeColor(d, isSelected))
          .attr("stroke", isSelected ? "#ffffff" : 
                d.type === "episode" ? "rgba(255, 255, 255, 0.3)" : "rgba(255, 255, 255, 0.15)")
          .attr("stroke-width", d.type === "episode" ? 2 : 1);

        // Hide entity labels again
        nodeEl.select(".node-label")
          .transition()
          .duration(100)
          .attr("fill", isSelected ? "#ffffff" : 
                d.type === "entity" ? "transparent" : "rgba(255, 255, 255, 0.65)");
      });

    // Update positions on tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Run simulation to stabilize layout first
    simulation.alpha(1);
    for (let i = 0; i < 300; i++) {
      simulation.tick();
    }
    // Keep simulation running but with low energy
    simulation.alpha(0.3).alphaDecay(0.02).restart();
    
    // Initial zoom to fit
    const initialZoom = d3.zoomIdentity.translate(0, 0).scale(0.8);
    svg.call(zoom.transform, initialZoom);

  }, [data, onNodeClick, selectedNode, viewLevel, getNodeColor, getNodeRadius, filterNodes, filterLinks]);

  useEffect(() => {
    renderGraph();

    const handleResize = () => renderGraph();
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [renderGraph]);

  const getLevelLabel = () => {
    switch (viewLevel) {
      case "episodes":
        return "Overview";
      case "episode":
        return "Episode View";
      case "facet":
        return "Facet View";
      default:
        return "";
    }
  };

  // Count node types for display
  const filteredNodes = filterNodes(data.nodes, viewLevel);
  const counts = {
    episodes: filteredNodes.filter(n => n.type === "episode").length,
    facets: filteredNodes.filter(n => n.type === "facet").length,
    points: filteredNodes.filter(n => n.type === "facet_point").length,
    entities: filteredNodes.filter(n => n.type === "entity").length,
  };

  return (
    <div ref={containerRef} className="knowledge-graph-container">
      <svg ref={svgRef} />
      {filteredNodes.length === 0 && (
        <div className="empty-graph">
          <div className="empty-icon">◯</div>
          <p>No graph data for this view</p>
          <p className="empty-hint">Try selecting a different dataset</p>
        </div>
      )}
      <div className="graph-level-indicator">
        <span className="level-label">{getLevelLabel()}</span>
        <span className="node-count">
          {counts.episodes > 0 && `${counts.episodes} Episodes · `}
          {counts.facets > 0 && `${counts.facets} Facets · `}
          {counts.points > 0 && `${counts.points} Points · `}
          {counts.entities > 0 && `${counts.entities} Entities`}
        </span>
      </div>
    </div>
  );
}
