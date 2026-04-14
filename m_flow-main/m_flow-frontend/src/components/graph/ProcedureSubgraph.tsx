"use client";

import React, { useEffect, useRef, useCallback, useState, useMemo } from "react";
import { Loader2, AlertCircle, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import * as d3 from "d3";
import { useQuery } from "@tanstack/react-query";
import type { GraphNode, GraphLink } from "@/types";
import { Button } from "@/components/ui/button";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const NODE_COLORS: Record<string, { base: string; hover: string }> = {
  Procedure:             { base: "rgba(255,255,255,0.85)", hover: "#ffffff" },
  ProcedureStepPoint:    { base: "rgba(160,160,160,0.7)",  hover: "rgba(200,200,200,0.9)" },
  ProcedureContextPoint: { base: "rgba(100,100,100,0.6)",  hover: "rgba(140,140,140,0.8)" },
};

const DEFAULT_COLOR = { base: "rgba(80,80,80,0.4)", hover: "rgba(120,120,120,0.6)" };

function getColor(type: string, hovered: boolean) {
  const c = NODE_COLORS[type] || DEFAULT_COLOR;
  return hovered ? c.hover : c.base;
}

function getSize(type: string, isCenter: boolean) {
  if (isCenter) return 18;
  if (type === "ProcedureStepPoint") return 11;
  if (type === "ProcedureContextPoint") return 10;
  return 13;
}

interface SimNode extends GraphNode, d3.SimulationNodeDatum {
  fx?: number | null;
  fy?: number | null;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  relationship: string;
}

interface Props {
  procedureId: string;
  datasetId?: string;
  onBack: () => void;
  onSelectNode?: (node: GraphNode) => void;
}

function cleanLabel(raw: string): string {
  let name = raw
    .replace(/^point:\s*[\d-]+\s*\|\s*/i, "")
    .replace(/^(when|why|boundary|outcome|prereq|exception|misc):\s*/i, "")
    .replace(/^auto_coverage$/i, "")
    .replace(/\\n\d+/g, "")
    .trim();
  if (!name) return "";
  return name.length > 28 ? name.slice(0, 25) + "…" : name;
}

export function ProcedureSubgraph({ procedureId, datasetId, onBack, onSelectNode }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [isStable, setIsStable] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ["procedure-subgraph", procedureId, datasetId],
    queryFn: async () => {
      const token = typeof window !== "undefined" ? localStorage.getItem("mflow_token") || "" : "";
      const params = new URLSearchParams();
      if (datasetId) params.set("dataset_id", datasetId);
      const resp = await fetch(
        `${API_BASE}/api/v1/graph/procedure/${procedureId}?${params}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!resp.ok) throw new Error(`${resp.status}`);
      return resp.json() as Promise<{ nodes: GraphNode[]; edges: GraphLink[] }>;
    },
    enabled: !!procedureId,
    retry: 1,
  });

  const processedData = useMemo(() => {
    if (!data) return null;
    const nodes: SimNode[] = data.nodes.map((n) => ({ ...n }));
    const links: SimLink[] = data.edges.map((e) => ({
      source: typeof e.source === "string" ? e.source : e.source,
      target: typeof e.target === "string" ? e.target : e.target,
      relationship: e.relationship,
    }));
    return { nodes, links };
  }, [data]);

  const handleZoomIn = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 1.4);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 0.7);
    }
  }, []);

  const handleFitView = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(500).call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  useEffect(() => {
    if (!processedData || !svgRef.current || !containerRef.current) return;

    const { nodes, links } = processedData;
    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);
    setIsStable(false);

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    const g = svg.append("g");

    // Pin the center Procedure node
    const centerNode = nodes.find((n) => n.id === procedureId);
    if (centerNode) {
      centerNode.fx = width / 2;
      centerNode.fy = height / 2;
    }

    const simulation = d3.forceSimulation<SimNode>(nodes)
      .force(
        "link",
        d3.forceLink<SimNode, SimLink>(links)
          .id((d) => d.id)
          .distance(100)
          .strength(0.8)
      )
      .force("charge", d3.forceManyBody().strength(-300).distanceMax(300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30))
      .alphaDecay(0.02)
      .velocityDecay(0.3);

    // Glow filter
    const defs = svg.append("defs");
    const filter = defs.append("filter")
      .attr("id", "proc-glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Links
    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "rgba(255,255,255,0.08)")
      .attr("stroke-width", 1.5)
      .attr("stroke-linecap", "round");

    const linkLabel = g.append("g")
      .selectAll("text")
      .data(links)
      .enter().append("text")
      .attr("font-size", 8)
      .attr("fill", "rgba(255,255,255,0.18)")
      .attr("text-anchor", "middle")
      .attr("dy", -4)
      .style("pointer-events", "none")
      .text((d: any) => {
        const r = d.relationship || "";
        return r.replace("has_key_point", "step").replace("has_context_point", "context").replace("supersedes", "supersedes");
      });

    // Nodes
    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", (d) => getSize(d.type || "", d.id === procedureId))
      .attr("fill", (d) => getColor(d.type || "", false))
      .attr("stroke", (d) => d.id === procedureId ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.1)")
      .attr("stroke-width", (d) => d.id === procedureId ? 2.5 : 1)
      .style("cursor", "pointer")
      .style("filter", (d) => d.id === procedureId ? "url(#proc-glow)" : "none")
      .on("mouseenter", function (_e, d) {
        if (d.id === procedureId) return;
        d3.select(this).transition().duration(150)
          .attr("r", getSize(d.type || "", false) * 1.3)
          .attr("fill", getColor(d.type || "", true))
          .style("filter", "url(#proc-glow)");
        link.transition().duration(150)
          .attr("stroke", (l: any) => {
            const sId = typeof l.source === "string" ? l.source : l.source.id;
            const tId = typeof l.target === "string" ? l.target : l.target.id;
            return sId === d.id || tId === d.id ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.04)";
          })
          .attr("stroke-width", (l: any) => {
            const sId = typeof l.source === "string" ? l.source : l.source.id;
            const tId = typeof l.target === "string" ? l.target : l.target.id;
            return sId === d.id || tId === d.id ? 2.5 : 1;
          });
      })
      .on("mouseleave", function (_e, d) {
        if (d.id === procedureId) return;
        d3.select(this).transition().duration(150)
          .attr("r", getSize(d.type || "", false))
          .attr("fill", getColor(d.type || "", false))
          .style("filter", "none");
        link.transition().duration(150)
          .attr("stroke", "rgba(255,255,255,0.08)")
          .attr("stroke-width", 1.5);
      })
      .on("click", (_e, d) => {
        setSelectedNode(d as GraphNode);
        if (onSelectNode) onSelectNode(d as GraphNode);
      })
      .call(
        d3.drag<SVGCircleElement, SimNode>()
          .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); if (d.id !== procedureId) { d.fx = d.x; d.fy = d.y; } })
          .on("drag", (e, d) => { if (d.id !== procedureId) { d.fx = e.x; d.fy = e.y; } })
          .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); if (d.id !== procedureId) { d.fx = null; d.fy = null; } })
      );

    // Labels
    const label = g.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .attr("font-size", (d) => d.id === procedureId ? 12 : 10)
      .attr("font-weight", (d) => d.id === procedureId ? 600 : 400)
      .attr("fill", (d) => d.id === procedureId ? "#ffffff" : "#808080")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => d.id === procedureId ? 34 : 24)
      .style("pointer-events", "none")
      .style("text-shadow", "0 1px 2px rgba(0,0,0,0.5)")
      .text((d) => cleanLabel(d.name || d.properties?.search_text as string || d.type || "Node"));

    // Tick
    let frameId: number;
    simulation.on("tick", () => {
      cancelAnimationFrame(frameId);
      frameId = requestAnimationFrame(() => {
        link
          .attr("x1", (d: any) => d.source.x).attr("y1", (d: any) => d.source.y)
          .attr("x2", (d: any) => d.target.x).attr("y2", (d: any) => d.target.y);
        linkLabel
          .attr("x", (d: any) => (d.source.x + d.target.x) / 2)
          .attr("y", (d: any) => (d.source.y + d.target.y) / 2);
        node.attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y);
        label.attr("x", (d: any) => d.x).attr("y", (d: any) => d.y);
      });
    });

    simulation.on("end", () => setIsStable(true));

    return () => { simulation.stop(); cancelAnimationFrame(frameId); };
  }, [processedData, procedureId, onSelectNode]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <Loader2 size={24} className="animate-spin text-[var(--text-muted)]" />
        <p className="text-[13px] text-[var(--text-muted)] mt-3">Loading procedure graph...</p>
      </div>
    );
  }

  if (error || !data?.nodes.length) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2">
        <AlertCircle size={24} className="text-[var(--text-muted)]" />
        <p className="text-[13px] text-[var(--text-muted)]">No procedure subgraph data found</p>
        <button onClick={onBack} className="text-xs text-[var(--accent)] hover:underline">Back</button>
      </div>
    );
  }

  const nodeCount = data.nodes.length;
  const linkCount = data.edges.length;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="px-2 py-1 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-secondary)] rounded border border-[var(--border-subtle)] hover:text-[var(--text-primary)]"
          >
            ← Back
          </button>
          <span className="text-[10px] text-[var(--text-muted)] font-mono uppercase tracking-wider">Procedure Subgraph</span>
          <span className="text-[12px] text-[var(--text-muted)]">
            {nodeCount} nodes · {linkCount} edges
            {!isStable && <span className="ml-2 animate-pulse">(stabilizing...)</span>}
          </span>
        </div>
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

      {/* Graph container — same pattern as EpisodeSubgraph */}
      <div
        ref={containerRef}
        className="flex-1 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded-lg overflow-hidden relative"
      >
        <svg ref={svgRef} className="w-full h-full" />
      </div>

      {/* Legend */}
      <div className="mt-3 flex items-center gap-4 text-[11px] text-[var(--text-muted)]">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full border-2 border-white/30" style={{ backgroundColor: "rgba(255,255,255,0.85)" }} />
          <span>Procedure</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: "rgba(160,160,160,0.7)" }} />
          <span>Step</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: "rgba(100,100,100,0.6)" }} />
          <span>Context</span>
        </div>
      </div>

      {/* Node detail panel */}
      {selectedNode && (() => {
        const p = (selectedNode.properties ?? {}) as Record<string, string>;
        return (
        <div className="absolute top-14 right-3 w-72 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg shadow-lg overflow-hidden z-20">
          <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border-subtle)]">
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">{selectedNode.type}</span>
            <button onClick={() => setSelectedNode(null)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] text-xs">✕</button>
          </div>
          <div className="p-3 space-y-2.5 max-h-80 overflow-y-auto">
            <div>
              <p className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Name</p>
              <p className="text-[13px] text-[var(--text-primary)] font-medium leading-snug">
                {cleanLabel(selectedNode.name || selectedNode.type || "") || selectedNode.type}
              </p>
            </div>
            {p.search_text && (
              <div>
                <p className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Search Text</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">{p.search_text}</p>
              </div>
            )}
            {p.summary && (
              <div>
                <p className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Summary</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed line-clamp-6">{p.summary}</p>
              </div>
            )}
            {p.context_text && (
              <div>
                <p className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Context</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap line-clamp-4">{p.context_text}</p>
              </div>
            )}
            {p.points_text && (
              <div>
                <p className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Key Points</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap line-clamp-6">{p.points_text}</p>
              </div>
            )}
            {p.description && (
              <div>
                <p className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">Description</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed line-clamp-4">{p.description}</p>
              </div>
            )}
            <div>
              <p className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider mb-0.5">ID</p>
              <code className="text-[9px] text-[var(--text-muted)] font-mono bg-[var(--bg-elevated)] px-1.5 py-0.5 rounded block truncate">{selectedNode.id}</code>
            </div>
          </div>
        </div>
        );
      })()}
    </div>
  );
}
