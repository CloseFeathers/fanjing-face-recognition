"use client";

import React, { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useGraph, useDatasets } from "@/hooks/use-api";
import { getActionableErrorMessage } from "@/components/common";
import { toast } from "sonner";
import { Download, Loader2, FileJson, FileText, AlertCircle } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

// ============================================================================
// API Example
// ============================================================================

const API_EXAMPLE = `from m_flow import MemoryExporter
from pydantic import BaseModel

class MyMemory(BaseModel):
    content: str
    category: str
    entities: list[str]

exporter = MemoryExporter(schema=MyMemory)
data = await exporter.export(
    dataset_name="my_dataset",
    format="json",
)`;

// ============================================================================
// Types & Constants
// ============================================================================

type ExportFormat = "json" | "jsonl" | "csv";

const FORMATS: Array<{ id: ExportFormat; name: string; desc: string; icon: React.ReactNode }> = [
  { id: "json", name: "JSON", desc: "Structured, for programmatic use", icon: <FileJson size={16} /> },
  { id: "jsonl", name: "JSON Lines", desc: "One record per line, streaming", icon: <FileText size={16} /> },
  { id: "csv", name: "CSV", desc: "Tabular, for spreadsheets", icon: <FileText size={16} /> },
];

// ============================================================================
// Components
// ============================================================================

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="relative group">
      <pre className="bg-[#0a0a0a] border border-[var(--border-subtle)] rounded p-3 overflow-x-auto">
        <code className="text-xs text-[var(--text-secondary)] font-mono">{code}</code>
      </pre>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 px-2 py-1 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

// ============================================================================
// Export Utilities
// ============================================================================

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function formatAsJson(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

interface ExportNode { id: string; name?: string; label?: string; type?: string; datasetId?: string; properties?: Record<string, unknown> }
interface ExportEdge { source: string; target: string; relationship: string; properties?: Record<string, unknown> }

function toExportEdges(edges: Array<{ source: string | { id: string }; target: string | { id: string }; relationship: string; properties?: Record<string, unknown> }>): ExportEdge[] {
  return edges.map(edge => ({
    source: typeof edge.source === "string" ? edge.source : edge.source.id,
    target: typeof edge.target === "string" ? edge.target : edge.target.id,
    relationship: edge.relationship,
    properties: edge.properties,
  }));
}

function flattenNode(node: ExportNode): Record<string, string> {
  const row: Record<string, string> = {
    id: node.id,
    name: node.name || node.label || "",
    type: node.type || "",
    dataset_id: node.datasetId || "",
  };
  if (node.properties) {
    for (const [k, v] of Object.entries(node.properties)) {
      if (k === "metadata" || v === null || v === undefined) continue;
      row[k] = typeof v === "object" ? JSON.stringify(v) : String(v);
    }
  }
  return row;
}

function formatAsJsonl(nodes: ExportNode[], edges: ExportEdge[], includeNodes: boolean, includeEdges: boolean): string {
  const lines: string[] = [];
  if (includeNodes) {
    for (const node of nodes) {
      lines.push(JSON.stringify({ _type: "node", ...node }));
    }
  }
  if (includeEdges) {
    for (const edge of edges) {
      lines.push(JSON.stringify({ _type: "edge", ...edge }));
    }
  }
  return lines.join("\n");
}

function formatAsCsv(nodes: ExportNode[], edges: ExportEdge[], includeNodes: boolean, includeEdges: boolean): string {
  const parts: string[] = [];

  if (includeNodes && nodes.length > 0) {
    const rows = nodes.map(flattenNode);
    const allKeys = new Set<string>();
    for (const r of rows) Object.keys(r).forEach((k) => allKeys.add(k));
    const headers = Array.from(allKeys);
    const csvEscape = (v: string) => `"${v.replace(/"/g, '""').replace(/\n/g, " ")}"`;
    parts.push(headers.map(csvEscape).join(","));
    for (const r of rows) {
      parts.push(headers.map((h) => csvEscape(r[h] || "")).join(","));
    }
  }

  if (includeEdges && edges.length > 0) {
    if (parts.length > 0) parts.push("");
    const edgeHeaders = ["source", "target", "relationship"];
    const csvEscape = (v: string) => `"${v.replace(/"/g, '""')}"`;
    parts.push(edgeHeaders.map(csvEscape).join(","));
    for (const e of edges) {
      parts.push([
        csvEscape(typeof e.source === "string" ? e.source : (e.source as ExportNode).id),
        csvEscape(typeof e.target === "string" ? e.target : (e.target as ExportNode).id),
        csvEscape(e.relationship || ""),
      ].join(","));
    }
  }

  return parts.join("\n");
}

// ============================================================================
// Main Page
// ============================================================================

export function ExportPage() {
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>("json");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [includeNodes, setIncludeNodes] = useState(true);
  const [includeLinks, setIncludeLinks] = useState(true);
  const [isExporting, setIsExporting] = useState(false);

  const { data: datasets } = useDatasets();
  
  // Use dataset filter - pass undefined for "all datasets"
  const effectiveDatasetId = selectedDatasetId || undefined;
  const { data: graphData, isLoading: graphLoading, error: graphError, refetch: refetchGraph } = useGraph(effectiveDatasetId);
  
  // Get selected dataset name for display
  const selectedDatasetName = useMemo(() => {
    if (!selectedDatasetId) return "All datasets";
    const ds = datasets?.find(d => d.id === selectedDatasetId);
    return ds?.name || selectedDatasetId;
  }, [selectedDatasetId, datasets]);

  const handleExport = async () => {
    setIsExporting(true);

    try {
      // Refetch to get latest data
      const result = await refetchGraph();
      const data = result.data;

      if (!data || (!data.nodes?.length && !data.edges?.length)) {
        toast.error("No data to export");
        setIsExporting(false);
        return;
      }

      const exportData: Record<string, unknown> = {};
      if (includeNodes) {
        exportData.nodes = data.nodes;
      }
      if (includeLinks) {
        exportData.edges = data.edges;
      }

      let content: string;
      let filename: string;
      let mimeType: string;

      const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

      switch (selectedFormat) {
        case "json":
          content = formatAsJson(exportData);
          filename = `mflow-export-${timestamp}.json`;
          mimeType = "application/json";
          break;
        case "jsonl":
          content = formatAsJsonl(data.nodes || [], toExportEdges(data.edges || []), includeNodes, includeLinks);
          filename = `mflow-export-${timestamp}.jsonl`;
          mimeType = "application/jsonlines";
          break;
        case "csv":
          content = formatAsCsv(data.nodes || [], toExportEdges(data.edges || []), includeNodes, includeLinks);
          filename = `mflow-export-${timestamp}.csv`;
          mimeType = "text/csv";
          break;
      }

      downloadFile(content, filename, mimeType);
      toast.success(`Exported ${data.nodes?.length || 0} nodes`);
    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Export failed");
      toast.error(actionableMessage);
      console.error("Export error:", error);
    } finally {
      setIsExporting(false);
    }
  };

  const nodeCount = graphData?.nodes?.length || 0;
  const linkCount = graphData?.edges?.length || 0;

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Export Knowledge Graph</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Export your knowledge graph data in various formats for backup, analysis, or integration.
        </p>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Config */}
        <div className="col-span-7 space-y-5">
          {/* Data Summary */}
          <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-[var(--text-muted)] uppercase tracking-wider">Current Data</span>
              {graphLoading && <Loader2 size={12} className="animate-spin text-[var(--text-muted)]" />}
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-2xl font-semibold text-[var(--text-primary)]">{nodeCount}</p>
                <p className="text-xs text-[var(--text-muted)]">Nodes</p>
              </div>
              <div>
                <p className="text-2xl font-semibold text-[var(--text-primary)]">{linkCount}</p>
                <p className="text-xs text-[var(--text-muted)]">Relationships</p>
              </div>
            </div>
            {graphError && (
              <div className="mt-3 flex items-center gap-2 text-xs text-[var(--error)]">
                <AlertCircle size={12} />
                Failed to load graph data
              </div>
            )}
          </div>

          {/* Format Selection */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Format
            </label>
            <div className="grid grid-cols-3 gap-2">
              {FORMATS.map((f) => (
                <button
                  key={f.id}
                  onClick={() => setSelectedFormat(f.id)}
                  disabled={isExporting}
                  className={cn(
                    "p-3 rounded border text-left transition-colors",
                    selectedFormat === f.id
                      ? "border-[var(--text-primary)] bg-[var(--bg-surface)]"
                      : "border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
                    isExporting && "opacity-50 cursor-not-allowed"
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[var(--text-muted)]">{f.icon}</span>
                    <span className="text-xs font-medium text-[var(--text-primary)]">{f.name}</span>
                  </div>
                  <p className="text-[10px] text-[var(--text-muted)]">{f.desc}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Dataset Selection */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Dataset
            </label>
            <select
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              disabled={isExporting}
              className={cn(
                "w-full bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--text-muted)]",
                isExporting && "opacity-50 cursor-not-allowed"
              )}
            >
              <option value="">All datasets</option>
              {datasets?.map((ds) => (
                <option key={ds.id} value={ds.id}>{ds.name}</option>
              ))}
            </select>
            <p className="text-[10px] text-[var(--text-muted)] mt-1">
              {selectedDatasetId 
                ? `Exporting from: ${selectedDatasetName}` 
                : "Export data from all accessible datasets"}
            </p>
          </div>

          {/* Options */}
          <div className="space-y-3">
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block">
              Include
            </label>
            <div className="flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
              <div>
                <Label className="text-xs text-[var(--text-secondary)]">Nodes (Entities)</Label>
                <p className="text-[10px] text-[var(--text-muted)]">Episodes, Facets, Concepts</p>
              </div>
              <Switch 
                checked={includeNodes} 
                onCheckedChange={setIncludeNodes} 
                disabled={isExporting || !includeLinks}
              />
            </div>
            <div className="flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
              <div>
                <Label className="text-xs text-[var(--text-secondary)]">Relationships</Label>
                <p className="text-[10px] text-[var(--text-muted)]">Edges between entities</p>
              </div>
              <Switch 
                checked={includeLinks} 
                onCheckedChange={setIncludeLinks} 
                disabled={isExporting || !includeNodes}
              />
            </div>
          </div>

          {/* Submit */}
          <button
            onClick={handleExport}
            disabled={isExporting || graphLoading || (!includeNodes && !includeLinks) || nodeCount === 0}
            className="w-full py-2.5 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isExporting ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download size={14} />
                Export {nodeCount} Nodes
              </>
            )}
          </button>
        </div>

        {/* Right: Reference */}
        <div className="col-span-5 space-y-5">
          {/* API Example */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Python API
            </label>
            <CodeBlock code={API_EXAMPLE} />
          </div>

          {/* Use Cases */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Use Cases
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-3 space-y-1 text-xs text-[var(--text-secondary)]">
              <p>• Data backup and migration</p>
              <p>• Integration with other systems</p>
              <p>• Data analysis and visualization</p>
              <p>• Model fine-tuning data preparation</p>
            </div>
          </div>

          {/* Format Details */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Format Details
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              <div className="px-3 py-2">
                <code className="text-xs text-[var(--text-primary)]">JSON</code>
                <p className="text-[10px] text-[var(--text-muted)] mt-0.5">Complete graph structure with nodes and edges</p>
              </div>
              <div className="px-3 py-2">
                <code className="text-xs text-[var(--text-primary)]">JSONL</code>
                <p className="text-[10px] text-[var(--text-muted)] mt-0.5">One node per line, good for streaming</p>
              </div>
              <div className="px-3 py-2">
                <code className="text-xs text-[var(--text-primary)]">CSV</code>
                <p className="text-[10px] text-[var(--text-muted)] mt-0.5">Tabular format, nodes only</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
