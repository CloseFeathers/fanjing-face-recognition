"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { X, Copy, ExternalLink, FileText, Users, Hash, Pencil, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { usePatchNode } from "@/hooks/use-api";
import { GraphNode } from "@/types";

interface NodeDetailPanelProps {
  node: GraphNode | null;
  onClose: () => void;
  onNavigateToEpisode?: (episodeId: string) => void;
  onNavigateToFacet?: (facetId: string) => void;
  onNavigateToEntity?: (entityId: string) => void;
}

/**
 * Side panel showing detailed information about a selected node.
 * Supports Episode, Facet, FacetPoint, and Entity node types.
 */
export function NodeDetailPanel({
  node,
  onClose,
  onNavigateToEpisode,
  onNavigateToFacet,
  onNavigateToEntity,
}: NodeDetailPanelProps) {
  const [displayOnly, setDisplayOnly] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const patchNode = usePatchNode();

  if (!node) return null;

  const handleCopyId = () => {
    navigator.clipboard.writeText(node.id);
    toast.success("ID copied to clipboard");
  };

  const handleNavigate = () => {
    switch (node.type) {
      case "Episode":
        onNavigateToEpisode?.(node.id);
        break;
      case "Facet":
        onNavigateToFacet?.(node.id);
        break;
      case "Entity":
      case "Entity":
        onNavigateToEntity?.(node.id);
        break;
    }
  };

  // Get node-specific icon
  const getNodeIcon = () => {
    switch (node.type) {
      case "Episode":
        return <FileText size={14} />;
      case "Facet":
        return <Hash size={14} />;
      case "Entity":
      case "Entity":
        return <Users size={14} />;
      default:
        return <FileText size={14} />;
    }
  };

  // Get node color based on type
  const getNodeColor = () => {
    switch (node.type) {
      case "Episode":
        return "#ffffff";
      case "Facet":
        return "#a0a0a0";
      case "Entity":
      case "Entity":
        return "#666666";
      case "Procedure":
        return "#8c6edc";
      case "ProcedureStepPoint":
        return "#64a0f0";
      case "ProcedureContextPoint":
        return "#a08cd8";
      case "ContentFragment":
        return "#e89040";
      case "MemorySpace":
        return "#9b6dd7";
      case "EntityType":
        return "#d96ba0";
      default:
        return "#404040";
    }
  };

  // Check if node can be navigated to
  const canNavigate =
    (node.type === "Episode" && onNavigateToEpisode) ||
    (node.type === "Facet" && onNavigateToFacet) ||
    ((node.type === "Entity" || node.type === "Entity") && onNavigateToEntity);

  return (
    <motion.div
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 16 }}
      transition={{ duration: 0.15 }}
      className="absolute top-4 right-4 w-72 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg shadow-lg overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-subtle)]">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: getNodeColor() }}
          />
          <span className="text-[12px] text-[var(--text-muted)] uppercase tracking-wider">
            {node.type}
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors rounded"
        >
          <X size={14} />
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Name */}
        <div>
          <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
            Name
          </p>
          <p className="text-[14px] text-[var(--text-primary)] font-medium leading-snug">
            {node.name || `${node.type}_${node.id.slice(0, 8)}`}
          </p>
        </div>

        {/* Display Text (full content, not indexed) */}
        {node.properties?.display_only != null && (
          <div>
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
              Content
            </p>
            <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed line-clamp-6">
              {String(node.properties.display_only)}
            </p>
          </div>
        )}

        {/* Summary/Description if available */}
        {(node.summary || node.description) && (
          <div>
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
              Summary
            </p>
            <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed line-clamp-4">
              {node.summary || node.description}
            </p>
          </div>
        )}

        {/* Procedure-specific fields */}
        {node.type === "Procedure" && node.properties && (
          <>
            {node.properties.search_text != null && (
              <div>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Search Text</p>
                <p className="text-[12px] text-[var(--text-secondary)]">{String(node.properties.search_text)}</p>
              </div>
            )}
            {node.properties.context_text != null && (
              <div>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Context</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap line-clamp-6">{String(node.properties.context_text)}</p>
              </div>
            )}
            {node.properties.points_text != null && (
              <div>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Steps / Key Points</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap line-clamp-8">{String(node.properties.points_text)}</p>
              </div>
            )}
            <div className="flex gap-3 text-[10px] text-[var(--text-muted)]">
              {node.properties.version != null && <span>v{String(node.properties.version)}</span>}
              {node.properties.status != null && <span>{String(node.properties.status)}</span>}
              {node.properties.confidence != null && <span>{String(node.properties.confidence)}</span>}
            </div>
          </>
        )}

        {/* ProcedureStepPoint / ProcedureContextPoint */}
        {(node.type === "ProcedureStepPoint" || node.type === "ProcedureContextPoint") && node.properties && (
          <>
            {node.properties.search_text != null && (
              <div>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Search Text</p>
                <p className="text-[12px] text-[var(--text-secondary)]">{String(node.properties.search_text)}</p>
              </div>
            )}
            {node.properties.description != null && (
              <div>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Description</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap line-clamp-6">{String(node.properties.description)}</p>
              </div>
            )}
          </>
        )}

        {/* ID */}
        <div>
          <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
            ID
          </p>
          <div className="flex items-center gap-2">
            <code className="flex-1 text-[11px] text-[var(--text-muted)] font-mono bg-[var(--bg-elevated)] px-2 py-1 rounded truncate">
              {node.id}
            </code>
            <button
              onClick={handleCopyId}
              className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              title="Copy ID"
            >
              <Copy size={12} />
            </button>
          </div>
        </div>

        {/* Properties */}
        {node.properties && Object.keys(node.properties).length > 0 && (
          <div>
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
              Properties
            </p>
            <div className="bg-[var(--bg-elevated)] rounded p-2 max-h-32 overflow-auto">
              <pre className="text-[10px] text-[var(--text-muted)] font-mono whitespace-pre-wrap">
                {JSON.stringify(node.properties, (key, value) => {
                  if (
                    (key === "mentioned_time_start_ms" || key === "mentioned_time_end_ms") &&
                    typeof value === "number"
                  ) {
                    return new Date(value).toLocaleString("sv-SE", {
                      year: "numeric", month: "2-digit", day: "2-digit",
                      hour: "2-digit", minute: "2-digit",
                    }).replace(",", "");
                  }
                  return value;
                }, 2)}
              </pre>
            </div>
          </div>
        )}

        {/* Dataset ID if available */}
        {node.dataset_id && (
          <div>
            <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
              Dataset
            </p>
            <code className="text-[11px] text-[var(--text-muted)] font-mono">
              {node.dataset_id.slice(0, 8)}...
            </code>
          </div>
        )}
      </div>

      {/* Edit */}
      <div className="px-4 py-3 border-t border-[var(--border-subtle)] space-y-2">
        <button
          onClick={() => {
            setIsEditing((prev) => !prev);
            if (!isEditing) {
              setDisplayOnly(node.properties?.display_only != null ? String(node.properties.display_only) : (node.name ?? ""));
            }
          }}
          className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
        >
          <Pencil size={11} />
          <span>Edit Display Name</span>
        </button>
        {isEditing && (
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={displayOnly}
              onChange={(e) => setDisplayOnly(e.target.value)}
              className="flex-1 text-[12px] px-2 py-1.5 rounded border border-[var(--border-subtle)] bg-[var(--bg-elevated)] text-[var(--text-primary)] outline-none focus:border-[var(--text-muted)]"
              placeholder="display_only"
            />
            <Button
              variant="outline"
              size="sm"
              disabled={patchNode.isPending}
              onClick={() => {
                patchNode.mutate(
                  { node_id: node.id, node_type: node.type, display_only: displayOnly },
                  {
                    onSuccess: () => {
                      toast.success("Node updated");
                      setIsEditing(false);
                    },
                    onError: (err) => {
                      toast.error(`Update failed: ${err.message}`);
                    },
                  }
                );
              }}
              className="text-[11px] px-2"
            >
              {patchNode.isPending ? <Loader2 size={12} className="animate-spin" /> : "Save"}
            </Button>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="px-4 py-3 border-t border-[var(--border-subtle)] flex gap-2">
        {canNavigate && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleNavigate}
            className="flex-1 text-[12px]"
          >
            {getNodeIcon()}
            <span className="ml-1.5">
              {node.type === "Episode"
                ? "View Episode"
                : node.type === "Facet"
                ? "View Facet"
                : "View Entity"}
            </span>
          </Button>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopyId}
          className="text-[12px]"
        >
          <Copy size={12} />
        </Button>
      </div>
    </motion.div>
  );
}
