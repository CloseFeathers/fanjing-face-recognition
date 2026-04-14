"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, X, Loader2, Trash2, Info } from "lucide-react";
import { apiClient, DeletionPreview, DeletionResult } from "@/lib/api/client";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

interface DeleteConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  nodeId: string;
  nodeType: string;
  nodeName?: string;
  datasetId: string;
  onSuccess?: (result: DeletionResult) => void;
}

type DeletionMode = "soft" | "hard";

export function DeleteConfirmDialog({
  isOpen,
  onClose,
  nodeId,
  nodeType,
  nodeName,
  datasetId,
  onSuccess,
}: DeleteConfirmDialogProps) {
  const [preview, setPreview] = useState<DeletionPreview | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [mode, setMode] = useState<DeletionMode>("soft");

  const queryClient = useQueryClient();

  const loadPreview = useCallback(async () => {
    if (!nodeId || !datasetId) return;
    
    setLoadingPreview(true);
    setPreviewError(null);
    
    try {
      const data = await apiClient.previewNodeDeletion(nodeId, datasetId);
      setPreview(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load preview";
      setPreviewError(message);
    } finally {
      setLoadingPreview(false);
    }
  }, [nodeId, datasetId]);

  useEffect(() => {
    if (isOpen && nodeId) {
      loadPreview();
      setMode("soft");
    } else {
      setPreview(null);
      setPreviewError(null);
    }
  }, [isOpen, nodeId, loadPreview]);

  const deleteMutation = useMutation({
    mutationFn: async () => {
      if (nodeType === "Episode") {
        return apiClient.deleteEpisode(nodeId, datasetId, mode);
      }
      return apiClient.deleteNode(nodeId, datasetId, mode === "hard");
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["graph"] });
      queryClient.invalidateQueries({ queryKey: ["episodeSubgraph"] });
      queryClient.invalidateQueries({ queryKey: ["episodesOverview"] });
      queryClient.invalidateQueries({ queryKey: ["episodeQuality"] });
      queryClient.invalidateQueries({ queryKey: ["dashboardStats"] });
      queryClient.invalidateQueries({ queryKey: ["facetSubgraph"] });
      queryClient.invalidateQueries({ queryKey: ["entityNetwork"] });
      
      toast.success(
        mode === "hard"
          ? `Deleted ${result.deleted_count} node(s)`
          : "Node deleted successfully"
      );
      
      onSuccess?.(result);
      onClose();
    },
    onError: (error: Error) => {
      const message = error.message || "Deletion failed";
      
      if (message.includes("Permission") || message.includes("403")) {
        toast.error("Permission denied: " + message);
      } else if (message.includes("not found") || message.includes("404")) {
        toast.error("Node not found or already deleted");
      } else {
        toast.error("Deletion failed: " + message);
      }
    },
  });

  useEffect(() => {
    if (!isOpen) return;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !deleteMutation.isPending) {
        onClose();
      }
    };
    
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose, deleteMutation.isPending]);

  const handleDelete = () => {
    deleteMutation.mutate();
  };

  const isEpisode = nodeType === "Episode";

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
        >
          <div 
            className="absolute inset-0 bg-black/60" 
            onClick={() => !deleteMutation.isPending && onClose()} 
          />
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="relative bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg shadow-xl max-w-md w-full mx-4 overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-[var(--border-subtle)]">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-red-500/10">
                  <Trash2 size={18} className="text-red-500" />
                </div>
                <h3 className="text-sm font-medium text-[var(--text-primary)]">
                  Delete {nodeType}
                </h3>
              </div>
              <button
                onClick={onClose}
                disabled={deleteMutation.isPending}
                className="p-1 rounded hover:bg-[var(--bg-elevated)] text-[var(--text-muted)] transition-colors disabled:opacity-50"
              >
                <X size={16} />
              </button>
            </div>

            {/* Content */}
            <div className="p-4 space-y-4">
              {/* Node Info */}
              <div className="text-sm text-[var(--text-secondary)]">
                <p>
                  Are you sure you want to delete{" "}
                  <span className="font-medium text-[var(--text-primary)]">
                    {nodeName || nodeId}
                  </span>
                  ?
                </p>
              </div>

              {/* Preview Loading */}
              {loadingPreview && (
                <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
                  <Loader2 size={14} className="animate-spin" />
                  Loading deletion preview...
                </div>
              )}

              {/* Preview Error */}
              {previewError && (
                <div className="p-3 rounded bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                  {previewError}
                </div>
              )}

              {/* Preview Info */}
              {preview && !loadingPreview && (
                <div className="space-y-3">
                  {/* Warning */}
                  <div className="flex items-start gap-2 p-3 rounded bg-amber-500/10 border border-amber-500/20">
                    <AlertTriangle size={16} className="text-amber-500 mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-amber-300">{preview.warning}</p>
                  </div>

                  {/* Edge Info */}
                  {preview.edge_count > 0 && (
                    <div className="flex items-start gap-2 p-3 rounded bg-[var(--bg-elevated)]">
                      <Info size={16} className="text-[var(--text-muted)] mt-0.5 flex-shrink-0" />
                      <div className="text-sm text-[var(--text-secondary)]">
                        <p>Connected to {preview.edge_count} edge(s)</p>
                        {preview.neighbor_types.length > 0 && (
                          <p className="text-xs text-[var(--text-muted)] mt-1">
                            Types: {preview.neighbor_types.join(", ")}
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Mode Selection (Episodes only) */}
              {isEpisode && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
                    Deletion Mode
                  </p>
                  <div className="space-y-2">
                    <label className="flex items-start gap-3 p-3 rounded border border-[var(--border-subtle)] cursor-pointer hover:bg-[var(--bg-elevated)] transition-colors">
                      <input
                        type="radio"
                        name="deleteMode"
                        value="soft"
                        checked={mode === "soft"}
                        onChange={() => setMode("soft")}
                        className="mt-0.5"
                      />
                      <div>
                        <p className="text-sm font-medium text-[var(--text-primary)]">
                          Soft Delete
                        </p>
                        <p className="text-xs text-[var(--text-muted)]">
                          Only delete this Episode. Facets and Entities may still be used by other Episodes.
                        </p>
                      </div>
                    </label>
                    <label className="flex items-start gap-3 p-3 rounded border border-[var(--border-subtle)] cursor-pointer hover:bg-[var(--bg-elevated)] transition-colors">
                      <input
                        type="radio"
                        name="deleteMode"
                        value="hard"
                        checked={mode === "hard"}
                        onChange={() => setMode("hard")}
                        className="mt-0.5"
                      />
                      <div>
                        <p className="text-sm font-medium text-[var(--text-primary)]">
                          Hard Delete
                        </p>
                        <p className="text-xs text-[var(--text-muted)]">
                          Also clean up orphan Facets, FacetPoints, and Entities with no remaining connections.
                        </p>
                      </div>
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex justify-end gap-2 p-4 border-t border-[var(--border-subtle)] bg-[var(--bg-elevated)]">
              <button
                onClick={onClose}
                disabled={deleteMutation.isPending}
                className="px-4 py-2 text-sm text-[var(--text-secondary)] bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded hover:bg-[var(--bg-hover)] transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleteMutation.isPending || loadingPreview || preview?.can_delete === false}
                className="px-4 py-2 text-sm text-white bg-red-600 hover:bg-red-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                title={preview?.can_delete === false ? "This node type cannot be deleted directly" : undefined}
              >
                {deleteMutation.isPending ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 size={14} />
                    Delete
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
