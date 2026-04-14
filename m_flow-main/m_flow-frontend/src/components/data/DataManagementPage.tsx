"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import {
  Trash2, RefreshCw, Loader2, FileDown, Database, AlertCircle,
  CheckCircle, Clock,
} from "lucide-react";
import { useConfirm } from "@/components/ui/confirm-dialog";
import {
  useDatasetsWithCounts,
  useDeleteDocument,
  useSyncData,
  useSyncStatus,
  useDatasetsStatus,
} from "@/hooks/use-api";

export function DataManagementPage() {
  const { data: datasets, isLoading: dsLoading } = useDatasetsWithCounts({ refetchInterval: 10000 });
  const { data: syncStatus } = useSyncStatus();
  const { data: dsStatus } = useDatasetsStatus(datasets?.map(d => d.id));
  const deleteDoc = useDeleteDocument();
  const syncData = useSyncData();
  const confirm = useConfirm();

  const [deleteDataId, setDeleteDataId] = useState("");
  const [deleteDatasetId, setDeleteDatasetId] = useState("");
  const [deleteMode, setDeleteMode] = useState<"soft" | "hard">("soft");

  const handleDeleteDocument = async () => {
    if (!deleteDataId || !deleteDatasetId) {
      toast.error("Please provide both Data ID and Dataset ID");
      return;
    }
    const confirmed = await confirm({
      title: "Delete Document",
      message: `Delete document ${deleteDataId} from dataset? Mode: ${deleteMode}`,
      confirmText: "Delete",
      cancelText: "Cancel",
      variant: "danger",
    });
    if (!confirmed) return;
    try {
      const result = await deleteDoc.mutateAsync({
        data_id: deleteDataId,
        dataset_id: deleteDatasetId,
        mode: deleteMode,
      });
      toast.success(result.message || "Document deleted successfully");
      setDeleteDataId("");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Delete failed");
    }
  };

  const handleSync = async (datasetIds?: string[]) => {
    try {
      await syncData.mutateAsync({ dataset_ids: datasetIds });
      toast.success("Sync started");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Sync failed");
    }
  };

  const statusIcon = (status?: string) => {
    if (!status || status === "Unknown") return <Clock size={12} className="text-[var(--text-muted)]" />;
    const s = status.toLowerCase();
    if (s.includes("completed")) return <CheckCircle size={12} className="text-green-500" />;
    if (s.includes("error")) return <AlertCircle size={12} className="text-[var(--error)]" />;
    return <Loader2 size={12} className="text-blue-400 animate-spin" />;
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Data Management</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Delete documents, sync data, and monitor dataset processing status.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Delete Document */}
        <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Trash2 size={16} className="text-[var(--text-muted)]" />
            <h2 className="text-[14px] font-medium text-[var(--text-primary)]">Delete Document</h2>
          </div>
          <div className="space-y-3">
            <div>
              <label className="text-[11px] text-[var(--text-muted)] block mb-1">Dataset</label>
              <select
                value={deleteDatasetId}
                onChange={(e) => setDeleteDatasetId(e.target.value)}
                className="w-full text-xs bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-[var(--text-secondary)]"
              >
                <option value="">Select dataset...</option>
                {datasets?.map((ds) => (
                  <option key={ds.id} value={ds.id}>{ds.name} ({ds.dataCount ?? 0} items)</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[11px] text-[var(--text-muted)] block mb-1">Data ID (UUID)</label>
              <input
                type="text"
                value={deleteDataId}
                onChange={(e) => setDeleteDataId(e.target.value)}
                placeholder="e.g. 550e8400-e29b-41d4-a716-..."
                className="w-full text-xs bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-[var(--text-secondary)] placeholder:text-[var(--text-muted)]"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="text-[11px] text-[var(--text-muted)]">Mode:</label>
              <label className="flex items-center gap-1 text-xs text-[var(--text-secondary)]">
                <input type="radio" name="deleteMode" checked={deleteMode === "soft"} onChange={() => setDeleteMode("soft")} className="accent-[var(--text-primary)]" />
                Soft
              </label>
              <label className="flex items-center gap-1 text-xs text-[var(--text-secondary)]">
                <input type="radio" name="deleteMode" checked={deleteMode === "hard"} onChange={() => setDeleteMode("hard")} className="accent-[var(--text-primary)]" />
                Hard (cascade)
              </label>
            </div>
            <button
              onClick={handleDeleteDocument}
              disabled={deleteDoc.isPending || !deleteDataId || !deleteDatasetId}
              className={cn(
                "w-full py-2 text-xs font-medium rounded transition-all flex items-center justify-center gap-2",
                deleteDoc.isPending || !deleteDataId || !deleteDatasetId
                  ? "bg-[var(--bg-elevated)] text-[var(--text-muted)] cursor-not-allowed"
                  : "bg-[var(--bg-elevated)] text-[var(--error)] hover:bg-[var(--error)]/10 border border-transparent hover:border-[var(--error)]/30"
              )}
            >
              {deleteDoc.isPending && <Loader2 size={12} className="animate-spin" />}
              Delete Document
            </button>
          </div>
        </div>

        {/* Data Sync */}
        <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <RefreshCw size={16} className="text-[var(--text-muted)]" />
              <h2 className="text-[14px] font-medium text-[var(--text-primary)]">Data Sync</h2>
            </div>
            {syncStatus?.has_running_sync && (
              <span className="text-[10px] text-blue-400 flex items-center gap-1">
                <Loader2 size={10} className="animate-spin" />
                {syncStatus.running_sync_count} running
              </span>
            )}
          </div>
          <p className="text-[12px] text-[var(--text-muted)]">
            Re-process data that may have been added but not fully memorized.
          </p>
          <button
            onClick={() => handleSync()}
            disabled={syncData.isPending}
            className={cn(
              "w-full py-2 text-xs font-medium rounded transition-all flex items-center justify-center gap-2",
              "bg-[var(--text-primary)] text-[var(--bg-base)]",
              syncData.isPending ? "opacity-50 cursor-not-allowed" : "hover:opacity-90"
            )}
          >
            {syncData.isPending && <Loader2 size={12} className="animate-spin" />}
            Sync All Datasets
          </button>
          {syncStatus?.latest_running_sync && (
            <div className="p-2 bg-[var(--bg-elevated)] rounded text-[11px] text-[var(--text-muted)]">
              Latest sync: {syncStatus.latest_running_sync.dataset_names?.join(", ") || "All"} — {syncStatus.latest_running_sync.progress_percentage}%
            </div>
          )}
        </div>
      </div>

      {/* Dataset Status */}
      <div className="mt-8">
        <div className="flex items-center gap-2 mb-4">
          <Database size={16} className="text-[var(--text-muted)]" />
          <h2 className="text-[14px] font-medium text-[var(--text-primary)]">Dataset Status</h2>
        </div>
        {dsLoading ? (
          <div className="flex items-center gap-2 py-4"><Loader2 size={14} className="animate-spin text-[var(--text-muted)]" /> Loading...</div>
        ) : (
          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] divide-y divide-[var(--border-subtle)]">
            {datasets && datasets.length > 0 ? datasets.map((ds) => (
              <div key={ds.id} className="p-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {statusIcon(dsStatus?.[ds.id])}
                  <div>
                    <span className="text-xs text-[var(--text-primary)] font-medium">{ds.name}</span>
                    <span className="text-[10px] text-[var(--text-muted)] ml-2">{ds.dataCount ?? 0} items</span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] text-[var(--text-muted)]">
                    {dsStatus?.[ds.id] || "Unknown"}
                  </span>
                  <button
                    onClick={() => handleSync([ds.id])}
                    disabled={syncData.isPending}
                    className="text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] flex items-center gap-1"
                  >
                    <RefreshCw size={10} /> Sync
                  </button>
                </div>
              </div>
            )) : (
              <div className="p-4 text-xs text-[var(--text-muted)] text-center">No datasets found</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default DataManagementPage;
