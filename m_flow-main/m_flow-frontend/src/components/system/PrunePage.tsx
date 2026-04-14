"use client";

import React, { useState } from "react";
import { toast } from "sonner";
import { Trash2, AlertTriangle, Database, HardDrive, RefreshCw, Loader2, ShieldAlert, FileX, Server } from "lucide-react";
import { cn } from "@/lib/utils";
import { useActivePipelines } from "@/hooks/use-active-pipelines";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useAuthStore } from "@/lib/store/auth";
import { usePruneAll, usePruneData, usePruneSystem } from "@/hooks/use-api";
import { Switch } from "@/components/ui/switch";

export function PrunePage() {
  const [isPruning, setIsPruning] = useState(false);
  const { data: activePipelines } = useActivePipelines();
  const confirm = useConfirm();
  const { user } = useAuthStore();

  const pruneAllMutation = usePruneAll();
  const pruneDataMutation = usePruneData();
  const pruneSystemMutation = usePruneSystem();

  // Granular system prune toggles
  const [pruneGraph, setPruneGraph] = useState(true);
  const [pruneVector, setPruneVector] = useState(true);
  const [pruneMetadata, setPruneMetadata] = useState(true);
  const [pruneCache, setPruneCache] = useState(true);

  const hasActivePipelines = activePipelines && activePipelines.length > 0;
  const isSuperuser = user?.is_superuser ?? false;
  const disabled = isPruning || !!hasActivePipelines || !isSuperuser;

  const handlePruneAll = async () => {
    if (hasActivePipelines) {
      toast.error("Cannot prune while pipelines are active.");
      return;
    }
    const confirmed = await confirm({
      title: "Delete All Data",
      message: "This will permanently delete ALL data including file storage, graph database, vector database, relational database, and cache. This action cannot be undone!",
      confirmText: "Delete Everything",
      cancelText: "Cancel",
      variant: "danger",
    });
    if (!confirmed) return;
    setIsPruning(true);
    try {
      const result = await pruneAllMutation.mutateAsync();
      if (result.warnings && result.warnings.length > 0) {
        toast.warning(`Prune completed with warnings: ${result.warnings.join(", ")}`);
      } else {
        toast.success("All data has been permanently deleted");
      }
      setTimeout(() => window.location.reload(), 1500);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Prune operation failed");
    } finally {
      setIsPruning(false);
    }
  };

  const handlePruneData = async () => {
    const confirmed = await confirm({
      title: "Delete File Storage",
      message: "This will permanently delete all uploaded files and raw data. Database records will remain. This action cannot be undone!",
      confirmText: "Delete Files",
      cancelText: "Cancel",
      variant: "danger",
    });
    if (!confirmed) return;
    setIsPruning(true);
    try {
      const result = await pruneDataMutation.mutateAsync();
      toast.success(result.message || "File storage cleared");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Prune data failed");
    } finally {
      setIsPruning(false);
    }
  };

  const handlePruneSystem = async () => {
    const targets = [];
    if (pruneGraph) targets.push("Graph DB");
    if (pruneVector) targets.push("Vector DB");
    if (pruneMetadata) targets.push("Metadata DB");
    if (pruneCache) targets.push("Cache");
    if (targets.length === 0) {
      toast.error("Select at least one target to clear");
      return;
    }
    const confirmed = await confirm({
      title: "Clear System Databases",
      message: `This will permanently clear: ${targets.join(", ")}. This action cannot be undone!`,
      confirmText: "Clear Selected",
      cancelText: "Cancel",
      variant: "danger",
    });
    if (!confirmed) return;
    setIsPruning(true);
    try {
      const result = await pruneSystemMutation.mutateAsync({
        graph: pruneGraph,
        vector: pruneVector,
        metadata: pruneMetadata,
        cache: pruneCache,
      });
      toast.success(result.message || "System databases cleared");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "System prune failed");
    } finally {
      setIsPruning(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Maintenance</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Manage system data and perform maintenance operations.
        </p>
      </div>

      {/* Active Pipelines Warning */}
      {hasActivePipelines && (
        <div className="mb-6 flex items-center gap-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded text-xs">
          <RefreshCw size={14} className="text-blue-400 shrink-0 animate-spin" />
          <span className="text-[var(--text-secondary)]">
            {activePipelines.length} pipeline(s) running. Wait for completion or dismiss them before pruning.
          </span>
        </div>
      )}

      {/* Superuser Required Warning */}
      {!isSuperuser && (
        <div className="mb-6 flex items-center gap-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded text-xs">
          <ShieldAlert size={14} className="text-yellow-400 shrink-0" />
          <span className="text-[var(--text-secondary)]">
            Administrator privileges required to perform maintenance operations.
          </span>
        </div>
      )}

      <div className="space-y-6">
        {/* Granular Operations */}
        <div className="space-y-4">
          <h2 className="text-[13px] font-medium text-[var(--text-primary)]">Selective Cleanup</h2>

          {/* Prune Data (Files Only) */}
          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-[var(--bg-elevated)] flex items-center justify-center text-[var(--text-muted)]">
                  <FileX size={18} />
                </div>
                <div>
                  <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Clear File Storage</h3>
                  <p className="text-[12px] text-[var(--text-muted)] mt-0.5">
                    Delete uploaded files and raw data only. Database records are preserved.
                  </p>
                </div>
              </div>
              <button
                onClick={handlePruneData}
                disabled={disabled}
                className={cn(
                  "px-4 py-2 rounded-lg text-[13px] font-medium transition-all shrink-0 flex items-center gap-2",
                  disabled
                    ? "bg-[var(--bg-elevated)] text-[var(--text-muted)] cursor-not-allowed"
                    : "bg-[var(--bg-elevated)] text-amber-500 hover:bg-amber-500/10 border border-transparent hover:border-amber-500/30"
                )}
              >
                {isPruning && pruneDataMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : null}
                Delete Files
              </button>
            </div>
          </div>

          {/* Prune System (Selective) */}
          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-[var(--bg-elevated)] flex items-center justify-center text-[var(--text-muted)]">
                  <Server size={18} />
                </div>
                <div>
                  <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Clear System Databases</h3>
                  <p className="text-[12px] text-[var(--text-muted)] mt-0.5">
                    Selectively clear system databases. Choose which components to reset.
                  </p>
                </div>
              </div>
              <button
                onClick={handlePruneSystem}
                disabled={disabled}
                className={cn(
                  "px-4 py-2 rounded-lg text-[13px] font-medium transition-all shrink-0 flex items-center gap-2",
                  disabled
                    ? "bg-[var(--bg-elevated)] text-[var(--text-muted)] cursor-not-allowed"
                    : "bg-[var(--bg-elevated)] text-amber-500 hover:bg-amber-500/10 border border-transparent hover:border-amber-500/30"
                )}
              >
                {isPruning && pruneSystemMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : null}
                Clear Selected
              </button>
            </div>
            <div className="mt-4 pt-4 border-t border-[var(--border-subtle)] grid grid-cols-2 gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
                  <Database size={12} /> Graph DB
                </div>
                <Switch checked={pruneGraph} onCheckedChange={setPruneGraph} disabled={disabled} />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
                  <Database size={12} /> Vector DB
                </div>
                <Switch checked={pruneVector} onCheckedChange={setPruneVector} disabled={disabled} />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
                  <Database size={12} /> Metadata
                </div>
                <Switch checked={pruneMetadata} onCheckedChange={setPruneMetadata} disabled={disabled} />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[12px] text-[var(--text-secondary)]">
                  <HardDrive size={12} /> Cache
                </div>
                <Switch checked={pruneCache} onCheckedChange={setPruneCache} disabled={disabled} />
              </div>
            </div>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <AlertTriangle size={14} className="text-[var(--warning)]" />
            <h2 className="text-[13px] font-medium text-[var(--text-primary)]">Danger Zone</h2>
          </div>

          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className="w-9 h-9 rounded-lg bg-[var(--bg-elevated)] flex items-center justify-center text-[var(--text-muted)]">
                  <Trash2 size={18} />
                </div>
                <div>
                  <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Clear All Data</h3>
                  <p className="text-[12px] text-[var(--text-muted)] mt-0.5">
                    Permanently remove all stored data and reset the entire system
                  </p>
                </div>
              </div>
              <button
                onClick={handlePruneAll}
                disabled={disabled}
                className={cn(
                  "px-4 py-2 rounded-lg text-[13px] font-medium transition-all shrink-0 flex items-center gap-2",
                  disabled
                    ? "bg-[var(--bg-elevated)] text-[var(--text-muted)] cursor-not-allowed"
                    : "bg-[var(--bg-elevated)] text-[var(--error)] hover:bg-[var(--error)]/10 border border-transparent hover:border-[var(--error)]/30"
                )}
              >
                {isPruning && pruneAllMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : null}
                Delete All
              </button>
            </div>
            <div className="mt-4 pt-4 border-t border-[var(--border-subtle)]">
              <p className="text-[11px] text-[var(--text-muted)] mb-2">This will delete:</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div className="flex items-center gap-2 text-[11px] text-[var(--text-muted)]"><HardDrive size={12} /> File Storage</div>
                <div className="flex items-center gap-2 text-[11px] text-[var(--text-muted)]"><Database size={12} /> Graph DB</div>
                <div className="flex items-center gap-2 text-[11px] text-[var(--text-muted)]"><Database size={12} /> Vector DB</div>
                <div className="flex items-center gap-2 text-[11px] text-[var(--text-muted)]"><Database size={12} /> Relational DB</div>
              </div>
            </div>
          </div>
        </div>

        <p className="text-[11px] text-[var(--text-muted)]">
          Requires <code className="px-1.5 py-0.5 bg-[var(--bg-elevated)] rounded text-[10px]">MFLOW_ENABLE_PRUNE_API=true</code> in environment.
        </p>
      </div>
    </div>
  );
}
