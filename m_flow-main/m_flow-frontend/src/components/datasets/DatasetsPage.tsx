"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDatasetsWithCounts, useDeleteDataset } from "@/hooks/use-api";
import { Button } from "@/components/ui/button";
import { cn, formatTimeAgo } from "@/lib/utils";
import { toast } from "sonner";
import { Database, FileText, Trash2, Loader2, AlertCircle, FolderOpen, ArrowRight } from "lucide-react";

export function DatasetsPage() {
  const { data: datasets, isLoading, error, refetch } = useDatasetsWithCounts();
  const deleteDataset = useDeleteDataset();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete dataset "${name}"? This cannot be undone.`)) return;
    setDeletingId(id);
    try {
      await deleteDataset.mutateAsync(id);
      toast.success(`Dataset "${name}" deleted`);
    } catch {
      toast.error("Delete failed");
    } finally {
      setDeletingId(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <Loader2 size={20} className="text-[var(--text-muted)] animate-spin mb-3" />
        <p className="text-[13px] text-[var(--text-muted)]">Loading datasets...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <AlertCircle size={24} className="text-[var(--error)] mb-3" />
        <p className="text-[14px] text-[var(--text-primary)] mb-1">Failed to load</p>
        <p className="text-[13px] text-[var(--text-muted)] mb-4">Check if backend is running</p>
        <Button variant="outline" onClick={() => refetch()}>Retry</Button>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-lg font-medium text-[var(--text-primary)]">Datasets</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">Manage your imported data.</p>
        </div>
        <span className="text-[13px] text-[var(--text-muted)]">{datasets?.length || 0} total</span>
      </div>

      {/* List */}
      {datasets && datasets.length > 0 ? (
        <div className="space-y-2">
          <AnimatePresence mode="popLayout">
            {datasets.map((ds, index) => (
              <motion.div
                key={ds.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ delay: index * 0.03 }}
                className={cn(
                  "flex items-center gap-4 p-4 border border-[var(--border-subtle)] rounded-lg hover:border-[var(--border-default)] transition-colors group",
                  deletingId === ds.id && "opacity-50"
                )}
              >
                <Database size={18} strokeWidth={1.5} className="text-[var(--text-muted)] flex-shrink-0" />

                <div className="flex-1 min-w-0">
                  <p className="text-[14px] font-medium text-[var(--text-primary)] truncate">{ds.name}</p>
                  <div className="flex items-center gap-3 mt-1 text-[11px] text-[var(--text-muted)]">
                    <span className="flex items-center gap-1">
                      <FileText size={10} /> {ds.dataCount || 0} items
                    </span>
                    {ds.createdAt && <span>{formatTimeAgo(ds.createdAt)}</span>}
                  </div>
                </div>

                {ds.status && (
                  <span className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-medium",
                    ds.status === "ready" ? "bg-[var(--success)]/20 text-[var(--success)]" : ds.status === "processing" ? "bg-[var(--bg-elevated)] text-[var(--text-muted)]" : "bg-[var(--error)]/20 text-[var(--error)]"
                  )}>
                    {ds.status}
                  </span>
                )}

                <button
                  onClick={() => handleDelete(ds.id, ds.name)}
                  disabled={deletingId === ds.id}
                  className="p-2 text-[var(--text-muted)] hover:text-[var(--error)] opacity-0 group-hover:opacity-100 transition-all"
                >
                  {deletingId === ds.id ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                </button>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-20">
          <FolderOpen size={32} className="text-[var(--text-muted)] mb-3" />
          <p className="text-[14px] text-[var(--text-primary)] mb-1">No datasets</p>
          <p className="text-[13px] text-[var(--text-muted)]">Upload files to get started</p>
        </div>
      )}
    </div>
  );
}
