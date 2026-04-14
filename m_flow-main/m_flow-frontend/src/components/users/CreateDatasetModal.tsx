"use client";

import React, { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Database, X, Loader2 } from "lucide-react";
import { useCreateDataset } from "@/hooks/use-api";
import { toast } from "sonner";

interface CreateDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (datasetName: string) => void;
}

export function CreateDatasetModal({ isOpen, onClose, onSuccess }: CreateDatasetModalProps) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const createDataset = useCreateDataset();
  
  const isProcessing = createDataset.isPending;

  const handleClose = useCallback(() => {
    if (isProcessing) return;
    setName("");
    setError(null);
    onClose();
  }, [isProcessing, onClose]);

  const handleSubmit = useCallback(async () => {
    const trimmedName = name.trim();
    
    if (!trimmedName) {
      setError("Dataset name is required");
      return;
    }
    
    if (/\s/.test(trimmedName)) {
      setError("Dataset name cannot contain spaces. Use underscores or hyphens instead.");
      return;
    }

    if (trimmedName.length > 255) {
      setError("Dataset name must be 255 characters or less");
      return;
    }

    try {
      await createDataset.mutateAsync(trimmedName);
      toast.success(`Dataset "${trimmedName}" created`);
      handleClose();
      onSuccess?.(trimmedName);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to create dataset";
      setError(message);
      toast.error(message);
    }
  }, [name, createDataset, onSuccess, handleClose]);

  // ESC key to close
  useEffect(() => {
    if (!isOpen) return;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        handleClose();
      }
    };
    
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, handleClose]);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setName("");
      setError(null);
    }
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
        >
          <div className="absolute inset-0 bg-black/60" onClick={handleClose} />
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="relative bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg shadow-xl max-w-md w-full mx-4 overflow-hidden"
          >
            <div className="flex items-center justify-between p-4 border-b border-[var(--border-subtle)]">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-blue-500/10">
                  <Database size={18} className="text-blue-500" />
                </div>
                <h3 className="text-sm font-medium text-[var(--text-primary)]">Create Dataset</h3>
              </div>
              <button
                onClick={handleClose}
                disabled={isProcessing}
                className="p-1 rounded hover:bg-[var(--bg-elevated)] text-[var(--text-muted)] transition-colors disabled:opacity-50"
              >
                <X size={16} />
              </button>
            </div>
            
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-2">
                  Dataset Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => {
                    setName(e.target.value);
                    setError(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !isProcessing) {
                      handleSubmit();
                    }
                  }}
                  placeholder="Enter dataset name..."
                  disabled={isProcessing}
                  className="w-full px-3 py-2 text-sm bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded-md text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                  autoFocus
                />
                {error && (
                  <p className="mt-2 text-xs text-red-500">{error}</p>
                )}
              </div>
              <p className="text-xs text-[var(--text-muted)]">
                A dataset is a container for organizing your data. You can add content to a dataset later.
              </p>
            </div>
            
            <div className="flex justify-end gap-2 p-4 border-t border-[var(--border-subtle)] bg-[var(--bg-elevated)]">
              <button
                onClick={handleClose}
                disabled={isProcessing}
                className="px-4 py-2 text-sm text-[var(--text-secondary)] bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded hover:bg-[var(--bg-hover)] transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={isProcessing || !name.trim()}
                className="px-4 py-2 text-sm text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {isProcessing && <Loader2 size={14} className="animate-spin" />}
                {isProcessing ? "Creating..." : "Create"}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
