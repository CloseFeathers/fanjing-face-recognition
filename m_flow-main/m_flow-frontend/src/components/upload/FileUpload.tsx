"use client";

import React, { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useIngestText, useIngestFiles } from "@/hooks/use-api";
import { useMemorizeWebSocket } from "@/hooks/use-memorize-websocket";
import { useIngestionConfigStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ProgressTracker, getActionableErrorMessage, FileTypeIcon } from "@/components/common";
import { cn, truncate } from "@/lib/utils";
import { toast } from "sonner";
import { Upload, X, Check, AlertCircle, Plus, Settings2, ChevronDown, ChevronUp, Info } from "lucide-react";
import { ConfigWizard, type WizardResult } from "@/components/memorize/ConfigWizard";

// ============================================================================
// Types
// ============================================================================

interface FileWithStatus {
  file: File;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

// ============================================================================
// Imports
// ============================================================================

import { 
  ACCEPTED_FILE_TYPES, 
  validateFiles, 
  validateDatasetName,
  formatFileSize 
} from "@/lib/file-validation";

// ============================================================================
// Main Component
// ============================================================================

export function FileUpload() {
  // File state
  const [files, setFiles] = useState<FileWithStatus[]>([]);
  const [textInput, setTextInput] = useState("");
  const [datasetName, setDatasetName] = useState("");
  const [datasetNameError, setDatasetNameError] = useState<string | undefined>();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Ingest options
  const [skipMemorize, setSkipMemorize] = useState(false);
  const [runInBackground, setRunInBackground] = useState(true);
  
  // Feature toggles
  const [enableEpisodeRouting, setEnableEpisodeRouting] = useState<boolean | undefined>(undefined);
  const [enableContentRouting, setEnableContentRouting] = useState<boolean | undefined>(undefined);
  const [enableProcedural, setEnableProcedural] = useState<boolean | undefined>(undefined);
  
  // Fixed to "error" mode - prevent concurrent memorize operations on the same dataset
  const conflictMode = "error" as const;

  // Progress tracking state
  const [pipelineRunId, setPipelineRunId] = useState<string | null>(null);
  const [processedDatasetName, setProcessedDatasetName] = useState<string>("");

  // API hooks
  const ingestText = useIngestText();
  const ingestFiles = useIngestFiles();
  const ingestionConfig = useIngestionConfigStore();

  // WebSocket progress tracking
  const {
    progress,
    connectionState,
    error: wsError,
    isCompleted,
    isError,
    isProcessing,
    disconnect,
    reconnect,
  } = useMemorizeWebSocket(pipelineRunId);

  // Handle completion
  React.useEffect(() => {
    let timeoutId: NodeJS.Timeout | null = null;
    
    if (isCompleted) {
      toast.success("Knowledge graph built successfully!");
      // Clear after a delay
      timeoutId = setTimeout(() => {
        setPipelineRunId(null);
        setProcessedDatasetName("");
      }, 3000);
    }
    if (isError) {
      toast.error("Processing failed. Please try again.");
    }
    
    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [isCompleted, isError]);

  // File handlers with enhanced validation
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const { valid, rejected } = validateFiles(selectedFiles);
    
    rejected.forEach(({ file, reason }) => {
      toast.error(`${file.name}: ${reason}`);
    });
    
    if (valid.length > 0) {
      setFiles((prev) => [...prev, ...valid.map((file) => ({ file, status: "pending" as const }))]);
    }
    
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    const { valid, rejected } = validateFiles(droppedFiles);
    
    rejected.forEach(({ file, reason }) => {
      toast.error(`${file.name}: ${reason}`);
    });
    
    if (valid.length > 0) {
      setFiles((prev) => [...prev, ...valid.map((file) => ({ file, status: "pending" as const }))]);
    }
  }, []);

  const removeFile = (index: number) => setFiles((prev) => prev.filter((_, i) => i !== index));

  // Build ingest options from current state
  const buildIngestOptions = () => ({
    datasetName: datasetName || "main_dataset",
    skipMemorize,
    runInBackground,
    enableEpisodeRouting,
    enableContentRouting,
    enableProcedural,
    conflictMode,
  });

  // Upload handler
  const handleUploadAll = async () => {
    const pendingFiles = files.filter((f) => f.status === "pending");
    if (pendingFiles.length === 0 && !textInput.trim()) {
      toast.error("Add files or text first");
      return;
    }

    // Validate dataset name before submission
    const validation = validateDatasetName(datasetName);
    if (!validation.valid) {
      setDatasetNameError(validation.error);
      toast.error(validation.error || "Invalid dataset name");
      return;
    }

    const targetDataset = datasetName || "main_dataset";
    const options = buildIngestOptions();

    // Ingest text content
    if (textInput.trim()) {
      try {
        const response = await ingestText.mutateAsync({
          content: textInput,
          dataset_name: targetDataset,
          skip_memorize: skipMemorize,
          run_in_background: runInBackground,
          enable_episode_routing: enableEpisodeRouting,
          enable_content_routing: enableContentRouting,
          enable_procedural: enableProcedural,
          conflict_mode: conflictMode,
        });
        
        setTextInput("");
        
        // Track progress if running in background
        if (runInBackground && response.memorize_run_id) {
          setPipelineRunId(response.memorize_run_id);
          setProcessedDatasetName(targetDataset);
          toast.info("Text ingested - tracking progress...");
        } else if (response.status === "completed") {
          toast.success("Text ingested and knowledge graph built");
        } else {
          toast.success("Text added to dataset");
        }
      } catch (error) {
        const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Failed to ingest text");
        toast.error(actionableMessage);
        console.error("Text ingest error:", error);
      }
    }

    // Ingest files
    if (pendingFiles.length > 0) {
      // Mark all as uploading
      setFiles((prev) => prev.map((f) => 
        f.status === "pending" ? { ...f, status: "uploading" as const } : f
      ));

      try {
        const response = await ingestFiles.mutateAsync({
          files: pendingFiles.map(f => f.file),
          options,
        });

        // Mark all as success
        setFiles((prev) => prev.map((f) => 
          f.status === "uploading" ? { ...f, status: "success" as const } : f
        ));

        // Track progress if running in background
        if (runInBackground && response.memorize_run_id) {
          setPipelineRunId(response.memorize_run_id);
          setProcessedDatasetName(targetDataset);
          toast.info(`${pendingFiles.length} file(s) ingested - tracking progress...`);
        } else if (response.status === "completed") {
          toast.success(`${pendingFiles.length} file(s) ingested and knowledge graph built`);
        } else {
          toast.success(`${pendingFiles.length} file(s) added to dataset`);
        }
      } catch (error) {
        // Mark all uploading as error
        setFiles((prev) => prev.map((f) => 
          f.status === "uploading" 
            ? { ...f, status: "error" as const, error: error instanceof Error ? error.message : "Upload failed" } 
            : f
        ));
        const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Failed to ingest files");
        toast.error(actionableMessage);
        console.error("File ingest error:", error);
      }
    }
  };

  // Cancel progress tracking
  const handleCancelTracking = () => {
    disconnect();
    setPipelineRunId(null);
    setProcessedDatasetName("");
  };

  // Retry progress tracking
  const handleRetry = () => {
    reconnect();
  };

  // Stats
  const stats = {
    total: files.length,
    pending: files.filter((f) => f.status === "pending").length,
    uploading: files.filter((f) => f.status === "uploading").length,
    success: files.filter((f) => f.status === "success").length,
    error: files.filter((f) => f.status === "error").length,
  };
  const isUploading = stats.uploading > 0 || ingestText.isPending || ingestFiles.isPending;

  return (
    <div className="max-w-2xl mx-auto py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-[28px] font-semibold text-[var(--text-primary)] tracking-tight mb-2">Add Memory</h1>
        <p className="text-[var(--text-secondary)]">Upload files or enter text to add to your knowledge base.</p>
      </div>

      {/* Progress Tracker */}
      {pipelineRunId && (
        <ProgressTracker
          pipelineRunId={pipelineRunId}
          status={progress?.status || null}
          connectionState={connectionState}
          error={wsError}
          datasetName={processedDatasetName}
          onRetry={handleRetry}
          onCancel={!isProcessing ? undefined : handleCancelTracking}
        />
      )}

      {/* Drop Zone with Enhanced Visual Feedback */}
      <motion.div
        animate={{
          scale: isDragging ? 1.02 : 1,
          borderColor: isDragging ? "var(--text-primary)" : "var(--border-subtle)",
        }}
        transition={{ duration: 0.15 }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && inputRef.current?.click()}
        className={cn(
          "relative flex flex-col items-center justify-center p-10 border-2 border-dashed rounded-lg cursor-pointer transition-colors overflow-hidden",
          isDragging ? "border-[var(--text-primary)] bg-[var(--text-primary)]/5" : "border-[var(--border-subtle)] hover:border-[var(--border-default)]",
          isUploading && "opacity-50 cursor-not-allowed"
        )}
        role="button"
        tabIndex={0}
        aria-label="Drop files here or click to browse"
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !isUploading) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        {/* Animated background when dragging */}
        <AnimatePresence>
          {isDragging && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-blue-500/5"
              style={{
                backgroundSize: "200% 100%",
                animation: "shimmer 2s ease-in-out infinite",
              }}
            />
          )}
        </AnimatePresence>
        
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED_FILE_TYPES.join(",")}
          onChange={handleFileSelect}
          className="hidden"
          disabled={isUploading}
          aria-hidden="true"
        />
        
        <motion.div
          animate={{ y: isDragging ? -4 : 0 }}
          transition={{ duration: 0.15 }}
          className="relative z-10"
        >
          <Upload 
            size={24} 
            strokeWidth={1.5} 
            className={cn(
              "mb-3 mx-auto transition-colors",
              isDragging ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]"
            )} 
          />
        </motion.div>
        <p className="relative z-10 text-[14px] text-[var(--text-primary)] mb-1">
          {isDragging ? "Release to upload" : "Drop files here"}
        </p>
        <p className="relative z-10 text-[12px] text-[var(--text-muted)]">
          or click to browse · max 50MB
        </p>
      </motion.div>

      {/* Text Input */}
      <div className="space-y-2">
        <label className="text-[12px] text-[var(--text-muted)]">Text input</label>
        <textarea
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Enter text directly..."
          rows={4}
          disabled={isUploading}
          className={cn(
            "w-full px-3 py-2 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] resize-none",
            isUploading && "opacity-50 cursor-not-allowed"
          )}
        />
      </div>

      {/* Dataset Name */}
      <div className="space-y-2">
        <label className="text-[12px] text-[var(--text-muted)]">Dataset (optional)</label>
        <input
          value={datasetName}
          onChange={(e) => {
            const value = e.target.value;
            setDatasetName(value);
            const validation = validateDatasetName(value);
            setDatasetNameError(validation.error);
          }}
          placeholder="Enter dataset name"
          disabled={isUploading}
          className={cn(
            "w-full h-10 px-3 bg-[var(--bg-surface)] border rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none",
            datasetNameError 
              ? "border-red-500/50 focus:border-red-500" 
              : "border-[var(--border-subtle)] focus:border-[var(--border-default)]",
            isUploading && "opacity-50 cursor-not-allowed"
          )}
        />
        {datasetNameError && (
          <p className="text-[11px] text-red-400">{datasetNameError}</p>
        )}
      </div>

      {/* Advanced */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        disabled={isUploading}
        className={cn(
          "flex items-center gap-2 text-[13px] text-[var(--text-muted)] hover:text-[var(--text-secondary)]",
          isUploading && "opacity-50 cursor-not-allowed"
        )}
      >
        <Settings2 size={14} strokeWidth={1.5} />
        Advanced
        {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      <AnimatePresence>
        {showAdvanced && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="pt-4 border-t border-[var(--border-subtle)] space-y-4">
              {/* Skip Memorize */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label className="text-[13px]">Skip Knowledge Graph</Label>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Store data only without building knowledge graph. Useful for batch uploads.
                    </div>
                  </div>
                </div>
                <Switch checked={skipMemorize} onCheckedChange={setSkipMemorize} disabled={isUploading} />
              </div>

              {/* Background Processing */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label className="text-[13px]">Background Processing</Label>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Process asynchronously with real-time progress tracking. Recommended for large files.
                    </div>
                  </div>
                </div>
                <Switch 
                  checked={runInBackground} 
                  onCheckedChange={setRunInBackground} 
                  disabled={isUploading || skipMemorize} 
                />
              </div>

              {/* Scenario Presets + Feature Toggles */}
              {!skipMemorize && (
                <div className="space-y-3 pt-3 border-t border-[var(--border-subtle)]">
                  <ConfigWizard
                    onApply={(result: WizardResult) => {
                      if (result.enableEpisodeRouting !== undefined) setEnableEpisodeRouting(result.enableEpisodeRouting);
                      if (result.enableContentRouting !== undefined) setEnableContentRouting(result.enableContentRouting);
                    }}
                    disabled={isUploading}
                  />
                  <p className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider">Processing Options</p>
                  
                  {/* Episode Routing */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] text-[var(--text-secondary)]">Episode Routing</span>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          When new content may closely relate to previously ingested content, enable this to merge them into coherent episodes. Costs extra LLM tokens and time. Disable for faster independent-document ingestion.
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-[var(--text-muted)]">
                        {enableEpisodeRouting === undefined ? "Auto" : enableEpisodeRouting ? "On" : "Off"}
                      </span>
                      <Switch
                        checked={enableEpisodeRouting === true}
                        onCheckedChange={(v) => setEnableEpisodeRouting(v ? true : (enableEpisodeRouting === true ? false : undefined))}
                        disabled={isUploading}
                      />
                    </div>
                  </div>

                  {/* Content Routing */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] text-[var(--text-secondary)]">Content Routing</span>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          When a single text block may contain multiple topics, enable this to classify each sentence for cleaner memory structure. Costs extra LLM tokens per chunk.
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-[var(--text-muted)]">
                        {enableContentRouting === undefined ? "Auto" : enableContentRouting ? "On" : "Off"}
                      </span>
                      <Switch
                        checked={enableContentRouting === true}
                        onCheckedChange={(v) => setEnableContentRouting(v ? true : (enableContentRouting === true ? false : undefined))}
                        disabled={isUploading}
                      />
                    </div>
                  </div>

                  {/* Procedural Memory */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] text-[var(--text-secondary)]">Procedural Memory</span>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          Extract reusable procedures and preferences to complement episodic facts with abstract knowledge. Experimental. Costs extra LLM tokens.
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-[var(--text-muted)]">
                        {enableProcedural === undefined ? "Auto" : enableProcedural ? "On" : "Off"}
                      </span>
                      <Switch
                        checked={enableProcedural === true}
                        onCheckedChange={(v) => setEnableProcedural(v ? true : (enableProcedural === true ? false : undefined))}
                        disabled={isUploading}
                      />
                    </div>
                  </div>

                </div>
              )}

              {/* Config Summary */}
              <div className="p-3 bg-[var(--bg-elevated)] rounded-md">
                <p className="text-[11px] text-[var(--text-muted)] mb-2">Current Settings</p>
                <div className="flex flex-wrap gap-2 text-[11px]">
                  <span className={cn(
                    "px-2 py-0.5 rounded",
                    skipMemorize ? "bg-amber-500/20 text-amber-400" : "bg-[var(--success)]/20 text-[var(--success)]"
                  )}>
                    {skipMemorize ? "Add Only" : "Full Ingest"}
                  </span>
                  {!skipMemorize && (
                    <>
                      <span className={cn(
                        "px-2 py-0.5 rounded",
                        runInBackground ? "bg-blue-500/20 text-blue-400" : "bg-[var(--bg-surface)] text-[var(--text-muted)]"
                      )}>
                        {runInBackground ? "Async" : "Sync"}
                      </span>
                      {enableEpisodeRouting !== undefined && (
                        <span className={cn(
                          "px-2 py-0.5 rounded",
                          enableEpisodeRouting ? "bg-[var(--success)]/20 text-[var(--success)]" : "bg-[var(--bg-surface)] text-[var(--text-muted)]"
                        )}>
                          Episode: {enableEpisodeRouting ? "on" : "off"}
                        </span>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-3">
          {/* Overall Progress Header */}
          <div className="flex items-center justify-between">
            <p className="text-[13px] text-[var(--text-secondary)]">{stats.total} files</p>
            <div className="flex items-center gap-3 text-[12px]">
              {stats.uploading > 0 && <span className="text-blue-400">⟳ {stats.uploading} uploading</span>}
              {stats.success > 0 && <span className="text-[var(--success)]">✓ {stats.success}</span>}
              {stats.error > 0 && <span className="text-[var(--error)]">✗ {stats.error}</span>}
            </div>
          </div>
          
          {/* Overall Progress Bar */}
          {(stats.uploading > 0 || stats.success > 0 || stats.error > 0) && (
            <div className="space-y-1">
              <div className="h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden flex">
                {stats.success > 0 && (
                  <motion.div
                    className="h-full bg-[var(--success)]"
                    initial={{ width: 0 }}
                    animate={{ width: `${(stats.success / stats.total) * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                )}
                {stats.uploading > 0 && (
                  <motion.div
                    className="h-full bg-blue-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${(stats.uploading / stats.total) * 100}%` }}
                    style={{ 
                      background: "linear-gradient(90deg, #3b82f6, #60a5fa, #3b82f6)",
                      backgroundSize: "200% 100%",
                    }}
                  />
                )}
                {stats.error > 0 && (
                  <motion.div
                    className="h-full bg-[var(--error)]"
                    initial={{ width: 0 }}
                    animate={{ width: `${(stats.error / stats.total) * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                )}
              </div>
              <p className="text-[10px] text-[var(--text-muted)] text-right">
                {stats.success + stats.error}/{stats.total} completed
              </p>
            </div>
          )}
          
          <div className="space-y-1 max-h-48 overflow-y-auto">
            <AnimatePresence>
              {files.map((fileItem, index) => (
                <motion.div
                  key={`${fileItem.file.name}-${index}`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={cn(
                    "flex items-center gap-3 p-3 rounded-md border transition-colors",
                    fileItem.status === "success"
                      ? "border-[var(--success)]/30 bg-[var(--success)]/5"
                      : fileItem.status === "error"
                      ? "border-[var(--error)]/30 bg-[var(--error)]/5"
                      : "border-[var(--border-subtle)]"
                  )}
                >
                  <div className="flex-shrink-0 relative">
                    <FileTypeIcon filename={fileItem.file.name} size={16} />
                    {fileItem.status === "uploading" && (
                      <motion.div
                        className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full"
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ repeat: Infinity, duration: 1 }}
                      />
                    )}
                    {fileItem.status === "success" && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute -top-1 -right-1"
                      >
                        <Check size={10} className="text-[var(--success)] bg-[var(--bg-base)] rounded-full" />
                      </motion.div>
                    )}
                    {fileItem.status === "error" && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute -top-1 -right-1"
                      >
                        <AlertCircle size={10} className="text-[var(--error)] bg-[var(--bg-base)] rounded-full" />
                      </motion.div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] text-[var(--text-primary)] truncate">{truncate(fileItem.file.name, 40)}</p>
                    <p className="text-[11px] text-[var(--text-muted)]">{formatFileSize(fileItem.file.size)}</p>
                  </div>
                  {fileItem.status !== "uploading" && (
                    <button onClick={() => removeFile(index)} className="p-1 text-[var(--text-muted)] hover:text-[var(--text-secondary)]" disabled={isUploading}>
                      <X size={14} />
                    </button>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-end gap-3 pt-4 border-t border-[var(--border-subtle)]">
        {files.length > 0 && (
          <Button variant="ghost" onClick={() => setFiles([])} disabled={isUploading}>
            Clear
          </Button>
        )}
        <Button onClick={handleUploadAll} loading={isUploading} disabled={stats.pending === 0 && !textInput.trim()}>
          <Plus size={14} className="mr-2" />
          {isUploading ? `Uploading (${stats.uploading}/${stats.total})` : `Upload (${stats.pending + (textInput.trim() ? 1 : 0)})`}
        </Button>
      </div>
    </div>
  );
}
