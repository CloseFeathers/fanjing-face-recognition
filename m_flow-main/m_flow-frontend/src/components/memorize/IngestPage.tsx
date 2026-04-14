"use client";

import React, { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useUIStore, useIngestionConfigStore, useChunkConfigStore } from "@/lib/store";
import { useIngestText, useIngestFiles } from "@/hooks/use-api";
import { useActivePipelines } from "@/hooks/use-active-pipelines";
import { useMemorizeWebSocket } from "@/hooks/use-memorize-websocket";
import { ProgressTracker, IngestionResult, getActionableErrorMessage, Term, FileTypeIcon } from "@/components/common";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { toast } from "sonner";
import { Upload, X, Check, AlertCircle, Loader2, ChevronDown, ChevronUp, Info, Settings2, RotateCcw, AlertTriangle } from "lucide-react";
import type { ChunkerType } from "@/types";
import { ConfigWizard } from "./ConfigWizard";

// ============================================================================
// Types
// ============================================================================

interface FileWithStatus {
  file: File;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

// ============================================================================
// Imports & Constants
// ============================================================================

import { 
  ACCEPTED_FILE_TYPES, 
  MAX_FILE_SIZE, 
  validateFiles 
} from "@/lib/file-validation";

const API_EXAMPLE = `import m_flow

# Add data to dataset
await m_flow.add(data="document.pdf", dataset_name="ds")

# Build knowledge graph
await m_flow.memorize(
    datasets=["ds"],
    run_in_background=True,
)`;

// ============================================================================
// Parameters Info
// ============================================================================

interface ParamInfo {
  name: string;
  type: string;
  default?: string;
  required?: boolean;
  description: string;
}

const INGEST_PARAMS: ParamInfo[] = [
  { name: "data", type: "str | List[str] | BinaryIO", required: true, description: "Text content, file path, URL, or binary stream" },
  { name: "dataset_name", type: "str", default: '"main_dataset"', description: "Target dataset name" },
  { name: "skip_memorize", type: "bool", default: "False", description: "Skip knowledge graph construction if True" },
  { name: "chunk_size", type: "int", default: "auto", description: "Chunk size in tokens" },
  { name: "incremental_loading", type: "bool", default: "True", description: "Skip already processed files" },
  { name: "run_in_background", type: "bool", default: "False", description: "Run asynchronously for large datasets" },
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
        <code className="text-xs text-[var(--text-secondary)] font-mono whitespace-pre">{code}</code>
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

function FileItem({ 
  file, 
  onRemove 
}: { 
  file: FileWithStatus; 
  onRemove: () => void;
}) {
  return (
    <div className={cn(
      "flex items-center gap-3 p-2 bg-[var(--bg-surface)] border rounded transition-colors",
      file.status === "success" 
        ? "border-[var(--success)]/30 bg-[var(--success)]/5" 
        : file.status === "error"
          ? "border-[var(--error)]/30 bg-[var(--error)]/5"
          : "border-[var(--border-subtle)]"
    )}>
      <div className="relative flex-shrink-0">
        <FileTypeIcon filename={file.file.name} size={14} />
        {file.status === "uploading" && (
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-[var(--text-primary)] truncate">{file.file.name}</p>
        <p className="text-[10px] text-[var(--text-muted)]">
          {(file.file.size / 1024).toFixed(1)} KB
        </p>
      </div>
      {file.status === "pending" && (
        <button onClick={onRemove} className="text-[var(--text-muted)] hover:text-[var(--error)] transition-colors">
          <X size={14} />
        </button>
      )}
      {file.status === "uploading" && (
        <Loader2 size={14} className="text-blue-400 animate-spin" />
      )}
      {file.status === "success" && (
        <Check size={14} className="text-[var(--success)]" />
      )}
      {file.status === "error" && (
        <AlertCircle size={14} className="text-[var(--error)]" />
      )}
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

interface IngestAdvancedOptions {
  skipMemorize: boolean;
  runInBackground: boolean;
  chunkSizeOverride: number | undefined;
  chunker: ChunkerType;
  incrementalLoading: boolean;
  chunksPerBatch: number;
  dataPerBatch: number;
  enableEpisodeRouting: boolean | undefined;
  enableContentRouting: boolean | undefined;
  enableProcedural: boolean | undefined;
  enableFacetPoints: boolean | undefined;
  preciseMode: boolean | undefined;
  contentType: "text" | "dialog";
}

const DEFAULT_INGEST_OPTIONS: IngestAdvancedOptions = {
  skipMemorize: false,
  runInBackground: true,
  chunkSizeOverride: undefined,
  chunker: "TextChunker",
  incrementalLoading: true,
  chunksPerBatch: 100,
  dataPerBatch: 20,
  enableEpisodeRouting: undefined,
  enableContentRouting: undefined,
  enableProcedural: undefined,
  enableFacetPoints: undefined,
  preciseMode: undefined,
  contentType: "text",
};

export function IngestPage() {
  const { datasetContext } = useUIStore();
  const ingestionConfig = useIngestionConfigStore();
  const chunkConfig = useChunkConfigStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const confirm = useConfirm();

  // Input state
  const [inputType, setInputType] = useState<"text" | "file">("text");
  const [textContent, setTextContent] = useState("");
  const [files, setFiles] = useState<FileWithStatus[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  // Advanced options as object for easier comparison
  const [options, setOptions] = useState<IngestAdvancedOptions>(DEFAULT_INGEST_OPTIONS);
  
  // Advanced options panel
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Fixed to "error" mode - prevent concurrent memorize operations on the same dataset
  const conflictMode = "error" as const;
  
  // Use override if set, otherwise use global chunk config
  const effectiveChunkSize = options.chunkSizeOverride ?? chunkConfig.chunk_size;

  // Check if options are modified
  const isOptionsModified = JSON.stringify(options) !== JSON.stringify(DEFAULT_INGEST_OPTIONS);

  // Handle option change with confirmation
  const handleOptionChange = useCallback(
    async <K extends keyof IngestAdvancedOptions>(key: K, value: IngestAdvancedOptions[K]) => {
      const confirmed = await confirm({
        title: "Change Setting",
        message: `Are you sure you want to change "${key}"?`,
        confirmText: "Confirm",
        cancelText: "Cancel",
        variant: "default",
      });
      if (confirmed) {
        setOptions((prev) => ({ ...prev, [key]: value }));
      }
    },
    [confirm]
  );

  // Reset options to default
  const handleResetOptions = useCallback(async () => {
    const confirmed = await confirm({
      title: "Reset to Defaults",
      message: "Are you sure you want to reset all options to default values?",
      confirmText: "Reset",
      cancelText: "Cancel",
      variant: "warning",
    });
    if (confirmed) {
      setOptions(DEFAULT_INGEST_OPTIONS);
      toast.success("Settings reset to defaults");
    }
  }, [confirm]);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState<"adding" | "memorizing" | null>(null);

  // WebSocket progress tracking
  const [pipelineRunId, setPipelineRunId] = useState<string | null>(null);
  const [processedDatasetName, setProcessedDatasetName] = useState<string>("");
  
  // Result state
  const [resultStatus, setResultStatus] = useState<"idle" | "success" | "error" | "processing">("idle");
  const [resultError, setResultError] = useState<string>("");
  const [filesProcessedCount, setFilesProcessedCount] = useState(0);
  const [processStartTime, setProcessStartTime] = useState<number | null>(null);

  // API hooks
  const ingestText = useIngestText();
  const ingestFiles = useIngestFiles();
  const { data: activePipelines } = useActivePipelines();

  // Dataset name from context or default
  const datasetName = datasetContext.datasetName || "main_dataset";

  // Check if target dataset is currently being processed
  const isTargetDatasetProcessing = useMemo(() => {
    if (!activePipelines || !datasetName) return false;
    return activePipelines.some(p => p.datasetName === datasetName);
  }, [activePipelines, datasetName]);

  // WebSocket hook
  const {
    progress,
    connectionState,
    error: wsError,
    isCompleted,
    isError,
    disconnect,
    reconnect,
  } = useMemorizeWebSocket(pipelineRunId);

  // Handle WebSocket completion
  useEffect(() => {
    let timeoutId: NodeJS.Timeout | null = null;
    
    if (isCompleted) {
      setResultStatus("success");
      toast.success("Knowledge graph built successfully!");
      timeoutId = setTimeout(() => {
        setPipelineRunId(null);
        setProcessedDatasetName("");
      }, 3000);
    }
    if (isError) {
      setResultStatus("error");
      setResultError("Pipeline processing failed");
      toast.error("Processing failed. Please try again.");
    }
    
    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [isCompleted, isError]);

  // File handling with validation
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

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // Cancel progress tracking
  const handleCancelTracking = () => {
    disconnect();
    setPipelineRunId(null);
    setProcessedDatasetName("");
  };

  // Submit handler
  const handleSubmit = async () => {
    // Validation
    if (inputType === "text" && !textContent.trim()) {
      toast.error("Please enter some text content");
      return;
    }
    if (inputType === "file" && files.filter(f => f.status === "pending").length === 0) {
      toast.error("Please select at least one file");
      return;
    }

    setIsProcessing(true);
    setResultStatus("processing");
    setResultError("");
    setProcessStartTime(Date.now());
    
    let processedFiles = 0;

    try {
      if (inputType === "text") {
        // Process text via ingest API
        setProcessingStep("adding");
        
        const response = await ingestText.mutateAsync({
          content: textContent,
          dataset_name: datasetName,
          skip_memorize: options.skipMemorize,
          run_in_background: options.runInBackground,
          chunk_size: effectiveChunkSize || undefined,
          chunks_per_batch: options.chunksPerBatch,
          enable_episode_routing: options.enableEpisodeRouting,
          enable_content_routing: options.enableContentRouting,
          enable_procedural: options.enableProcedural,
          enable_facet_points: options.enableFacetPoints,
          precise_mode: options.preciseMode,
          content_type: options.contentType,
          conflict_mode: conflictMode,
        });

        setTextContent("");
        processedFiles = 1;
        setFilesProcessedCount(processedFiles);

        // Handle response
        if (response.memorize_run_id && options.runInBackground && !options.skipMemorize) {
          setProcessingStep("memorizing");
          setPipelineRunId(response.memorize_run_id);
          setProcessedDatasetName(datasetName);
          toast.info("Processing started - tracking progress...");
        } else if (response.status === "completed") {
          setResultStatus("success");
          toast.success("Knowledge graph built successfully");
        } else if (response.status === "memorize_skipped") {
          setResultStatus("success");
          toast.success("Text added to dataset");
        } else {
          setResultStatus("success");
          toast.success("Processing started in background");
        }
        
      } else if (inputType === "file") {
        // Process files via ingest API
        const pendingFiles = files.filter(f => f.status === "pending");
        
        // Mark all as uploading
        setFiles((prev) => prev.map((f) => 
          f.status === "pending" ? { ...f, status: "uploading" as const } : f
        ));
        setProcessingStep("adding");

        const response = await ingestFiles.mutateAsync({
          files: pendingFiles.map(f => f.file),
          options: {
            datasetName,
            skipMemorize: options.skipMemorize,
            runInBackground: options.runInBackground,
            chunkSize: effectiveChunkSize || undefined,
            chunksPerBatch: options.chunksPerBatch,
            enableEpisodeRouting: options.enableEpisodeRouting,
            enableContentRouting: options.enableContentRouting,
            enableProcedural: options.enableProcedural,
            preciseMode: options.preciseMode,
            conflictMode,
          },
        });

        // Mark all as success
        setFiles((prev) => prev.map((f) => 
          f.status === "uploading" ? { ...f, status: "success" as const } : f
        ));
        processedFiles = pendingFiles.length;
        setFilesProcessedCount(processedFiles);

        // Handle response
        if (response.memorize_run_id && options.runInBackground && !options.skipMemorize) {
          setProcessingStep("memorizing");
          setPipelineRunId(response.memorize_run_id);
          setProcessedDatasetName(datasetName);
          toast.info(`${pendingFiles.length} file(s) ingested - tracking progress...`);
        } else if (response.status === "completed") {
          setResultStatus("success");
          toast.success(`${pendingFiles.length} file(s) processed - knowledge graph built`);
        } else if (response.status === "memorize_skipped") {
          setResultStatus("success");
          toast.success(`${pendingFiles.length} file(s) added to dataset`);
        } else {
          setResultStatus("success");
          toast.success("Processing started in background");
        }
      }

    } catch (error) {
      const message = error instanceof Error ? error.message : "Ingestion failed";
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : message);
      setResultStatus("error");
      setResultError(actionableMessage);
      
      // Mark any uploading files as error
      setFiles((prev) => prev.map((f) => 
        f.status === "uploading" 
          ? { ...f, status: "error" as const, error: actionableMessage } 
          : f
      ));
      
      toast.error(actionableMessage);
      console.error("Ingest error:", error);
    } finally {
      setIsProcessing(false);
      setProcessingStep(null);
    }
  };
  
  // Reset result and retry
  const handleRetry = () => {
    setResultStatus("idle");
    setResultError("");
    setFiles([]);
    setTextContent("");
  };
  
  // Calculate elapsed time
  const elapsedTime = processStartTime 
    ? (resultStatus === "processing" ? Date.now() - processStartTime : Date.now() - processStartTime)
    : undefined;

  // Check if can submit
  const canSubmit = !isProcessing && !isTargetDatasetProcessing && (
    (inputType === "text" && textContent.trim()) ||
    (inputType === "file" && files.filter(f => f.status === "pending").length > 0)
  );

  // Processing status text
  const getProcessingText = () => {
    if (processingStep === "adding") return "Adding data...";
    if (processingStep === "memorizing") return "Building knowledge graph...";
    return "Processing...";
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Quick Import</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Import your data and automatically build a <Term termKey="knowledgeGraph">knowledge graph</Term>.
        </p>
      </div>

      {/* Progress Tracker */}
      {pipelineRunId && (
        <div className="mb-6">
          <ProgressTracker
            pipelineRunId={pipelineRunId}
            status={progress?.status || null}
            connectionState={connectionState}
            error={wsError}
            datasetName={processedDatasetName}
            onRetry={reconnect}
            onCancel={handleCancelTracking}
          />
        </div>
      )}

      {/* Ingestion Result */}
      {resultStatus !== "idle" && !pipelineRunId && (
        <div className="mb-6">
          <IngestionResult
            status={resultStatus}
            datasetName={datasetName}
            filesProcessed={filesProcessedCount}
            timeElapsed={elapsedTime}
            errorMessage={resultError}
            onRetry={handleRetry}
          />
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Input Form */}
        <div className="col-span-7 space-y-5">
          {/* Active Processing Warning */}
          {isTargetDatasetProcessing && (
            <div className="flex items-start gap-2 p-3 bg-amber-500/10 border border-amber-500/30 rounded">
              <AlertTriangle size={14} className="text-amber-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs text-amber-500 font-medium">
                  Dataset &quot;{datasetName}&quot; is currently being processed
                </p>
                <p className="text-[10px] text-amber-500/80 mt-0.5">
                  Please wait for the current operation to complete before adding more data to this dataset.
                </p>
              </div>
            </div>
          )}

          {/* Source Type */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Source Type
            </label>
            <div className="flex gap-1 p-1 bg-[var(--bg-surface)] rounded border border-[var(--border-subtle)]">
              {["text", "file"].map((type) => (
                <button
                  key={type}
                  onClick={() => setInputType(type as "text" | "file")}
                  disabled={isProcessing}
                  className={cn(
                    "flex-1 py-1.5 text-xs rounded transition-colors",
                    inputType === type
                      ? "bg-[var(--bg-base)] text-[var(--text-primary)]"
                      : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]",
                    isProcessing && "opacity-50 cursor-not-allowed"
                  )}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
              {/* URL support requires backend REST API changes */}
              <div className="relative group flex-1">
                <button
                  disabled
                  className="w-full py-1.5 text-xs rounded text-[var(--text-muted)] opacity-50 cursor-not-allowed"
                >
                  URL
                </button>
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                  Coming soon
                </div>
              </div>
            </div>
          </div>

          {/* Input Area */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              {inputType === "text" ? "Content" : "Files"}
            </label>

            {inputType === "text" && (
              <textarea
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                placeholder="Enter or paste text content to ingest..."
                disabled={isProcessing}
                className={cn(
                  "w-full h-40 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] resize-none focus:outline-none focus:border-[var(--text-muted)]",
                  isProcessing && "opacity-50 cursor-not-allowed"
                )}
              />
            )}

            {inputType === "file" && (
              <div className="space-y-3">
                {/* Drop Zone with Accessibility */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => !isProcessing && inputRef.current?.click()}
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" || e.key === " ") && !isProcessing) {
                      e.preventDefault();
                      inputRef.current?.click();
                    }
                  }}
                  role="button"
                  tabIndex={isProcessing ? -1 : 0}
                  aria-label="Drop files here or click to browse. Maximum 50MB per file."
                  aria-disabled={isProcessing}
                  className={cn(
                    "flex flex-col items-center justify-center p-6 border-2 border-dashed rounded cursor-pointer transition-all",
                    "focus:outline-none focus:ring-2 focus:ring-[var(--text-primary)] focus:ring-offset-2 focus:ring-offset-[var(--bg-base)]",
                    isDragging 
                      ? "border-[var(--text-primary)] bg-[var(--text-primary)]/5 scale-[1.02]" 
                      : "border-[var(--border-subtle)] hover:border-[var(--border-default)]",
                    isProcessing && "opacity-50 cursor-not-allowed"
                  )}
                >
                  <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept={ACCEPTED_FILE_TYPES.join(",")}
                    onChange={handleFileSelect}
                    className="hidden"
                    disabled={isProcessing}
                    aria-hidden="true"
                  />
                  <Upload size={20} className={cn(
                    "mb-2 transition-transform",
                    isDragging ? "text-[var(--text-primary)] -translate-y-1" : "text-[var(--text-muted)]"
                  )} />
                  <p className="text-xs text-[var(--text-secondary)]">
                    {isDragging ? "Release to upload" : "Drop files here or click to browse"}
                  </p>
                  <p className="text-[10px] text-[var(--text-muted)] mt-1">Max 50MB per file</p>
                </div>

                {/* File List */}
                {files.length > 0 && (
                  <div className="space-y-2">
                    {files.map((file, index) => (
                      <FileItem 
                        key={`${file.file.name}-${index}`}
                        file={file} 
                        onRemove={() => removeFile(index)} 
                      />
                    ))}
                  </div>
                )}
              </div>
            )}

          </div>

          {/* Options */}
          <div className="flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <div>
              <span className="text-xs text-[var(--text-secondary)]">Skip Memorize</span>
              <p className="text-[10px] text-[var(--text-muted)]">Store data only, build graph later</p>
            </div>
            <Switch
              checked={options.skipMemorize}
              onCheckedChange={(v) => handleOptionChange("skipMemorize", v)}
              disabled={isProcessing}
            />
          </div>

          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={isProcessing || options.skipMemorize}
            className={cn(
              "w-full flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded text-left",
              (isProcessing || options.skipMemorize) && "opacity-50 cursor-not-allowed"
            )}
          >
            <div className="flex items-center gap-2">
              <Settings2 size={14} className="text-[var(--text-muted)]" />
              <span className="text-xs text-[var(--text-secondary)]">Advanced Options</span>
              {isOptionsModified && (
                <span className="px-1.5 py-0.5 text-[9px] bg-amber-500/20 text-amber-400 rounded">Modified</span>
              )}
            </div>
            {showAdvanced ? (
              <ChevronUp size={14} className="text-[var(--text-muted)]" />
            ) : (
              <ChevronDown size={14} className="text-[var(--text-muted)]" />
            )}
          </button>

          {/* Advanced Options Panel */}
          {showAdvanced && !options.skipMemorize && (
            <div className="space-y-4 p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
              {/* Reset Button */}
              {isOptionsModified && (
                <div className="flex justify-end pb-3 mb-3 border-b border-[var(--border-subtle)]">
                  <button
                    onClick={handleResetOptions}
                    className="flex items-center gap-1.5 px-2 py-1 text-[10px] text-amber-400 hover:text-amber-300 hover:bg-amber-500/10 rounded transition-colors"
                  >
                    <RotateCcw size={10} />
                    Reset to defaults
                  </button>
                </div>
              )}

              {/* ── Configuration Wizard ── */}
              <ConfigWizard
                onApply={(result) => setOptions(prev => ({ ...prev, ...result }))}
                disabled={isProcessing}
              />

              {/* Chunk Size */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Chunk Size</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Maximum tokens per chunk. Smaller values = more granular but fragmented. Auto-calculated if not set.
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] text-[var(--text-muted)]">
                    {options.chunkSizeOverride ? `${options.chunkSizeOverride} tokens (override)` : `${chunkConfig.chunk_size} tokens (global)`}
                  </span>
                </div>
                <input
                  type="range"
                  min="256"
                  max="8192"
                  step="256"
                  value={effectiveChunkSize || 2048}
                  onChange={(e) => setOptions(prev => ({ ...prev, chunkSizeOverride: parseInt(e.target.value) }))}
                  disabled={isProcessing}
                  className="w-full h-1.5 bg-[var(--bg-elevated)] rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-[var(--text-muted)]">256</span>
                  <button
                    onClick={() => setOptions(prev => ({ ...prev, chunkSizeOverride: undefined }))}
                    className="text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] underline"
                  >
                    Reset to Auto
                  </button>
                  <span className="text-[10px] text-[var(--text-muted)]">8192</span>
                </div>
              </div>

              {/* Chunker Type */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs text-[var(--text-secondary)]">Chunker Strategy</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      TextChunker: Paragraph-based, preserves semantic boundaries. LangchainChunker: Recursive splitting with overlap.
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setOptions(prev => ({ ...prev, chunker: "TextChunker" }))}
                    disabled={isProcessing}
                    className={cn(
                      "p-2 text-xs rounded border transition-colors",
                      options.chunker === "TextChunker"
                        ? "bg-[var(--text-primary)] text-[var(--bg-base)] border-[var(--text-primary)]"
                        : "bg-[var(--bg-elevated)] text-[var(--text-secondary)] border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
                      isProcessing && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <div className="font-medium">TextChunker</div>
                    <div className="text-[10px] opacity-70 mt-0.5">Paragraph-based</div>
                  </button>
                  <button
                    onClick={() => setOptions(prev => ({ ...prev, chunker: "LangchainChunker" }))}
                    disabled={isProcessing}
                    className={cn(
                      "p-2 text-xs rounded border transition-colors",
                      options.chunker === "LangchainChunker"
                        ? "bg-[var(--text-primary)] text-[var(--bg-base)] border-[var(--text-primary)]"
                        : "bg-[var(--bg-elevated)] text-[var(--text-secondary)] border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
                      isProcessing && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <div className="font-medium">LangchainChunker</div>
                    <div className="text-[10px] opacity-70 mt-0.5">Recursive split</div>
                  </button>
                </div>
              </div>

              {/* Incremental Loading */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-secondary)]">Skip Duplicates</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-40 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Skip files that have already been processed.
                    </div>
                  </div>
                </div>
                <Switch
                  checked={options.incrementalLoading}
                  onCheckedChange={(v) => handleOptionChange("incrementalLoading", v)}
                  disabled={isProcessing}
                />
              </div>

              {/* Chunks Per Batch */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Chunks per Batch</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Number of text chunks processed in each batch. Higher values increase throughput but use more memory.
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] text-[var(--text-muted)]">{options.chunksPerBatch}</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="25"
                  value={options.chunksPerBatch}
                  onChange={(e) => setOptions(prev => ({ ...prev, chunksPerBatch: parseInt(e.target.value) }))}
                  disabled={isProcessing}
                  className="w-full h-1.5 bg-[var(--bg-elevated)] rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-[var(--text-muted)]">50</span>
                  <span className="text-[10px] text-[var(--text-muted)]">300</span>
                </div>
              </div>

              {/* Data Per Batch */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Data Items per Batch</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Number of files/texts processed per batch in the pipeline. Affects task granularity.
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] text-[var(--text-muted)]">{options.dataPerBatch}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="5"
                  value={options.dataPerBatch}
                  onChange={(e) => setOptions(prev => ({ ...prev, dataPerBatch: parseInt(e.target.value) }))}
                  disabled={isProcessing}
                  className="w-full h-1.5 bg-[var(--bg-elevated)] rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-[var(--text-muted)]">5</span>
                  <span className="text-[10px] text-[var(--text-muted)]">100</span>
                </div>
              </div>

              {/* Feature Toggles Section */}
              <div className="pt-3 border-t border-[var(--border-subtle)]">
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-3">Processing Features</p>
                
                {/* Episode Routing */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Episode Routing</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        When new content may closely relate to previously ingested content, enable this to merge them into coherent episodes. Costs extra LLM tokens and time. Disable to allow concurrent episode creation for much faster ingestion of independent documents.
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {options.enableEpisodeRouting === undefined ? "Auto" : options.enableEpisodeRouting ? "On" : "Off"}
                    </span>
                    <Switch
                      checked={options.enableEpisodeRouting === true}
                      onCheckedChange={(v) => handleOptionChange("enableEpisodeRouting", v ? true : (options.enableEpisodeRouting === true ? false : undefined))}
                      disabled={isProcessing}
                    />
                  </div>
                </div>

                {/* Content Routing */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Content Routing</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        When a single text block may contain multiple topics or semantic focuses, enable this to classify each sentence for cleaner memory structure. Costs extra LLM tokens and time per chunk. Disable for simple, single-topic documents.
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {options.enableContentRouting === undefined ? "Auto" : options.enableContentRouting ? "On" : "Off"}
                    </span>
                    <Switch
                      checked={options.enableContentRouting === true}
                      onCheckedChange={(v) => handleOptionChange("enableContentRouting", v ? true : (options.enableContentRouting === true ? false : undefined))}
                      disabled={isProcessing}
                    />
                  </div>
                </div>

                {/* Procedural Memory */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Procedural Memory</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Extract reusable procedures, preferences, and step-by-step knowledge to complement the factual episodic layer with an abstract dimension. Experimental feature. Costs extra LLM tokens and time.
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {options.enableProcedural === undefined ? "Auto" : options.enableProcedural ? "On" : "Off"}
                    </span>
                    <Switch
                      checked={options.enableProcedural === true}
                      onCheckedChange={(v) => handleOptionChange("enableProcedural", v ? true : (options.enableProcedural === true ? false : undefined))}
                      disabled={isProcessing}
                    />
                  </div>
                </div>

                {/* Facet Points */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Facet Points</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Generate fine-grained FacetPoint nodes under each Facet for more precise retrieval matching. Costs extra LLM tokens per facet. Disable for faster processing when broad-level matching is sufficient.
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {options.enableFacetPoints === undefined ? "Auto" : options.enableFacetPoints ? "On" : "Off"}
                    </span>
                    <Switch
                      checked={options.enableFacetPoints === true}
                      onCheckedChange={(v) => handleOptionChange("enableFacetPoints", v ? true : (options.enableFacetPoints === true ? false : undefined))}
                      disabled={isProcessing}
                    />
                  </div>
                </div>

                {/* Precise Summarization */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Precise Summarization</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Preserve ALL factual information (dates, numbers, names, constraints) with zero information loss. Use for contracts, financial reports, or any content where every data point matters. Costs significantly more LLM tokens and time.
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-muted)]">
                      {options.preciseMode === undefined ? "Auto" : options.preciseMode ? "On" : "Off"}
                    </span>
                    <Switch
                      checked={options.preciseMode === true}
                      onCheckedChange={(v) => handleOptionChange("preciseMode", v ? true : (options.preciseMode === true ? false : undefined))}
                      disabled={isProcessing}
                    />
                  </div>
                </div>

                {/* Content Type */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Content Type</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Text: articles, documents, notes. Dialog: chat logs, meeting transcripts. Auto defaults to Text.
                      </div>
                    </div>
                  </div>
                  <select
                    value={options.contentType}
                    onChange={(e) => handleOptionChange("contentType", e.target.value as "text" | "dialog")}
                    disabled={isProcessing}
                    className="text-[10px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-1.5 py-0.5 text-[var(--text-secondary)]"
                  >
                    <option value="text">Text / Article</option>
                    <option value="dialog">Dialog / Conversation</option>
                  </select>
                </div>

              </div>

            </div>
          )}

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={cn(
              "w-full py-2.5 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded transition-opacity flex items-center justify-center gap-2",
              canSubmit ? "hover:opacity-90" : "opacity-50 cursor-not-allowed"
            )}
          >
            {isProcessing && <Loader2 size={14} className="animate-spin" />}
            {isProcessing ? getProcessingText() : "Run Ingest"}
          </button>

          {/* Dataset Info */}
          <p className="text-xs text-[var(--text-muted)]">
            Target: <span className="text-[var(--text-secondary)]">{datasetName}</span>
          </p>
        </div>

        {/* Right: Reference */}
        <div className="col-span-5 space-y-4">
          {/* Current Workflow */}
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <p className="text-xs text-[var(--text-muted)] mb-2">Current Workflow</p>
            <div className="flex items-center gap-2 text-xs">
              <span className="px-2 py-0.5 bg-[var(--bg-elevated)] rounded text-[var(--text-secondary)]">add()</span>
              <span className="text-[var(--text-muted)]">→</span>
              <span className={cn(
                "px-2 py-0.5 rounded",
                options.skipMemorize 
                  ? "bg-[var(--bg-elevated)] text-[var(--text-muted)] line-through" 
                  : "bg-[var(--bg-elevated)] text-[var(--text-secondary)]"
              )}>memorize()</span>
              {!options.skipMemorize && (
                <span className="text-[10px] text-[var(--text-muted)]">
                  {options.runInBackground ? "(async)" : "(sync)"}
                </span>
              )}
            </div>
          </div>

          {/* API Example - Compact */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              API Example
            </label>
            <CodeBlock code={API_EXAMPLE} />
          </div>

          {/* Key Parameters - Compact */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Key Parameters
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              {INGEST_PARAMS.slice(0, 4).map((param) => (
                <div key={param.name} className="px-3 py-1.5 flex items-center justify-between">
                  <code className="text-xs text-[var(--text-primary)]">{param.name}</code>
                  <span className="text-[10px] text-[var(--text-muted)] font-mono">
                    {param.default || param.type}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
