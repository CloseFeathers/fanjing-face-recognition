"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { cn } from "@/lib/utils";
import { useUIStore, useIngestionConfigStore, useChunkConfigStore } from "@/lib/store";
import { useDatasets, useMemorize } from "@/hooks/use-api";
import { useActivePipelines } from "@/hooks/use-active-pipelines";
import { useMemorizeWebSocket, extractPipelineRunIds, areAllPipelinesCompleted } from "@/hooks/use-memorize-websocket";
import { MultiProgressTracker, IngestionResult, getActionableErrorMessage, Term } from "@/components/common";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { Loader2, AlertCircle, Database, ChevronDown, ChevronUp, Info, Settings2, AlertTriangle } from "lucide-react";
import type { MemorizeResponse, ChunkerType } from "@/types";
import { ConfigWizard, type WizardResult } from "./ConfigWizard";

// ============================================================================
// Constants
// ============================================================================

const API_EXAMPLE = `import m_flow

result = await m_flow.memorize(
    datasets=["my_dataset"],
    chunk_size=512,
    run_in_background=True,
)`;

const MEMORIZE_PARAMS = [
  { name: "datasets", type: "str | List[str] | None", default: "None", description: "Datasets to process. None = all user datasets" },
  { name: "chunker", type: "TextChunker | LangchainChunker", default: "TextChunker", description: "Text chunking strategy" },
  { name: "chunk_size", type: "int | None", default: "auto", description: "Max tokens per chunk" },
  { name: "chunks_per_batch", type: "int", default: "100", description: "Chunks per processing batch" },
  { name: "run_in_background", type: "bool", default: "False", description: "Async processing for large datasets" },
];

const PIPELINE_STEPS = [
  "Document Classification",
  "Text Chunking", 
  "Content Routing",
  "Text Summarization",
  "Entity Extraction",
  "Graph Persistence",
];

// ============================================================================
// Types
// ============================================================================

interface PipelineTrackingInfo {
  pipelineRunId: string;
  datasetName: string;
}

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

// ============================================================================
// Main Page
// ============================================================================

export function MemorizePage() {
  const { datasetContext } = useUIStore();
  const ingestionConfig = useIngestionConfigStore();

  // Form state
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [allDatasets, setAllDatasets] = useState(true);
  const [runInBackground, setRunInBackground] = useState(true);
  
  // Chunk config from global settings
  const chunkConfig = useChunkConfigStore();
  
  // Advanced options
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [chunkSizeOverride, setChunkSizeOverride] = useState<number | undefined>(undefined);
  const [chunker, setChunker] = useState<ChunkerType>("TextChunker");
  const [incrementalLoading, setIncrementalLoading] = useState(true);
  const [chunksPerBatch, setChunksPerBatch] = useState(100);
  const [dataPerBatch, setDataPerBatch] = useState(20);
  
  // Fixed to "error" mode - prevent concurrent memorize operations on the same dataset
  const conflictMode = "error" as const;
  
  // Feature toggles
  const [enableEpisodeRouting, setEnableEpisodeRouting] = useState<boolean | undefined>(undefined);
  const [enableContentRouting, setEnableContentRouting] = useState<boolean | undefined>(undefined);
  const [enableProcedural, setEnableProcedural] = useState<boolean | undefined>(undefined);
  const [contentType, setContentType] = useState<"text" | "dialog" | undefined>(undefined);
  const [enableFacetPoints, setEnableFacetPoints] = useState<boolean | undefined>(undefined);
  
  // Use override if set, otherwise use global chunk config
  const effectiveChunkSize = chunkSizeOverride ?? chunkConfig.chunk_size;

  // Pipeline tracking state
  const [trackingPipelines, setTrackingPipelines] = useState<PipelineTrackingInfo[]>([]);
  const [activePipelineIndex, setActivePipelineIndex] = useState(0);
  
  // Result state
  const [resultStatus, setResultStatus] = useState<"idle" | "success" | "error" | "processing">("idle");
  const [resultError, setResultError] = useState<string>("");
  const [processStartTime, setProcessStartTime] = useState<number | null>(null);

  // API hooks
  const { data: datasets, isLoading: datasetsLoading, error: datasetsError } = useDatasets();
  const memorize = useMemorize();
  const { data: activePipelines } = useActivePipelines();

  const isProcessing = memorize.isPending;

  // Determine which datasets are currently being processed by active pipelines
  const processingDatasetIds = useMemo(() => {
    if (!activePipelines) return new Set<string>();
    return new Set(activePipelines.filter(p => p.datasetId).map(p => p.datasetId!));
  }, [activePipelines]);

  const isDatasetProcessing = useCallback((datasetId: string) => {
    return processingDatasetIds.has(datasetId);
  }, [processingDatasetIds]);

  // Active pipeline for WebSocket tracking
  const activePipeline = trackingPipelines[activePipelineIndex];

  // WebSocket hook for current pipeline
  const {
    progress,
    connectionState,
    error: wsError,
    isCompleted,
    isError,
    disconnect,
    reconnect,
  } = useMemorizeWebSocket(activePipeline?.pipelineRunId || null);

  // Handle pipeline completion - move to next pipeline
  useEffect(() => {
    let timeoutId: NodeJS.Timeout | null = null;
    
    if (isCompleted && activePipelineIndex < trackingPipelines.length - 1) {
      setActivePipelineIndex((prev) => prev + 1);
    } else if (isCompleted && activePipelineIndex === trackingPipelines.length - 1) {
      setResultStatus("success");
      toast.success("All knowledge graphs built successfully!");
      timeoutId = setTimeout(() => {
        setTrackingPipelines([]);
        setActivePipelineIndex(0);
      }, 3000);
    }
    
    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [isCompleted, activePipelineIndex, trackingPipelines.length]);

  // Handle error
  useEffect(() => {
    if (isError) {
      setResultStatus("error");
      setResultError(`Pipeline processing failed for ${activePipeline?.datasetName || "dataset"}`);
      toast.error(`Pipeline ${activePipeline?.datasetName || ""} failed`);
    }
  }, [isError, activePipeline?.datasetName]);

  // Toggle dataset selection
  const toggleDataset = (datasetName: string) => {
    setAllDatasets(false);
    setSelectedDatasets((prev) => 
      prev.includes(datasetName) 
        ? prev.filter(d => d !== datasetName)
        : [...prev, datasetName]
    );
  };

  // Select all datasets
  const selectAll = () => {
    setAllDatasets(true);
    setSelectedDatasets([]);
  };

  // Extract tracking info from memorize response
  const extractTrackingInfo = (response: MemorizeResponse): PipelineTrackingInfo[] => {
    const pipelineRunIds = extractPipelineRunIds(response);
    return pipelineRunIds.map((pipelineRunId) => {
      const entry = Object.entries(response).find(
        ([, info]) => info.workflow_run_id === pipelineRunId
      );
      return {
        pipelineRunId,
        datasetName: entry?.[1]?.dataset_name || entry?.[0] || "Unknown",
      };
    });
  };

  // Cancel all tracking
  const handleCancelTracking = () => {
    disconnect();
    setTrackingPipelines([]);
    setActivePipelineIndex(0);
  };

  // Handle submit
  const handleSubmit = async () => {
    setResultStatus("processing");
    setResultError("");
    setProcessStartTime(Date.now());
    
    try {
      const datasetsToProcess = allDatasets 
        ? datasets?.map(ds => ds.name) || [] 
        : selectedDatasets.length > 0 
          ? selectedDatasets 
          : undefined;

      const result = await memorize.mutateAsync({
        datasets: datasetsToProcess,
        run_in_background: runInBackground,
        chunk_size: effectiveChunkSize || undefined,
        chunker: chunker,
        incremental_loading: incrementalLoading,
        chunks_per_batch: chunksPerBatch,
        items_per_batch: dataPerBatch,
        conflict_mode: conflictMode,
        enable_episode_routing: enableEpisodeRouting,
        enable_content_routing: enableContentRouting,
        enable_procedural: enableProcedural,
        enable_facet_points: enableFacetPoints,
        ...(contentType ? { content_type: contentType } : {}),
      });

      // Check if all datasets were already processed (no new data)
      if (areAllPipelinesCompleted(result)) {
        setResultStatus("success");
        toast.info("All selected datasets are already up to date. Add new content first.");
        return;
      }

      // Extract pipeline info for WebSocket tracking
      const trackingInfo = extractTrackingInfo(result);
      
      if (runInBackground && trackingInfo.length > 0) {
        setTrackingPipelines(trackingInfo);
        setActivePipelineIndex(0);
        toast.info(`Tracking ${trackingInfo.length} pipeline(s)...`);
      } else if (runInBackground) {
        setResultStatus("success");
        toast.success("Knowledge graph processing started in background");
      } else {
        setResultStatus("success");
        toast.success("Knowledge graph built successfully");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Memorize operation failed";
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : message);
      setResultStatus("error");
      setResultError(actionableMessage);
      toast.error(actionableMessage);
      console.error("Memorize error:", error);
    }
  };

  // Can submit check
  const canSubmit = !isProcessing && (allDatasets || selectedDatasets.length > 0);

  // Reset and retry
  const handleRetry = () => {
    setResultStatus("idle");
    setResultError("");
    setSelectedDatasets([]);
    setAllDatasets(true);
  };
  
  // Calculate elapsed time
  const elapsedTime = processStartTime 
    ? Date.now() - processStartTime
    : undefined;
  
  // Get dataset name(s) for result display
  const resultDatasetName = allDatasets 
    ? "All datasets" 
    : selectedDatasets.join(", ") || "No datasets";

  // Build pipeline statuses for MultiProgressTracker
  const pipelineStatuses = trackingPipelines.map((pipeline, index) => ({
    pipelineRunId: pipeline.pipelineRunId,
    datasetName: pipeline.datasetName,
    status: index < activePipelineIndex 
      ? ("RunCompleted" as const)
      : index === activePipelineIndex 
        ? (progress?.status || ("RunStarted" as const))
        : ("RunStarted" as const),
    isActive: index === activePipelineIndex,
  }));

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Build Knowledge Graph</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Process your stored data into a structured <Term termKey="knowledgeGraph">knowledge graph</Term>. Extracts <Term termKey="entity">entities</Term>, <Term termKey="relationship">relationships</Term>, and creates semantic <Term termKey="embedding">embeddings</Term>.
        </p>
      </div>

      {/* Progress Tracker */}
      {trackingPipelines.length > 0 && (
        <div className="mb-6">
          <MultiProgressTracker
            pipelines={pipelineStatuses}
            connectionState={connectionState}
            error={wsError}
            onRetry={reconnect}
            onCancel={handleCancelTracking}
          />
        </div>
      )}

      {/* Ingestion Result */}
      {resultStatus !== "idle" && trackingPipelines.length === 0 && (
        <div className="mb-6">
          <IngestionResult
            status={resultStatus}
            datasetName={resultDatasetName}
            timeElapsed={elapsedTime}
            errorMessage={resultError}
            onRetry={handleRetry}
          />
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Config */}
        <div className="col-span-7 space-y-5">
          {/* Active Pipeline Warning */}
          {activePipelines && activePipelines.length > 0 && (
            <div className="flex items-start gap-2 p-3 bg-amber-500/10 border border-amber-500/30 rounded">
              <AlertTriangle size={14} className="text-amber-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs text-amber-500 font-medium">
                  {activePipelines.length} pipeline(s) currently processing
                </p>
                <p className="text-[10px] text-amber-500/80 mt-0.5">
                  Datasets being processed are disabled. Wait for completion or use different datasets.
                </p>
              </div>
            </div>
          )}

          {/* Dataset Selection */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Dataset
            </label>

            {datasetsLoading ? (
              <div className="flex items-center gap-2 p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
                <Loader2 size={14} className="animate-spin text-[var(--text-muted)]" />
                <span className="text-xs text-[var(--text-muted)]">Loading datasets...</span>
              </div>
            ) : datasetsError ? (
              <div className="flex items-center gap-2 p-3 bg-[var(--bg-surface)] border border-[var(--error)] rounded">
                <AlertCircle size={14} className="text-[var(--error)]" />
                <span className="text-xs text-[var(--error)]">Failed to load datasets</span>
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={selectAll}
                  disabled={isProcessing || trackingPipelines.length > 0}
                  className={cn(
                    "px-3 py-1.5 text-xs rounded border transition-colors",
                    allDatasets
                      ? "bg-[var(--text-primary)] text-[var(--bg-base)] border-[var(--text-primary)]"
                      : "bg-[var(--bg-surface)] text-[var(--text-secondary)] border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
                    (isProcessing || trackingPipelines.length > 0) && "opacity-50 cursor-not-allowed"
                  )}
                >
                  All Datasets
                </button>
                {datasets && datasets.length > 0 ? (
                  datasets.map((ds) => {
                    const isCurrentlyProcessing = isDatasetProcessing(ds.id);
                    return (
                      <button
                        key={ds.name}
                        onClick={() => !isCurrentlyProcessing && toggleDataset(ds.name)}
                        disabled={isProcessing || trackingPipelines.length > 0 || isCurrentlyProcessing}
                        className={cn(
                          "px-3 py-1.5 text-xs rounded border transition-colors flex items-center gap-1.5",
                          !allDatasets && selectedDatasets.includes(ds.name)
                            ? "bg-[var(--text-primary)] text-[var(--bg-base)] border-[var(--text-primary)]"
                            : "bg-[var(--bg-surface)] text-[var(--text-secondary)] border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
                          (isProcessing || trackingPipelines.length > 0) && "opacity-50 cursor-not-allowed",
                          isCurrentlyProcessing && "opacity-50 cursor-not-allowed bg-amber-500/10 border-amber-500/30"
                        )}
                      >
                        <Database size={10} />
                        {ds.name}
                        {isCurrentlyProcessing && (
                          <Loader2 size={10} className="animate-spin text-amber-500" />
                        )}
                      </button>
                    );
                  })
                ) : (
                  <span className="text-xs text-[var(--text-muted)] p-2">No datasets found. Add data first.</span>
                )}
              </div>
            )}

            {!allDatasets && selectedDatasets.length > 0 && (
              <p className="text-xs text-[var(--text-muted)] mt-2">
                Selected: {selectedDatasets.length} dataset(s)
              </p>
            )}
          </div>

          {/* Options */}
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
              <div>
                <span className="text-xs text-[var(--text-secondary)]">Run in Background</span>
                <p className="text-[10px] text-[var(--text-muted)]">Process asynchronously for large datasets</p>
              </div>
              <Switch
                checked={runInBackground}
                onCheckedChange={setRunInBackground}
                disabled={isProcessing || trackingPipelines.length > 0}
              />
            </div>

          </div>

          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={isProcessing || trackingPipelines.length > 0}
            className={cn(
              "w-full flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded text-left",
              (isProcessing || trackingPipelines.length > 0) && "opacity-50 cursor-not-allowed"
            )}
          >
            <div className="flex items-center gap-2">
              <Settings2 size={14} className="text-[var(--text-muted)]" />
              <span className="text-xs text-[var(--text-secondary)]">Advanced Processing Options</span>
            </div>
            {showAdvanced ? (
              <ChevronUp size={14} className="text-[var(--text-muted)]" />
            ) : (
              <ChevronDown size={14} className="text-[var(--text-muted)]" />
            )}
          </button>

          {/* Advanced Options Panel */}
          {showAdvanced && (
            <div className="space-y-4 p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
              {/* Chunk Size */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--text-secondary)]">Chunk Size</span>
                    <div className="group relative">
                      <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Maximum tokens per chunk. Smaller values create more granular but fragmented knowledge. Auto-calculated if not set.
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] text-[var(--text-muted)]">
                    {chunkSizeOverride ? `${chunkSizeOverride} tokens (override)` : `${chunkConfig.chunk_size} tokens (global)`}
                  </span>
                </div>
                <input
                  type="range"
                  min="256"
                  max="8192"
                  step="256"
                  value={effectiveChunkSize || 2048}
                  onChange={(e) => setChunkSizeOverride(parseInt(e.target.value))}
                  disabled={isProcessing || trackingPipelines.length > 0}
                  className="w-full h-1 bg-[var(--bg-elevated)] rounded-lg appearance-none cursor-pointer accent-[var(--text-primary)]"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-[var(--text-muted)]">256</span>
                  <button
                    onClick={() => setChunkSizeOverride(undefined)}
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
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      TextChunker: Paragraph-based chunking, preserves semantic boundaries. LangchainChunker: Recursive character splitting with overlap for better context.
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setChunker("TextChunker")}
                    disabled={isProcessing || trackingPipelines.length > 0}
                    className={cn(
                      "p-2 text-xs rounded border transition-colors",
                      chunker === "TextChunker"
                        ? "bg-[var(--text-primary)] text-[var(--bg-base)] border-[var(--text-primary)]"
                        : "bg-[var(--bg-elevated)] text-[var(--text-secondary)] border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
                      (isProcessing || trackingPipelines.length > 0) && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <div className="font-medium">TextChunker</div>
                    <div className="text-[10px] opacity-70 mt-0.5">Paragraph-based (default)</div>
                  </button>
                  <button
                    onClick={() => setChunker("LangchainChunker")}
                    disabled={isProcessing || trackingPipelines.length > 0}
                    className={cn(
                      "p-2 text-xs rounded border transition-colors",
                      chunker === "LangchainChunker"
                        ? "bg-[var(--text-primary)] text-[var(--bg-base)] border-[var(--text-primary)]"
                        : "bg-[var(--bg-elevated)] text-[var(--text-secondary)] border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
                      (isProcessing || trackingPipelines.length > 0) && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <div className="font-medium">LangchainChunker</div>
                    <div className="text-[10px] opacity-70 mt-0.5">Recursive with overlap</div>
                  </button>
                </div>
              </div>

              {/* Incremental Loading */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-secondary)]">Skip Duplicates</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-44 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Skip files and content that have already been processed.
                    </div>
                  </div>
                </div>
                <Switch
                  checked={incrementalLoading}
                  onCheckedChange={setIncrementalLoading}
                  disabled={isProcessing || trackingPipelines.length > 0}
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
                  <span className="text-[10px] text-[var(--text-muted)]">{chunksPerBatch}</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="25"
                  value={chunksPerBatch}
                  onChange={(e) => setChunksPerBatch(parseInt(e.target.value))}
                  disabled={isProcessing || trackingPipelines.length > 0}
                  className="w-full h-1 bg-[var(--bg-elevated)] rounded-lg appearance-none cursor-pointer accent-[var(--text-primary)]"
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
                  <span className="text-[10px] text-[var(--text-muted)]">{dataPerBatch}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="5"
                  value={dataPerBatch}
                  onChange={(e) => setDataPerBatch(parseInt(e.target.value))}
                  disabled={isProcessing || trackingPipelines.length > 0}
                  className="w-full h-1 bg-[var(--bg-elevated)] rounded-lg appearance-none cursor-pointer accent-[var(--text-primary)]"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-[var(--text-muted)]">5</span>
                  <span className="text-[10px] text-[var(--text-muted)]">100</span>
                </div>
              </div>

              {/* Configuration Wizard */}
              <ConfigWizard
                onApply={(result: WizardResult) => {
                  if (result.enableEpisodeRouting !== undefined) setEnableEpisodeRouting(result.enableEpisodeRouting);
                  if (result.enableContentRouting !== undefined) setEnableContentRouting(result.enableContentRouting);
                  if (result.preciseMode !== undefined) {
                    /* MemorizePage has no preciseMode state; skip */
                  }
                  if (result.contentType !== undefined) setContentType(result.contentType);
                }}
                disabled={isProcessing || trackingPipelines.length > 0}
              />

              {/* Feature Toggles Divider */}
              <div className="pt-2 border-t border-[var(--border-subtle)]">
                <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Feature Toggles</span>
              </div>

              {/* Episode Routing */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-secondary)]">Episode Routing</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      When new content may closely relate to previously ingested content, enable this to merge them into coherent episodes. Costs extra LLM tokens and time. Disable for faster independent-document ingestion.
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--text-muted)]">{enableEpisodeRouting === undefined ? "Auto" : enableEpisodeRouting ? "On" : "Off"}</span>
                  <select
                    value={enableEpisodeRouting === undefined ? "auto" : enableEpisodeRouting ? "on" : "off"}
                    onChange={(e) => setEnableEpisodeRouting(e.target.value === "auto" ? undefined : e.target.value === "on")}
                    disabled={isProcessing || trackingPipelines.length > 0}
                    className="text-[10px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-1.5 py-0.5 text-[var(--text-secondary)]"
                  >
                    <option value="auto">Auto</option>
                    <option value="on">On</option>
                    <option value="off">Off</option>
                  </select>
                </div>
              </div>

              {/* Content Routing */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-secondary)]">Content Routing</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      When a single text block may contain multiple topics, enable this to classify each sentence for cleaner memory structure. Costs extra LLM tokens per chunk.
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--text-muted)]">{enableContentRouting === undefined ? "Auto" : enableContentRouting ? "On" : "Off"}</span>
                  <select
                    value={enableContentRouting === undefined ? "auto" : enableContentRouting ? "on" : "off"}
                    onChange={(e) => setEnableContentRouting(e.target.value === "auto" ? undefined : e.target.value === "on")}
                    disabled={isProcessing || trackingPipelines.length > 0}
                    className="text-[10px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-1.5 py-0.5 text-[var(--text-secondary)]"
                  >
                    <option value="auto">Auto</option>
                    <option value="on">On</option>
                    <option value="off">Off</option>
                  </select>
                </div>
              </div>

              {/* Content Type */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-secondary)]">Content Type</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      &quot;text&quot; for articles/documents, &quot;dialog&quot; for chat logs and meeting transcripts (splits by speaker turn). Auto-detected when left on Auto.
                    </div>
                  </div>
                </div>
                <select
                  value={contentType ?? "auto"}
                  onChange={(e) => setContentType(e.target.value === "auto" ? undefined : e.target.value as "text" | "dialog")}
                  disabled={isProcessing || trackingPipelines.length > 0}
                  className="text-[10px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-1.5 py-0.5 text-[var(--text-secondary)]"
                >
                  <option value="auto">Auto</option>
                  <option value="text">Text / Article</option>
                  <option value="dialog">Dialog / Conversation</option>
                </select>
              </div>

              {/* Procedural Extraction */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-secondary)]">Procedural Extraction</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Extract reusable procedures and preferences to complement episodic facts with abstract knowledge. Experimental. Costs extra LLM tokens.
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--text-muted)]">{enableProcedural === undefined ? "Auto" : enableProcedural ? "On" : "Off"}</span>
                  <select
                    value={enableProcedural === undefined ? "auto" : enableProcedural ? "on" : "off"}
                    onChange={(e) => setEnableProcedural(e.target.value === "auto" ? undefined : e.target.value === "on")}
                    disabled={isProcessing || trackingPipelines.length > 0}
                    className="text-[10px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-1.5 py-0.5 text-[var(--text-secondary)]"
                  >
                    <option value="auto">Auto</option>
                    <option value="on">On</option>
                    <option value="off">Off</option>
                  </select>
                </div>
              </div>

              {/* Facet Points */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--text-secondary)]">Facet Points</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Generate fine-grained FacetPoint nodes under each Facet for more precise retrieval matching. Costs extra LLM tokens per facet.
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--text-muted)]">{enableFacetPoints === undefined ? "Auto" : enableFacetPoints ? "On" : "Off"}</span>
                  <select
                    value={enableFacetPoints === undefined ? "auto" : enableFacetPoints ? "on" : "off"}
                    onChange={(e) => setEnableFacetPoints(e.target.value === "auto" ? undefined : e.target.value === "on")}
                    disabled={isProcessing || trackingPipelines.length > 0}
                    className="text-[10px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-1.5 py-0.5 text-[var(--text-secondary)]"
                  >
                    <option value="auto">Auto</option>
                    <option value="on">On</option>
                    <option value="off">Off</option>
                  </select>
                </div>
              </div>

            </div>
          )}

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={!canSubmit || trackingPipelines.length > 0}
            className={cn(
              "w-full py-2.5 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded transition-opacity flex items-center justify-center gap-2",
              (canSubmit && trackingPipelines.length === 0) ? "hover:opacity-90" : "opacity-50 cursor-not-allowed"
            )}
          >
            {isProcessing && <Loader2 size={14} className="animate-spin" />}
            {isProcessing ? "Building knowledge graph..." : "Run Memorize"}
          </button>

          {/* Workflow Info */}
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <p className="text-xs text-[var(--text-muted)] mb-2">Workflow Position</p>
            <div className="flex items-center gap-2 text-xs">
              <span className="px-2 py-0.5 bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded">add()</span>
              <span className="text-[var(--text-muted)]">→</span>
              <span className="px-2 py-0.5 bg-[var(--text-primary)] text-[var(--bg-base)] rounded font-medium">memorize()</span>
            </div>
          </div>
        </div>

        {/* Right: Reference */}
        <div className="col-span-5 space-y-5">
          {/* Pipeline Steps */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Processing Pipeline
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              {PIPELINE_STEPS.map((step, i) => (
                <div key={step} className="px-3 py-2 flex items-center gap-3">
                  <span className="text-[10px] text-[var(--text-muted)] w-4">{i + 1}</span>
                  <span className="text-xs text-[var(--text-secondary)]">{step}</span>
                </div>
              ))}
            </div>
          </div>

          {/* API Example */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              API Reference
            </label>
            <CodeBlock code={API_EXAMPLE} />
          </div>

          {/* Parameters */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Parameters
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              {MEMORIZE_PARAMS.slice(0, 4).map((param) => (
                <div key={param.name} className="px-3 py-2">
                  <code className="text-xs text-[var(--text-primary)]">{param.name}</code>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{param.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
