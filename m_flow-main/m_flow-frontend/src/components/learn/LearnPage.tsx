"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { useExtractProcedural, useEpisodesOverview } from "@/hooks/use-api";
import { getActionableErrorMessage } from "@/components/common";
import { useUIStore } from "@/lib/store";
import { toast } from "sonner";
import { 
  Loader2,
  Zap,
  CheckCircle,
  AlertCircle,
  Info,
  ChevronDown,
  Eye,
  RefreshCw,
  Database,
  FileText
} from "lucide-react";

// ============================================================================
// API Example
// ============================================================================

const API_EXAMPLE = `import m_flow

# Extract procedural memories from existing episodic content
result = await m_flow.extract_procedural_from_episodic(
    limit=100,                    # Max episodes to process
    dataset_id="...",             # Optional: filter by dataset
    force_reprocess=False,        # Re-analyze already processed episodes
    dry_run=False,                # Preview mode (no writes)
    episode_ids=None,             # Optional: specific episode IDs
)

print(f"Analyzed: {result['episodes_analyzed']} episodes")
print(f"Created: {result['procedures_created']} procedures")
print(f"Nodes written: {result['nodes_written']} nodes")`;

const REST_EXAMPLE = `POST /api/v1/procedural/extract-from-episodic
Content-Type: application/json

{
  "dataset_id": "uuid-string",
  "limit": 100,
  "force_reprocess": false,
  "dry_run": false,
  "episode_ids": null
}`;

// ============================================================================
// Parameter Schema
// ============================================================================

const PARAM_SCHEMA = [
  { 
    name: "dataset_id", 
    type: "UUID", 
    default: "null", 
    description: "Filter to specific dataset (recommended for multi-dataset)" 
  },
  { 
    name: "limit", 
    type: "int", 
    default: "100", 
    description: "Maximum number of episodes to process (1-1000)" 
  },
  { 
    name: "force_reprocess", 
    type: "bool", 
    default: "false", 
    description: "Re-process episodes already marked as processed" 
  },
  { 
    name: "dry_run", 
    type: "bool", 
    default: "false", 
    description: "Preview which episodes would be processed (no writes)" 
  },
  { 
    name: "episode_ids", 
    type: "List[str]", 
    default: "null", 
    description: "Optional list of specific episode IDs to process" 
  },
];

// ============================================================================
// Components
// ============================================================================

function CodeBlock({ code, title }: { code: string; title?: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="relative group">
      {title && (
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
          {title}
        </div>
      )}
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

function ParamTable() {
  return (
    <div className="border border-[var(--border-subtle)] rounded overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-[var(--bg-elevated)]">
          <tr>
            <th className="text-left p-2 text-[var(--text-muted)] font-medium">Parameter</th>
            <th className="text-left p-2 text-[var(--text-muted)] font-medium">Type</th>
            <th className="text-left p-2 text-[var(--text-muted)] font-medium">Default</th>
            <th className="text-left p-2 text-[var(--text-muted)] font-medium">Description</th>
          </tr>
        </thead>
        <tbody>
          {PARAM_SCHEMA.map((param, idx) => (
            <tr key={param.name} className={idx % 2 === 0 ? "bg-[var(--bg-surface)]" : ""}>
              <td className="p-2 font-mono text-[var(--text-primary)]">{param.name}</td>
              <td className="p-2 text-[var(--text-secondary)]">{param.type}</td>
              <td className="p-2 text-[var(--text-muted)]">{param.default}</td>
              <td className="p-2 text-[var(--text-secondary)]">{param.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Switch({ 
  checked, 
  onChange, 
  label, 
  description 
}: { 
  checked: boolean; 
  onChange: (v: boolean) => void;
  label: string;
  description?: string;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer group">
      <div className="relative mt-0.5">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only"
        />
        <div className={cn(
          "w-9 h-5 rounded-full transition-colors",
          checked ? "bg-[var(--accent)]" : "bg-[var(--bg-elevated)]"
        )} />
        <div className={cn(
          "absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform shadow-sm",
          checked && "translate-x-4"
        )} />
      </div>
      <div className="flex-1">
        <span className="text-xs text-[var(--text-primary)] group-hover:text-[var(--accent)] transition-colors">
          {label}
        </span>
        {description && (
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{description}</p>
        )}
      </div>
    </label>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export function LearnPage() {
  // Dataset context from global store
  const datasetContext = useUIStore((s) => s.datasetContext);
  
  // Form state
  const [limit, setLimit] = useState(100);
  const [episodeIds, setEpisodeIds] = useState("");
  const [forceReprocess, setForceReprocess] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Result state - supports both extraction response types
  const [result, setResult] = useState<{
    success: boolean;
    message: string;
    // Pipeline response fields
    workflow_name?: string;
    datasets_submitted?: number;
    // Legacy extraction fields
    episodes_analyzed?: number;
    procedures_created?: number;
    nodes_written?: number;
    dry_run?: boolean;
    episodes_preview?: string[];
  } | null>(null);

  // Hooks
  const extractMutation = useExtractProcedural();
  const { data: episodesData } = useEpisodesOverview(
    datasetContext?.datasetId 
      ? { datasetId: datasetContext.datasetId }
      : undefined
  );
  
  const totalEpisodes = episodesData?.total ?? 0;

  const handleExtract = async () => {
    try {
      const options: {
        dataset_id?: string;
        limit: number;
        episode_ids?: string[];
        force_reprocess: boolean;
        dry_run: boolean;
      } = { 
        limit,
        force_reprocess: forceReprocess,
        dry_run: dryRun,
      };
      
      // Use dataset from context
      if (datasetContext?.datasetId) {
        options.dataset_id = datasetContext.datasetId;
      }
      
      if (episodeIds.trim()) {
        options.episode_ids = episodeIds
          .split(",")
          .map(id => id.trim())
          .filter(id => id.length > 0);
      }

      const data = await extractMutation.mutateAsync(options);
      const resultData = data as typeof result;
      setResult(resultData);
      
      if (resultData?.success) {
        // Handle different API response formats
        if (resultData.datasets_submitted != null && resultData.datasets_submitted > 0) {
          // Current API format: background extraction started
          toast.success(`Procedural extraction started for ${resultData.datasets_submitted} dataset(s)`);
        } else if (resultData.dry_run) {
          // Legacy: dry run preview
          toast.success(`Preview: ${resultData.episodes_preview?.length ?? 0} episodes would be processed`);
        } else if (resultData.episodes_analyzed != null) {
          // Legacy: direct extraction results
          toast.success(`Extracted ${resultData.procedures_created ?? 0} procedures from ${resultData.episodes_analyzed} episodes`);
        } else {
          // Fallback: generic success
          toast.success(resultData.message || "Extraction completed successfully");
        }
      } else {
        toast.error(resultData?.message || "Extraction completed with issues");
      }
    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Extraction failed");
      toast.error(actionableMessage);
      setResult(null);
    }
  };

  return (
    <div className="h-full overflow-auto bg-[var(--bg-base)]">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-lg font-medium text-[var(--text-primary)]">Extract Procedures</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            Extract procedural memories (methods, steps, patterns) from existing episodic content. 
            This analyzes your stored episodes and identifies reusable procedural knowledge.
          </p>
        </div>

        {/* Dataset Context Banner */}
        {datasetContext?.datasetId ? (
          <div className="mb-6 p-3 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)] flex items-center gap-3">
            <Database className="w-4 h-4 text-[var(--accent)]" />
            <div className="flex-1">
              <span className="text-xs text-[var(--text-muted)]">Active Dataset</span>
              <p className="text-sm text-[var(--text-primary)]">{datasetContext.datasetName}</p>
            </div>
            {totalEpisodes > 0 && (
              <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[var(--bg-surface)]">
                <FileText className="w-3.5 h-3.5 text-[var(--text-muted)]" />
                <span className="text-xs text-[var(--text-secondary)]">{totalEpisodes} episodes</span>
              </div>
            )}
          </div>
        ) : (
          <div className="mb-6 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 flex items-center gap-3">
            <AlertCircle className="w-4 h-4 text-amber-500" />
            <div className="flex-1">
              <span className="text-xs text-amber-500/80">No Dataset Selected</span>
              <p className="text-sm text-[var(--text-secondary)]">
                Processing will include all episodes you have access to. Select a dataset from the sidebar to filter.
              </p>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Documentation */}
          <div className="space-y-6">
            {/* Info Box */}
            <div className="flex items-start gap-3 p-4 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)]">
              <Info className="w-4 h-4 text-[var(--accent)] mt-0.5 flex-shrink-0" />
              <div className="text-xs text-[var(--text-secondary)]">
                <strong className="text-[var(--text-primary)]">What is procedural extraction?</strong>
                <p className="mt-1">
                  Procedural memory stores &quot;how to&quot; knowledge — methods, steps, and patterns. 
                  This feature analyzes your episodic memories to extract reusable procedures 
                  that can be searched and referenced later.
                </p>
              </div>
            </div>

            {/* Python API */}
            <div>
              <h3 className="text-sm font-medium text-[var(--text-primary)] mb-3">
                Python API
              </h3>
              <CodeBlock code={API_EXAMPLE} />
            </div>

            {/* REST API */}
            <div>
              <h3 className="text-sm font-medium text-[var(--text-primary)] mb-3">
                REST API
              </h3>
              <CodeBlock code={REST_EXAMPLE} />
            </div>

            {/* Parameters */}
            <div>
              <h3 className="text-sm font-medium text-[var(--text-primary)] mb-3">
                Parameters
              </h3>
              <ParamTable />
            </div>
          </div>

          {/* Right Column: Form */}
          <div className="space-y-6">
            {/* Extraction Form */}
            <div className="p-6 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
              <h3 className="text-sm font-medium text-[var(--text-primary)] mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4 text-[var(--accent)]" />
                Extract Procedures
              </h3>

              <div className="space-y-5">
                {/* Limit Input */}
                <div>
                  <label className="block text-xs text-[var(--text-muted)] mb-1.5">
                    Maximum Episodes to Process
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={1000}
                    value={limit}
                    onChange={(e) => setLimit(Math.min(1000, Math.max(1, parseInt(e.target.value) || 100)))}
                    className={cn(
                      "w-full px-3 py-2 rounded",
                      "bg-[var(--bg-base)] border border-[var(--border-subtle)]",
                      "text-sm text-[var(--text-primary)]",
                      "focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
                    )}
                  />
                </div>

                {/* Toggle Options */}
                <div className="space-y-3 pt-1">
                  <Switch
                    checked={dryRun}
                    onChange={setDryRun}
                    label="Preview Only"
                    description="See which episodes would be processed without making changes"
                  />
                  <Switch
                    checked={forceReprocess}
                    onChange={setForceReprocess}
                    label="Force Reprocess"
                    description="Re-analyze episodes even if already marked as processed"
                  />
                </div>

                {/* Advanced Options Toggle */}
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
                >
                  <ChevronDown className={cn(
                    "w-3.5 h-3.5 transition-transform",
                    showAdvanced && "rotate-180"
                  )} />
                  Advanced Options
                </button>

                {/* Advanced Options Content */}
                {showAdvanced && (
                  <div className="pl-2 border-l-2 border-[var(--border-subtle)]">
                    <label className="block text-xs text-[var(--text-muted)] mb-1.5">
                      Specific Episode IDs (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={episodeIds}
                      onChange={(e) => setEpisodeIds(e.target.value)}
                      placeholder="Leave empty to process all matching episodes"
                      className={cn(
                        "w-full px-3 py-2 rounded",
                        "bg-[var(--bg-base)] border border-[var(--border-subtle)]",
                        "text-sm text-[var(--text-primary)]",
                        "placeholder:text-[var(--text-muted)]",
                        "focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
                      )}
                    />
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={handleExtract}
                  disabled={extractMutation.isPending}
                  className={cn(
                    "w-full py-2.5 rounded font-medium text-sm",
                    "bg-[var(--accent)] text-white",
                    "hover:bg-[var(--accent-hover)] transition-colors",
                    "disabled:opacity-50 disabled:cursor-not-allowed",
                    "flex items-center justify-center gap-2"
                  )}
                >
                  {extractMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {dryRun ? "Checking..." : "Extracting..."}
                    </>
                  ) : dryRun ? (
                    <>
                      <Eye className="w-4 h-4" />
                      Preview Episodes
                    </>
                  ) : forceReprocess ? (
                    <>
                      <RefreshCw className="w-4 h-4" />
                      Reprocess & Extract
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Extract Procedures
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Result Display */}
            {result && (
              <div className={cn(
                "p-6 rounded-lg border",
                result.dry_run
                  ? "bg-blue-500/5 border-blue-500/20"
                  : result.success 
                    ? "bg-green-500/5 border-green-500/20" 
                    : "bg-amber-500/5 border-amber-500/20"
              )}>
                <div className="flex items-start gap-3">
                  {result.dry_run ? (
                    <Eye className="w-5 h-5 text-blue-500 flex-shrink-0" />
                  ) : result.success ? (
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <h4 className={cn(
                      "text-sm font-medium mb-2",
                      result.dry_run
                        ? "text-blue-500"
                        : result.success ? "text-green-500" : "text-amber-500"
                    )}>
                      {result.dry_run 
                        ? "Preview Results" 
                        : result.success 
                          ? "Extraction Complete" 
                          : "Extraction Finished"
                      }
                    </h4>
                    
                    {result.dry_run ? (
                      <div className="space-y-2">
                        <p className="text-xs text-[var(--text-secondary)]">
                          <strong className="text-[var(--text-primary)]">{result.episodes_preview?.length ?? 0}</strong> episodes would be processed
                        </p>
                        {result.episodes_preview && result.episodes_preview.length > 0 && (
                          <div className="mt-2 max-h-32 overflow-y-auto">
                            <p className="text-[10px] text-[var(--text-muted)] mb-1">Episode IDs:</p>
                            <div className="flex flex-wrap gap-1">
                              {result.episodes_preview.slice(0, 20).map((id) => (
                                <span 
                                  key={id}
                                  className="px-1.5 py-0.5 text-[10px] font-mono bg-[var(--bg-base)] rounded text-[var(--text-secondary)]"
                                >
                                  {id.slice(0, 8)}...
                                </span>
                              ))}
                              {result.episodes_preview.length > 20 && (
                                <span className="px-1.5 py-0.5 text-[10px] text-[var(--text-muted)]">
                                  +{result.episodes_preview.length - 20} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : result.datasets_submitted != null && result.datasets_submitted > 0 ? (
                      <div className="space-y-1 text-xs text-[var(--text-secondary)]">
                        <p>Datasets submitted: <strong className="text-[var(--text-primary)]">{result.datasets_submitted}</strong></p>
                        <p className="text-[var(--text-muted)]">
                          Extraction is running in the background. Check the Pipeline Dashboard for progress.
                        </p>
                      </div>
                    ) : result.episodes_analyzed != null ? (
                      <div className="space-y-1 text-xs text-[var(--text-secondary)]">
                        <p>Episodes analyzed: <strong className="text-[var(--text-primary)]">{result.episodes_analyzed}</strong></p>
                        <p>Procedures created: <strong className="text-[var(--text-primary)]">{result.procedures_created ?? 0}</strong></p>
                        <p>Nodes written: <strong className="text-[var(--text-primary)]">{result.nodes_written ?? 0}</strong></p>
                      </div>
                    ) : (
                      <div className="text-xs text-[var(--text-secondary)]">
                        <p>{result.message}</p>
                      </div>
                    )}
                    
                    <p className="mt-2 text-xs text-[var(--text-muted)]">
                      {result.message}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Tips */}
            <div className="p-4 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)]">
              <h4 className="text-xs font-medium text-[var(--text-primary)] mb-2">Tips</h4>
              <ul className="text-xs text-[var(--text-secondary)] space-y-1">
                <li>• Use <strong>Preview Only</strong> to see which episodes will be analyzed</li>
                <li>• Episodes are automatically marked to prevent duplicate processing</li>
                <li>• Use <strong>Force Reprocess</strong> to re-analyze already processed episodes</li>
                <li>• Extracted procedures can be searched using PROCEDURAL search mode</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
