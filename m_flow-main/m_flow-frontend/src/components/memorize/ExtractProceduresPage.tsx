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
  RefreshCw,
  Database,
  FileText,
  AlertTriangle
} from "lucide-react";

function Switch({ 
  checked, 
  onChange, 
  label, 
  description,
  warning
}: { 
  checked: boolean; 
  onChange: (v: boolean) => void;
  label: string;
  description?: string;
  warning?: string;
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
        {warning && checked && (
          <p className="text-[10px] text-amber-500 mt-0.5 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" />
            {warning}
          </p>
        )}
      </div>
    </label>
  );
}

function RemoveProceduralSection({ datasetId }: { datasetId?: string }) {
  const [confirming, setConfirming] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleRemove = async () => {
    setRemoving(true);
    setResult(null);
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = localStorage.getItem("mflow_token") || "";
      const resp = await fetch(`${baseUrl}/api/v1/prune/procedural`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(datasetId ? { dataset_id: datasetId } : {}),
      });
      const data = await resp.json();
      setResult(data.message || "Done");
      toast.success(data.message || "Procedural data removed");
    } catch {
      toast.error("Failed to remove procedural data");
    } finally {
      setRemoving(false);
      setConfirming(false);
    }
  };

  return (
    <div className="mt-6 p-4 rounded-lg border border-red-500/15 bg-red-500/5">
      <h4 className="text-xs font-medium text-red-400 mb-1">Remove All Procedural Data</h4>
      <p className="text-[10px] text-[var(--text-muted)] mb-3">
        Permanently delete all Procedure nodes and their relationships{datasetId ? " from the selected dataset" : " from all datasets"}.
        Episode data is not affected. Episode marks will be cleared so you can re-extract later.
      </p>
      {result && (
        <p className="text-[10px] text-green-400 mb-2">{result}</p>
      )}
      {!confirming ? (
        <button
          onClick={() => setConfirming(true)}
          className="px-3 py-1.5 text-[10px] text-red-400 border border-red-500/20 rounded hover:bg-red-500/10 transition-colors"
        >
          Remove Procedural Data...
        </button>
      ) : (
        <div className="flex items-center gap-2">
          <button
            onClick={handleRemove}
            disabled={removing}
            className="px-3 py-1.5 text-[10px] bg-red-500/20 text-red-400 border border-red-500/30 rounded hover:bg-red-500/30 transition-colors disabled:opacity-50"
          >
            {removing ? "Removing..." : "Confirm Remove"}
          </button>
          <button
            onClick={() => setConfirming(false)}
            className="px-3 py-1.5 text-[10px] text-[var(--text-muted)] border border-[var(--border-subtle)] rounded hover:text-[var(--text-secondary)]"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

export function ExtractProceduresPage() {
  const { datasetContext } = useUIStore();
  
  const [limit, setLimit] = useState(100);
  const [forceReprocess, setForceReprocess] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [resultMessage, setResultMessage] = useState("");

  const extractMutation = useExtractProcedural();
  const { data: episodesData } = useEpisodesOverview(
    datasetContext?.datasetId 
      ? { datasetId: datasetContext.datasetId }
      : undefined
  );
  
  const totalEpisodes = episodesData?.total ?? 0;

  const handleExtract = async () => {
    setSubmitted(false);
    setResultMessage("");
    
    try {
      const options: {
        dataset_id?: string;
        limit: number;
        force_reprocess: boolean;
      } = { 
        limit,
        force_reprocess: forceReprocess,
      };
      
      if (datasetContext?.datasetId) {
        options.dataset_id = datasetContext.datasetId;
      }

      const data = await extractMutation.mutateAsync(options);
      setSubmitted(true);
      setResultMessage(data.message);
      toast.success("Extraction started — check Dashboard for progress");
    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Extraction failed");
      toast.error(actionableMessage);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-4">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Extract Procedures</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Analyze existing episodes and extract reusable procedural knowledge. Progress is visible in the Dashboard.
        </p>
      </div>

      <div className="mb-6 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 flex items-start gap-2.5">
        <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
        <div className="text-xs text-amber-200/80 leading-relaxed">
          <strong className="text-amber-400">Experimental</strong> — Procedural extraction is currently in testing. Please understand the parsing and usage patterns thoroughly before deploying to production. Improper use may affect ingestion data quality, model response time, and output quality.
        </div>
      </div>

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
          <p className="text-sm text-[var(--text-secondary)]">
            No dataset selected. Will process all accessible datasets.
          </p>
        </div>
      )}

      <div className="p-6 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
        <div className="space-y-5">
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

          <div className="space-y-3 pt-1">
            <Switch
              checked={forceReprocess}
              onChange={setForceReprocess}
              label="Force Reprocess"
              description="Re-analyze episodes already marked as processed"
              warning="This may create duplicate procedures"
            />
          </div>

          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
          >
            <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", showAdvanced && "rotate-180")} />
            Advanced Options
          </button>

          {showAdvanced && (
            <div className="pl-2 border-l-2 border-[var(--border-subtle)] text-xs text-[var(--text-muted)]">
              <p>The extraction runs as a background pipeline per dataset.</p>
              <p className="mt-1">Each episode is analyzed by the LLM to identify procedural content (methods, steps, processes).</p>
              <p className="mt-1">Already-processed episodes are skipped unless Force Reprocess is enabled.</p>
            </div>
          )}

          <button
            onClick={handleExtract}
            disabled={extractMutation.isPending || submitted}
            className={cn(
              "w-full py-2.5 rounded font-medium text-sm",
              "bg-[var(--text-primary)] text-[var(--bg-base)]",
              "hover:opacity-90 transition-opacity",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "flex items-center justify-center gap-2"
            )}
          >
            {extractMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Submitting...
              </>
            ) : submitted ? (
              <>
                <CheckCircle className="w-4 h-4" />
                Submitted — Check Dashboard
              </>
            ) : forceReprocess ? (
              <>
                <RefreshCw className="w-4 h-4" />
                Reprocess &amp; Extract
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

      {submitted && (
        <div className="mt-6 p-4 rounded-lg bg-green-500/5 border border-green-500/20 flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-green-500 mb-1">Extraction Pipeline Started</h4>
            <p className="text-xs text-[var(--text-secondary)]">{resultMessage}</p>
            <p className="text-xs text-[var(--text-muted)] mt-2">
              Go to <strong>Dashboard</strong> to monitor progress. The pipeline will show as active until all episodes are processed.
            </p>
          </div>
        </div>
      )}

      <div className="mt-6 p-4 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)]">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-[var(--accent)] mt-0.5" />
          <div>
            <h4 className="text-xs font-medium text-[var(--text-primary)] mb-1">How it works</h4>
            <ul className="text-xs text-[var(--text-secondary)] space-y-0.5">
              <li>• Runs as a background pipeline — progress visible in Dashboard</li>
              <li>• Each dataset is processed separately with its own pipeline record</li>
              <li>• Episodes are automatically marked to prevent duplicate extraction</li>
              <li>• Extracted procedures can be searched using Procedural Search</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Remove All Procedures */}
      <RemoveProceduralSection datasetId={datasetContext?.datasetId ?? undefined} />
    </div>
  );
}
