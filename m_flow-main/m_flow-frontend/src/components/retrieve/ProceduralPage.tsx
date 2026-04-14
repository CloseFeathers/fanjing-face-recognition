"use client";

import React, { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useSearch, useDatasetsWithCounts, useUsers } from "@/hooks/use-api";
import { getActionableErrorMessage } from "@/components/common";
import { toast } from "sonner";
import { Loader2, Search, Database, ListOrdered, ChevronDown, ChevronUp, Settings2, Info, Eye, RotateCcw } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DatasetMultiSelect } from "@/components/ui/DatasetMultiSelect";
import type { SearchResultItem } from "@/types";
import { ProceduralResultCard } from "./ProceduralResultCard";

const API_EXAMPLE = `from m_flow import RecallMode

results = await m_flow.search(
    query_text="How to deploy Python to production?",
    query_type=RecallMode.PROCEDURAL,
    top_k=5,
)`;

const PARAMS = [
  { name: "query_text", type: "str", required: true, description: "Description of the procedure to find" },
  { name: "query_type", type: "RecallMode", default: "PROCEDURAL", description: "Use PROCEDURAL mode" },
  { name: "top_k", type: "int", default: "5", description: "Maximum procedures to return" },
  { name: "expand_steps", type: "bool", default: "True", description: "Include detailed steps" },
];

const EXAMPLES: string[] = [];

interface AdvancedOptions {
  onlyContext: boolean;
  useCombinedContext: boolean;
  wideSearchTopK: number;
  verbose: boolean;
  enableTimeBonus: boolean;
}

const DEFAULT_OPTIONS: AdvancedOptions = {
  onlyContext: false,
  useCombinedContext: false,
  wideSearchTopK: 100,
  verbose: false,
  enableTimeBonus: true,
};

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
        <code className="text-xs text-[var(--text-secondary)] font-mono">{code}</code>
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

// ProceduralResultCard is now imported from a separate file

function OptionSwitch({
  label,
  description,
  tooltip,
  checked,
  onChange,
}: {
  label: string;
  description?: string;
  tooltip?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-[var(--text-secondary)]">{label}</label>
          {tooltip && (
            <div className="group relative">
              <Info size={10} className="text-[var(--text-muted)] cursor-help" />
              <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 whitespace-normal">
                {tooltip}
              </div>
            </div>
          )}
        </div>
        {description && (
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{description}</p>
        )}
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  );
}

function OptionSlider({
  label,
  tooltip,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  tooltip?: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <div className="space-y-1.5 py-2">
      <div className="flex items-center gap-1.5">
        <label className="text-xs text-[var(--text-secondary)]">{label}</label>
        {tooltip && (
          <div className="group relative">
            <Info size={10} className="text-[var(--text-muted)] cursor-help" />
            <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 whitespace-normal">
              {tooltip}
            </div>
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="flex-1 h-1.5 bg-[var(--bg-elevated)] rounded-full appearance-none cursor-pointer accent-blue-600"
        />
        <span className="w-8 text-xs text-[var(--text-secondary)] text-right tabular-nums">{value}</span>
      </div>
    </div>
  );
}

export function ProceduralPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [options, setOptions] = useState<AdvancedOptions>(DEFAULT_OPTIONS);

  const searchMutation = useSearch();
  const { data: datasets } = useDatasetsWithCounts();
  const { data: usersData } = useUsers();
  const confirm = useConfirm();

  const datasetItems = useMemo(() => 
    (datasets ?? []).map(ds => ({
      id: ds.id,
      name: ds.name,
      ownerId: ds.ownerId,
    })),
    [datasets]
  );

  const userItems = useMemo(() =>
    (usersData ?? []).map(u => ({
      id: u.id,
      name: u.email.split("@")[0],
    })),
    [usersData]
  );

  const isAllSelected = selectedDatasets.length === 0 || 
    (datasets && selectedDatasets.length === datasets.length);

  const isModified = JSON.stringify(options) !== JSON.stringify(DEFAULT_OPTIONS);

  const handleOptionChange = useCallback(
    async <K extends keyof AdvancedOptions>(key: K, value: AdvancedOptions[K]) => {
      const confirmed = await confirm({
        title: "Change Setting",
        message: `Are you sure you want to change "${key}" to "${String(value)}"?`,
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

  const handleReset = useCallback(async () => {
    const confirmed = await confirm({
      title: "Reset to Defaults",
      message: "Are you sure you want to reset all options to default values? This cannot be undone.",
      confirmText: "Reset",
      cancelText: "Cancel",
      variant: "warning",
    });
    if (confirmed) {
      setOptions(DEFAULT_OPTIONS);
      toast.success("Settings reset to defaults");
    }
  }, [confirm]);

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.error("Please enter a search query");
      return;
    }

    try {
      await searchMutation.mutateAsync({
        recall_mode: "PROCEDURAL",
        query: query.trim(),
        datasets: isAllSelected ? undefined : selectedDatasets,
        top_k: topK,
        only_context: options.onlyContext,
        use_combined_context: options.useCombinedContext,
        wide_search_top_k: options.wideSearchTopK !== 100 ? options.wideSearchTopK : undefined,
        verbose: options.verbose || undefined,
        enable_time_bonus: options.enableTimeBonus,
      });
    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Search failed");
      toast.error(actionableMessage);
      console.error("Procedural search error:", error);
    }
  };

  const results = searchMutation.data?.results || [];
  const isLoading = searchMutation.isPending;

  const procedureCount = useMemo(() => {
    let count = 0;
    for (const r of results) {
      try {
        const cleaned = r.content
          .replace(/__node_content_start__/g, "")
          .replace(/__node_content_end__/g, "")
          .trim();
        const parsed = JSON.parse(cleaned);
        if (parsed?.__procedural_structured__ && Array.isArray(parsed.procedures)) {
          count += parsed.procedures.length;
          continue;
        }
      } catch {}
      count += 1;
    }
    return count;
  }, [results]);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-4">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Procedural Search</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Find step-by-step instructions, procedures, and guides. Best for &ldquo;how do I...&rdquo; questions.
        </p>
      </div>

      <div className="mb-6 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 flex items-start gap-2.5">
        <Info size={14} className="text-amber-500 mt-0.5 flex-shrink-0" />
        <div className="text-xs text-amber-200/80 leading-relaxed">
          <strong className="text-amber-400">Experimental</strong> — Procedural retrieval is currently in testing. Please understand the parsing and usage patterns thoroughly before deploying to production. Improper use may affect ingestion data quality, model response time, and output quality.
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-7 space-y-5">
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Query
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !isLoading && handleSearch()}
                placeholder="Enter your query..."
                disabled={isLoading}
                className={cn(
                  "flex-1 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)]",
                  isLoading && "opacity-50"
                )}
              />
              <button
                onClick={handleSearch}
                disabled={isLoading || !query.trim()}
                className="px-4 py-2 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search size={14} />
                    Search
                  </>
                )}
              </button>
            </div>

            <div className="flex flex-wrap gap-2 mt-3">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setQuery(ex)}
                  disabled={isLoading}
                  className="px-2 py-1 text-[10px] bg-[var(--bg-surface)] text-[var(--text-muted)] rounded border border-[var(--border-subtle)] hover:text-[var(--text-secondary)] transition-colors disabled:opacity-50"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-xs text-[var(--text-muted)]">Results:</label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={isLoading}
                className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-2 py-1 text-xs text-[var(--text-secondary)] focus:outline-none"
              >
                {[5, 10, 20, 50].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-[var(--text-muted)]">Datasets:</label>
              <DatasetMultiSelect
                datasets={datasetItems}
                users={userItems}
                selected={selectedDatasets}
                onChange={setSelectedDatasets}
                disabled={isLoading}
                placeholder="All datasets"
                className="min-w-[160px]"
              />
            </div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors ml-auto"
            >
              <Settings2 size={12} />
              Advanced
              {isModified && (
                <span className="px-1.5 py-0.5 text-[9px] bg-amber-500/20 text-amber-400 rounded">Modified</span>
              )}
              {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
          </div>

          {showAdvanced && (
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg p-4 space-y-1">
              {isModified && (
                <div className="flex justify-end pb-2 mb-2 border-b border-[var(--border-subtle)]">
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1.5 px-2 py-1 text-[10px] text-amber-400 hover:text-amber-300 hover:bg-amber-500/10 rounded transition-colors"
                  >
                    <RotateCcw size={10} />
                    Reset to defaults
                  </button>
                </div>
              )}

              <OptionSwitch
                label="Context only"
                description="Return raw procedures without processing"
                checked={options.onlyContext}
                onChange={(v) => handleOptionChange("onlyContext", v)}
              />

              {(isAllSelected || selectedDatasets.length > 1) && (
                <OptionSwitch
                  label="Merge multi-dataset results"
                  description="Merge results from multiple datasets"
                  checked={options.useCombinedContext}
                  onChange={(v) => handleOptionChange("useCombinedContext", v)}
                />
              )}

              <div className="pt-3 mt-2 border-t border-[var(--border-subtle)]">
                <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
                  Retrieval Options
                </label>
              </div>

              <OptionSwitch
                label="Time bonus"
                tooltip="Boost results matching query time references (e.g., 'last week', 'yesterday')"
                checked={options.enableTimeBonus}
                onChange={(v) => handleOptionChange("enableTimeBonus", v)}
              />

              <div className="pt-3 mt-2 border-t border-[var(--border-subtle)]">
                <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Expert</label>
              </div>

              <OptionSlider
                label="Wide Search K"
                tooltip="Candidates in wide search phase"
                value={options.wideSearchTopK}
                onChange={(v) => handleOptionChange("wideSearchTopK", v)}
                min={20}
                max={200}
                step={10}
              />

              <div className="flex items-center justify-between py-2">
                <div className="flex items-center gap-1.5">
                  <Eye size={12} className="text-[var(--text-muted)]" />
                  <label className="text-xs text-[var(--text-secondary)]">Verbose output</label>
                </div>
                <Switch
                  checked={options.verbose}
                  onCheckedChange={(v) => handleOptionChange("verbose", v)}
                />
              </div>
            </div>
          )}

          {results.length > 0 && (
            <div>
              <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
                Results ({procedureCount})
              </label>
              <div className="space-y-3">
                {results.map((result, index) => (
                  <ProceduralResultCard key={result.id || index} result={result} index={index} />
                ))}
              </div>
            </div>
          )}

          {results.length === 0 && !isLoading && !searchMutation.isError && (
            <div className="py-12 text-center">
              <p className="text-sm text-[var(--text-muted)]">Enter a query to find procedures</p>
            </div>
          )}

          {searchMutation.isError && (
            <div className="py-8 text-center">
              <p className="text-sm text-[var(--error)]">Search failed</p>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                {searchMutation.error instanceof Error 
                  ? searchMutation.error.message 
                  : "Please try again"}
              </p>
            </div>
          )}
        </div>

        <div className="col-span-5 space-y-5">
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              API Reference
            </label>
            <CodeBlock code={API_EXAMPLE} />
          </div>

          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Use Cases
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-3 space-y-1 text-xs text-[var(--text-secondary)]">
              <p>• Operation manuals and guides</p>
              <p>• Standard Operating Procedures (SOPs)</p>
              <p>• Troubleshooting steps</p>
              <p>• Workflow documentation</p>
              <p>• Best practices</p>
            </div>
          </div>

          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Parameters
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              {PARAMS.map((param) => (
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
