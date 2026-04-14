"use client";

import React, { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useSearch, useDatasetsWithCounts, useUsers } from "@/hooks/use-api";
import { Term, getActionableErrorMessage } from "@/components/common";
import { toast } from "sonner";
import { Loader2, Search, Database, ChevronDown, ChevronUp, Settings2, Info, Eye, RotateCcw } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DatasetMultiSelect } from "@/components/ui/DatasetMultiSelect";
import type { SearchResultItem } from "@/types";

const API_EXAMPLE = `from m_flow import RecallMode

results = await m_flow.search(
    query_text="What was decided in last week's meeting?",
    query_type=RecallMode.EPISODIC,
    top_k=10,
)`;

const PARAMS = [
  { name: "query_text", type: "str", required: true, description: "Natural language query" },
  { name: "query_type", type: "RecallMode", default: "EPISODIC", description: "Retrieval mode" },
  { name: "top_k", type: "int", default: "10", description: "Maximum results to return" },
  { name: "datasets", type: "List[str] | None", default: "None", description: "Limit search to specific datasets" },
  { name: "wide_search_top_k", type: "int", default: "100", description: "Initial candidate pool size" },
  { name: "display_mode", type: "str", default: '"summary"', description: '"summary" or "detail" output format' },
  { name: "max_facets_per_episode", type: "int", default: "4", description: "Max facets returned per episode" },
  { name: "max_points_per_facet", type: "int", default: "8", description: "Max detail points per facet" },
];

interface AdvancedOptions {
  onlyContext: boolean;
  useCombinedContext: boolean;
  wideSearchTopK: number;
  verbose: boolean;
  enableHybridSearch: boolean;
  enableTimeBonus: boolean;
  enableAdaptiveWeights: boolean;
  displayMode: "summary" | "detail";
  maxFacetsPerEpisode: number;
  maxPointsPerFacet: number;
}

const DEFAULT_OPTIONS: AdvancedOptions = {
  onlyContext: false,
  useCombinedContext: false,
  wideSearchTopK: 100,
  verbose: false,
  enableHybridSearch: true,
  enableTimeBonus: false,
  enableAdaptiveWeights: true,
  displayMode: "summary",
  maxFacetsPerEpisode: 4,
  maxPointsPerFacet: 8,
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

interface ParsedNode {
  label: string;
  content: string;
  nodeType: "episode" | "facet" | "facetpoint" | "entity" | "other";
}

function parseNodes(raw: string): { nodes: ParsedNode[]; connections: string | null } | null {
  if (!raw.includes("Node:")) return null;
  const cleaned = raw.replace(/__node_content_start__\n?/g, "").replace(/__node_content_end__\n?/g, "").trim();
  let mainText = cleaned;
  let connections: string | null = null;
  const connIdx = cleaned.indexOf("Connections:");
  if (connIdx !== -1) {
    mainText = cleaned.slice(0, connIdx).trim();
    connections = cleaned.slice(connIdx + "Connections:".length).trim();
  }
  const body = mainText.replace(/^Nodes:\s*\n?/, "");
  const parts = body.split(/(?=^Node: )/m).filter((s) => s.trim());
  const nodes: ParsedNode[] = [];
  for (const part of parts) {
    const match = part.match(/^Node:\s*(.+?)(?:\n|$)([\s\S]*)/);
    if (!match) continue;
    const label = match[1].trim();
    const content = match[2].trim();
    let nodeType: ParsedNode["nodeType"] = "other";
    if (/^\[Episode\]$/i.test(label)) nodeType = "episode";
    else if (/^\[Facet/i.test(label)) nodeType = "facet";
    else if (/^\[FacetPoint/i.test(label)) nodeType = "facetpoint";
    else nodeType = "entity";
    nodes.push({ label, content, nodeType });
  }
  return nodes.length > 0 ? { nodes, connections } : null;
}

const ntColors: Record<string, string> = {
  episode: "bg-blue-500/10 text-blue-400",
  facet: "bg-emerald-500/10 text-emerald-400",
  facetpoint: "bg-amber-500/10 text-amber-400",
  entity: "bg-purple-500/10 text-purple-400",
  other: "bg-[var(--bg-elevated)] text-[var(--text-muted)]",
};

function ResultCard({ result }: { result: SearchResultItem }) {
  const parsed = parseNodes(result.content);
  const plainContent = result.content.replace(/__node_content_start__|__node_content_end__/g, "").replace(/\n{3,}/g, "\n\n").trim();
  return (
    <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded hover:border-[var(--border-default)] transition-colors">
      {parsed ? (
        <div className="space-y-2">
          {parsed.nodes.map((node, i) => (
            <div key={i} className="pl-3 border-l-2 border-[var(--border-subtle)]">
              <span className={`inline-block px-1.5 py-0.5 text-[10px] rounded mb-1 ${ntColors[node.nodeType]}`}>
                {node.label}
              </span>
              {node.content && <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{node.content}</p>}
            </div>
          ))}
          {parsed.connections && (
            <details className="text-[10px] text-[var(--text-muted)]">
              <summary className="cursor-pointer hover:text-[var(--text-secondary)]">Connections</summary>
              <pre className="mt-1 p-2 bg-[var(--bg-elevated)] rounded text-[10px] overflow-x-auto whitespace-pre-wrap">{parsed.connections}</pre>
            </details>
          )}
        </div>
      ) : (
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">{plainContent}</p>
      )}
      <div className="flex items-center gap-3 mt-2 text-[10px] text-[var(--text-muted)]">
        {result.score !== undefined && (
          <span className="px-1.5 py-0.5 bg-[var(--bg-elevated)] rounded">
            Score: {(result.score * 100).toFixed(0)}%
          </span>
        )}
        {result.source_dataset && (
          <span className="flex items-center gap-1">
            <Database size={10} />
            {result.source_dataset}
          </span>
        )}
        {result.episode_id && (
          <span className="font-mono">Episode: {result.episode_id.slice(0, 8)}...</span>
        )}
      </div>
    </div>
  );
}

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

export function EpisodicPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [options, setOptions] = useState<AdvancedOptions>(DEFAULT_OPTIONS);

  const searchMutation = useSearch();
  const { data: datasets } = useDatasetsWithCounts();
  const { data: usersData } = useUsers();
  const confirm = useConfirm();

  // Transform data for DatasetMultiSelect
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

  // Check if all datasets are selected
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
        recall_mode: "EPISODIC",
        query: query.trim(),
        datasets: isAllSelected ? undefined : selectedDatasets,
        top_k: topK,
        only_context: options.onlyContext,
        use_combined_context: options.useCombinedContext,
        wide_search_top_k: options.wideSearchTopK !== 100 ? options.wideSearchTopK : undefined,
        verbose: options.verbose || undefined,
        enable_hybrid_search: options.enableHybridSearch,
        enable_time_bonus: options.enableTimeBonus,
        enable_adaptive_weights: options.enableAdaptiveWeights,
        display_mode: options.displayMode,
        max_facets_per_episode: options.maxFacetsPerEpisode !== 4 ? options.maxFacetsPerEpisode : undefined,
        max_points_per_facet: options.maxPointsPerFacet !== 8 ? options.maxPointsPerFacet : undefined,
      });
    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Search failed");
      toast.error(actionableMessage);
      console.error("Episodic search error:", error);
    }
  };

  const results = searchMutation.data?.results || [];
  const isLoading = searchMutation.isPending;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Episodic Search</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          <Term termKey="semanticSearch">Find information by meaning</Term>, not just keywords. Best for natural language questions and context-aware queries.
        </p>
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
                placeholder="Enter your question..."
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
                description="Return raw context without LLM processing"
                checked={options.onlyContext}
                onChange={(v) => handleOptionChange("onlyContext", v)}
              />

              {(isAllSelected || selectedDatasets.length > 1) && (
                <OptionSwitch
                  label="Merge multi-dataset results"
                  description="Combine results before LLM response"
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
                label="Hybrid search"
                tooltip="Falls back to keyword matching when the query contains numbers, mixed languages, or very short terms"
                checked={options.enableHybridSearch}
                onChange={(v) => handleOptionChange("enableHybridSearch", v)}
              />

              <OptionSwitch
                label="Time bonus"
                tooltip="Boost results matching query time references"
                checked={options.enableTimeBonus}
                onChange={(v) => handleOptionChange("enableTimeBonus", v)}
              />

              <OptionSwitch
                label="Adaptive scoring"
                tooltip="Dynamic weights based on query characteristics"
                checked={options.enableAdaptiveWeights}
                onChange={(v) => handleOptionChange("enableAdaptiveWeights", v)}
              />

              <div className="space-y-1.5 py-2">
                <div className="flex items-center gap-1.5">
                  <label className="text-xs text-[var(--text-secondary)]">Display mode</label>
                  <div className="group relative">
                    <Info size={10} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Summary: Episode summaries only. Detail: Include Facet/Entity details.
                    </div>
                  </div>
                </div>
                <select
                  value={options.displayMode}
                  onChange={(e) => handleOptionChange("displayMode", e.target.value as "summary" | "detail")}
                  className="w-full bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-xs text-[var(--text-secondary)] focus:outline-none focus:border-[var(--border-default)]"
                >
                  <option value="summary">Summary</option>
                  <option value="detail">Detail</option>
                </select>
              </div>

              <OptionSlider
                label="Max facets per episode"
                tooltip="Maximum number of facets returned per episode"
                value={options.maxFacetsPerEpisode}
                onChange={(v) => handleOptionChange("maxFacetsPerEpisode", v)}
                min={1}
                max={10}
                step={1}
              />

              <OptionSlider
                label="Max points per facet"
                tooltip="Maximum detail points per facet"
                value={options.maxPointsPerFacet}
                onChange={(v) => handleOptionChange("maxPointsPerFacet", v)}
                min={1}
                max={20}
                step={1}
              />

              <div className="pt-3 mt-2 border-t border-[var(--border-subtle)]">
                <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Expert</label>
              </div>

              <OptionSlider
                label="Wide Search K"
                tooltip="Candidates in wide search phase. Higher values improve recall but increase latency."
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
                Results ({results.length})
              </label>
              <div className="space-y-2">
                {results.map((result, index) => (
                  <ResultCard key={result.id || index} result={result} />
                ))}
              </div>
            </div>
          )}

          {results.length === 0 && !isLoading && !searchMutation.isError && (
            <div className="py-12 text-center">
              <p className="text-sm text-[var(--text-muted)]">Enter a query to search your memories</p>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                Try: "What decisions were made about..." or "Summarize the discussion on..."
              </p>
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
              Features
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-3 space-y-1 text-xs text-[var(--text-secondary)]">
              <p>• Bundle Search algorithm with path cost scoring</p>
              <p>• Time-aware retrieval with temporal matching</p>
              <p>• FacetPoint two-hop closure for fine-grained results</p>
              <p>• Adaptive scoring based on query characteristics</p>
            </div>
          </div>

          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Parameters
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              {PARAMS.map((param) => (
                <div key={param.name} className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <code className="text-xs text-[var(--text-primary)]">{param.name}</code>
                    {param.required && <span className="text-[10px] text-[var(--error)]">required</span>}
                  </div>
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
