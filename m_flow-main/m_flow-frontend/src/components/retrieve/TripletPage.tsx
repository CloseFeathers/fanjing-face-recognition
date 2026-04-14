"use client";

import React, { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useSearch, useDatasetsWithCounts, useUsers } from "@/hooks/use-api";
import { Term, getActionableErrorMessage } from "@/components/common";
import { toast } from "sonner";
import { Loader2, Search, Database, ArrowRight, ChevronDown, ChevronUp, Settings2, RotateCcw, Info, Eye } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DatasetMultiSelect } from "@/components/ui/DatasetMultiSelect";
import type { SearchResultItem } from "@/types";

const API_EXAMPLE = `from m_flow import RecallMode

# Natural language query - find relevant triplets
results = await m_flow.search(
    query_text="Who founded Apple Inc?",
    query_type=RecallMode.TRIPLET_COMPLETION,
    top_k=5,
)`;

const PARAMS = [
  { name: "query_text", type: "str", required: true, description: "Natural language query to search knowledge graph" },
  { name: "query_type", type: "RecallMode", default: "TRIPLET_COMPLETION", description: "Use TRIPLET_COMPLETION mode" },
  { name: "top_k", type: "int", default: "5", description: "Maximum triplets to return" },
  { name: "include_evidence", type: "bool", default: "True", description: "Include source text evidence" },
];

interface AdvancedOptions {
  onlyContext: boolean;
  useCombinedContext: boolean;
  systemPrompt: string;
  wideSearchTopK: number;
  tripletDistancePenalty: number;
  verbose: boolean;
  enableHybridSearch: boolean;
  enableTimeBonus: boolean;
  enableAdaptiveWeights: boolean;

}

const DEFAULT_OPTIONS: AdvancedOptions = {
  onlyContext: false,
  useCombinedContext: false,
  systemPrompt: "",
  wideSearchTopK: 100,
  tripletDistancePenalty: 3.5,
  verbose: false,
  enableHybridSearch: true,
  enableTimeBonus: false,
  enableAdaptiveWeights: true,

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

function parseNodeBlocks(raw: string): { nodes: ParsedNode[]; connections: string | null } {
  const cleaned = raw
    .replace(/__node_content_start__\n?/g, "")
    .replace(/__node_content_end__\n?/g, "")
    .trim();

  // Extract Connections block if present
  let mainText = cleaned;
  let connections: string | null = null;
  const connIdx = cleaned.indexOf("Connections:");
  if (connIdx !== -1) {
    mainText = cleaned.slice(0, connIdx).trim();
    connections = cleaned.slice(connIdx + "Connections:".length).trim();
  }

  // Remove leading "Nodes:" header
  const body = mainText.replace(/^Nodes:\s*\n?/, "");

  // Split by "Node:" boundaries
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
  return { nodes, connections };
}

const nodeTypeColors: Record<ParsedNode["nodeType"], string> = {
  episode: "bg-blue-500/10 text-blue-400",
  facet: "bg-emerald-500/10 text-emerald-400",
  facetpoint: "bg-amber-500/10 text-amber-400",
  entity: "bg-purple-500/10 text-purple-400",
  other: "bg-[var(--bg-elevated)] text-[var(--text-muted)]",
};

function StructuredNodeView({ content }: { content: string }) {
  const { nodes, connections } = parseNodeBlocks(content);
  if (nodes.length === 0) {
    const plain = content.replace(/__node_content_start__|__node_content_end__/g, "").replace(/\n{3,}/g, "\n\n").trim();
    return <p className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">{plain}</p>;
  }
  return (
    <div className="space-y-2">
      {nodes.map((node, i) => (
        <div key={i} className="pl-3 border-l-2 border-[var(--border-subtle)]">
          <span className={`inline-block px-1.5 py-0.5 text-[10px] rounded mb-1 ${nodeTypeColors[node.nodeType]}`}>
            {node.label}
          </span>
          {node.content && (
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{node.content}</p>
          )}
        </div>
      ))}
      {connections && (
        <details className="text-[10px] text-[var(--text-muted)]">
          <summary className="cursor-pointer hover:text-[var(--text-secondary)]">Connections ({connections.split("\n").filter(Boolean).length})</summary>
          <pre className="mt-1 p-2 bg-[var(--bg-elevated)] rounded text-[10px] overflow-x-auto whitespace-pre-wrap">{connections}</pre>
        </details>
      )}
    </div>
  );
}

function TripletResultCard({ result }: { result: SearchResultItem }) {
  const hasStructuredTriplet = result.subject && result.predicate && result.object;
  const hasNodeStructure = result.content.includes("Node:");
  
  return (
    <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded hover:border-[var(--border-default)] transition-colors">
      {hasStructuredTriplet ? (
        <div className="flex items-center gap-2 text-sm">
          <span className="px-2 py-0.5 bg-blue-500/10 text-blue-400 rounded text-xs">{result.subject}</span>
          <ArrowRight size={12} className="text-[var(--text-muted)]" />
          <span className="text-[var(--text-secondary)] italic">{result.predicate}</span>
          <ArrowRight size={12} className="text-[var(--text-muted)]" />
          <span className="px-2 py-0.5 bg-green-500/10 text-green-400 rounded text-xs">{result.object}</span>
        </div>
      ) : hasNodeStructure ? (
        <StructuredNodeView content={result.content} />
      ) : (
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
          {result.content.replace(/__node_content_start__|__node_content_end__/g, "").replace(/\n{3,}/g, "\n\n").trim()}
        </p>
      )}
      <div className="flex items-center gap-3 mt-2 text-[10px] text-[var(--text-muted)]">
        {result.score !== undefined && (
          <>
            <div className="flex-1 max-w-[100px] h-1 bg-[var(--bg-elevated)] rounded-full">
              <div
                className="h-full bg-[var(--text-muted)] rounded-full"
                style={{ width: `${result.score * 100}%` }}
              />
            </div>
            <span>{(result.score * 100).toFixed(0)}%</span>
          </>
        )}
        {result.source_dataset && (
          <span className="flex items-center gap-1">
            <Database size={10} />
            {result.source_dataset}
          </span>
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

const ALL_COLLECTIONS = [
  { id: "Episode_summary", label: "Episode Summary", defaultOn: true, locked: false },
  { id: "Entity_name", label: "Entity Name", defaultOn: true, locked: false },
  { id: "Concept_name", label: "Entity Name (legacy)", defaultOn: true, locked: false },
  { id: "Facet_search_text", label: "Facet Search Text", defaultOn: false, locked: false },
  { id: "Facet_anchor_text", label: "Facet Anchor Text", defaultOn: false, locked: false },
  { id: "FacetPoint_search_text", label: "FacetPoint Search Text", defaultOn: false, locked: false },
  { id: "Entity_canonical_name", label: "Entity Canonical Name", defaultOn: false, locked: false },
  { id: "RelationType_relationship_name", label: "Edge Text (always on)", defaultOn: true, locked: true },
];

const DEFAULT_COLLECTION_IDS = ALL_COLLECTIONS.filter(c => c.defaultOn).map(c => c.id);

export function TripletPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [options, setOptions] = useState<AdvancedOptions>(DEFAULT_OPTIONS);
  const [selectedCollections, setSelectedCollections] = useState<string[]>(DEFAULT_COLLECTION_IDS);

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
      if (key === "systemPrompt") {
        setOptions((prev) => ({ ...prev, [key]: value }));
        return;
      }
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

  const isQueryValid = (): boolean => {
    return query.trim().length > 0;
  };

  const handleSearch = async () => {
    if (!isQueryValid()) {
      toast.error("Please enter a query");
      return;
    }

    try {
      await searchMutation.mutateAsync({
        recall_mode: "TRIPLET_COMPLETION",
        query: query.trim(),
        datasets: isAllSelected ? undefined : selectedDatasets,
        top_k: topK,
        only_context: options.onlyContext || undefined,
        use_combined_context: options.useCombinedContext,
        system_prompt: options.systemPrompt || undefined,
        wide_search_top_k: options.wideSearchTopK !== 100 ? options.wideSearchTopK : undefined,
        triplet_distance_penalty: options.tripletDistancePenalty !== 3.5 ? options.tripletDistancePenalty : undefined,
        verbose: options.verbose || undefined,
        enable_hybrid_search: options.enableHybridSearch,
        enable_time_bonus: options.enableTimeBonus,
        enable_adaptive_weights: options.enableAdaptiveWeights,
        collections: selectedCollections.length > 0 ? selectedCollections : undefined,
      });
    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Search failed");
      toast.error(actionableMessage);
      console.error("Triplet search error:", error);
    }
  };

  const results = searchMutation.data?.results || [];
  const isLoading = searchMutation.isPending;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Graph Triplets</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Search <Term termKey="entity">entity</Term> <Term termKey="relationship">relationships</Term> in your knowledge graph using natural language.
        </p>
      </div>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-7 space-y-5">
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Query
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !isLoading && isQueryValid() && handleSearch()}
              placeholder="Enter your query..."
              disabled={isLoading}
              className={cn(
                "w-full bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-3 py-2.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)]",
                isLoading && "opacity-50"
              )}
            />
            <p className="text-xs text-[var(--text-muted)] mt-2">
              Ask any question - the system will find relevant triplets from your knowledge graph.
            </p>
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
                description="Return raw triplets without LLM answer"
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

              <OptionSlider
                label="Distance Penalty"
                tooltip="Penalty for triplet distance in graph"
                value={options.tripletDistancePenalty}
                onChange={(v) => handleOptionChange("tripletDistancePenalty", v)}
                min={0.5}
                max={10}
                step={0.5}
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

              {/* Collections */}
              <div className="pt-3 mt-2 border-t border-[var(--border-subtle)]">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
                    Vector Collections
                  </label>
                  <button
                    onClick={() => setSelectedCollections(DEFAULT_COLLECTION_IDS)}
                    className="text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                  >
                    Reset defaults
                  </button>
                </div>
                <p className="text-[10px] text-[var(--text-muted)] mb-2">
                  Select which vector collections to search. Dot (•) = default. Locked items are always included.
                </p>
                <div className="grid grid-cols-2 gap-1.5">
                  {ALL_COLLECTIONS.map((col) => (
                    <label
                      key={col.id}
                      className={cn(
                        "flex items-center gap-1.5 text-[10px]",
                        col.locked
                          ? "text-[var(--text-muted)] cursor-not-allowed"
                          : "text-[var(--text-secondary)] cursor-pointer"
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={col.locked || selectedCollections.includes(col.id)}
                        disabled={col.locked}
                        onChange={(e) => {
                          if (col.locked) return;
                          if (e.target.checked) {
                            setSelectedCollections((prev) => [...prev, col.id]);
                          } else {
                            setSelectedCollections((prev) => prev.filter((c) => c !== col.id));
                          }
                        }}
                        className={cn(
                          "w-3 h-3",
                          col.locked ? "accent-gray-600 opacity-50" : "accent-[var(--text-primary)]"
                        )}
                      />
                      <span>
                        {col.label}
                        {col.defaultOn && !col.locked && <span className="text-[var(--text-muted)]"> •</span>}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}

          <button
            onClick={handleSearch}
            disabled={isLoading || !isQueryValid()}
            className="w-full py-2.5 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Searching...
              </>
            ) : (
              <>
                <Search size={14} />
                Search Triplets
              </>
            )}
          </button>

          {results.length > 0 && (
            <div>
              <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
                Results ({results.length})
              </label>
              <div className="space-y-2">
                {results.map((result, index) => (
                  <TripletResultCard key={result.id || index} result={result} />
                ))}
              </div>
            </div>
          )}

          {results.length === 0 && !isLoading && !searchMutation.isError && (
            <div className="py-12 text-center">
              <p className="text-sm text-[var(--text-muted)]">Enter a triplet query to search relationships</p>
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
