"use client";

import React, { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useSearch, useDatasetsWithCounts, useUsers } from "@/hooks/use-api";
import { Term, getActionableErrorMessage } from "@/components/common";
import { toast } from "sonner";
import { Loader2, Play, Database, Code2, ChevronDown, ChevronUp, Settings2, RotateCcw } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { DatasetMultiSelect } from "@/components/ui/DatasetMultiSelect";
import type { SearchResultItem } from "@/types";

const API_EXAMPLE = `from m_flow import RecallMode

results = await m_flow.search(
    query_text="""
    MATCH (p:Person)-[:WORKS_AT]->(c:Company)
    WHERE c.name = 'Google'
    RETURN p.name, c.name LIMIT 10
    """,
    query_type=RecallMode.CYPHER,
)`;

const TEMPLATES = [
  { name: "All Entities", query: "MATCH (n:Entity)\nRETURN n.name, labels(n)\nLIMIT 20" },
  { name: "Find Relations", query: "MATCH (a)-[r]->(b)\nRETURN a.name, type(r), b.name\nLIMIT 20" },
  { name: "Entity Neighbors", query: 'MATCH (n {name: "TARGET"})-[r]-(neighbor)\nRETURN n.name, type(r), neighbor.name' },
  { name: "Two-Hop Paths", query: 'MATCH path = (a)-[*1..2]-(b)\nWHERE a.name = "START"\nRETURN path LIMIT 10' },
];

const PARAMS = [
  { name: "query_text", type: "str", required: true, description: "Cypher query statement" },
  { name: "query_type", type: "RecallMode", default: "CYPHER", description: "Use CYPHER mode" },
  { name: "timeout", type: "int", default: "30", description: "Query timeout in seconds" },
  { name: "limit", type: "int", default: "100", description: "Default result limit if not in query" },
];

interface AdvancedOptions {
  onlyContext: boolean;
}

const DEFAULT_OPTIONS: AdvancedOptions = {
  onlyContext: false,
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

export function CypherPage() {
  const [query, setQuery] = useState("");
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

  const handleExecute = async () => {
    if (!query.trim()) {
      toast.error("Please enter a Cypher query");
      return;
    }

    try {
      await searchMutation.mutateAsync({
        recall_mode: "CYPHER",
        query: query.trim(),
        datasets: isAllSelected ? undefined : selectedDatasets,
        only_context: options.onlyContext,
      });
    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Query execution failed");
      toast.error(actionableMessage);
      console.error("Cypher query error:", error);
    }
  };

  const results = searchMutation.data?.results || [];
  const isLoading = searchMutation.isPending;

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Graph Query</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Write custom graph queries using <Term termKey="cypher">Cypher</Term> syntax. For power users who need precise control over graph traversal.
        </p>
      </div>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-7 space-y-5">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider">
                Cypher Query
              </label>
              <span className="text-[10px] text-[var(--text-muted)]">Cmd+Enter to execute</span>
            </div>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !isLoading) {
                  handleExecute();
                }
              }}
              disabled={isLoading}
              className={cn(
                "w-full h-32 bg-[#0a0a0a] border border-[var(--border-subtle)] rounded p-3 text-xs text-[var(--text-secondary)] font-mono resize-none focus:outline-none focus:border-[var(--text-muted)]",
                isLoading && "opacity-50"
              )}
            />
          </div>

          <div className="flex items-center gap-4">
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
              {showAdvanced ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
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

              <div className="flex items-center justify-between py-2">
                <div>
                  <label className="text-xs text-[var(--text-secondary)]">Context only</label>
                  <p className="text-[10px] text-[var(--text-muted)]">Return raw query results only</p>
                </div>
                <Switch
                  checked={options.onlyContext}
                  onCheckedChange={(v) => handleOptionChange("onlyContext", v)}
                />
              </div>
            </div>
          )}

          <button
            onClick={handleExecute}
            disabled={isLoading || !query.trim()}
            className="w-full py-2.5 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Executing...
              </>
            ) : (
              <>
                <Play size={14} />
                Execute Query
              </>
            )}
          </button>

          {results.length > 0 && (
            <div>
              <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
                Results ({results.length} rows)
              </label>
              <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded overflow-x-auto max-h-96">
                <div className="space-y-2 p-3">
                  {results.map((result: SearchResultItem, index: number) => (
                    <div
                      key={result.id || index}
                      className="p-2 bg-[var(--bg-elevated)] rounded text-xs text-[var(--text-secondary)] font-mono"
                    >
                      {result.content}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {results.length === 0 && !isLoading && !searchMutation.isError && (
            <div className="py-8 text-center border border-dashed border-[var(--border-subtle)] rounded-lg">
              <Code2 size={24} className="mx-auto text-[var(--text-muted)] mb-2" />
              <p className="text-sm text-[var(--text-muted)]">Write a Cypher query and execute it</p>
            </div>
          )}

          {searchMutation.isError && (
            <div className="py-8 text-center">
              <p className="text-sm text-[var(--error)]">Query execution failed</p>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                {searchMutation.error instanceof Error 
                  ? searchMutation.error.message 
                  : "Please check your query syntax"}
              </p>
            </div>
          )}
        </div>

        <div className="col-span-5 space-y-5">
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Templates
            </label>
            <div className="space-y-1">
              {TEMPLATES.map((t) => (
                <button
                  key={t.name}
                  onClick={() => setQuery(t.query)}
                  disabled={isLoading}
                  className="w-full text-left p-2 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded text-xs text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors disabled:opacity-50"
                >
                  {t.name}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              API Reference
            </label>
            <CodeBlock code={API_EXAMPLE} />
          </div>

          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Cypher Syntax
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-3 space-y-1 text-[10px] text-[var(--text-secondary)] font-mono">
              <p><span className="text-[var(--text-muted)]">MATCH</span> — Pattern matching</p>
              <p><span className="text-[var(--text-muted)]">WHERE</span> — Filter conditions</p>
              <p><span className="text-[var(--text-muted)]">RETURN</span> — Results to return</p>
              <p><span className="text-[var(--text-muted)]">(n)</span> — Node</p>
              <p><span className="text-[var(--text-muted)]">-[r]-&gt;</span> — Relationship</p>
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
