"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSearch, useDatasets } from "@/hooks/use-api";
import { useSearchConfigStore, useRetrievalConfigStore } from "@/lib/store";
import { SearchModeSelector } from "./SearchModeSelector";
import { SearchResultItem } from "./SearchResultItem";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Loader2,
  MessageSquare,
  RotateCcw,
  Info,
  Eye,
  X,
} from "lucide-react";
import { RecallMode, SearchResultItem as SearchResultItemType } from "@/types";

export function SearchPanel() {
  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [results, setResults] = useState<SearchResultItemType[]>([]);
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [nodeNameInput, setNodeNameInput] = useState("");
  const [corefEnabled, setCorefEnabled] = useState(false);
  const [corefSessionId, setCorefSessionId] = useState("");
  const [corefNewTurn, setCorefNewTurn] = useState(false);
  const [keywordMatchBonus, setKeywordMatchBonus] = useState(0.15);
  const [directEpisodePenalty, setDirectEpisodePenalty] = useState(0.3);

  const {
    recallMode,
    topK,
    onlyContext,
    useCombinedContext,
    selectedDatasets,
    systemPrompt,
    wideSearchTopK,
    tripletDistancePenalty,
    verbose,
    setRecallMode,
    setTopK,
    setOnlyContext,
    setUseCombinedContext,
    setSelectedDatasets,
    setSystemPrompt,
    setWideSearchTopK,
    setTripletDistancePenalty,
    nodeNames,
    addNodeName,
    removeNodeName,
    setVerbose,
  } = useSearchConfigStore();

  const retrievalConfig = useRetrievalConfigStore();
  const searchMutation = useSearch();
  const { data: datasets } = useDatasets();

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    const startTime = Date.now();

    try {
      const response = await searchMutation.mutateAsync({
        recall_mode: recallMode,
        query: query.trim(),
        datasets: selectedDatasets.length > 0 ? selectedDatasets : undefined,
        top_k: topK,
        only_context: onlyContext,
        use_combined_context: useCombinedContext,
        system_prompt: systemPrompt || undefined,
        wide_search_top_k: wideSearchTopK !== 100 ? wideSearchTopK : undefined,
        triplet_distance_penalty: tripletDistancePenalty !== 3.5 ? tripletDistancePenalty : undefined,
        verbose: verbose || undefined,
        // Episodic retrieval parameters
        enable_hybrid_search: retrievalConfig.enable_hybrid_search,
        enable_time_bonus: retrievalConfig.enable_time_bonus,
        edge_miss_cost: retrievalConfig.edge_miss_cost !== 0.9 ? retrievalConfig.edge_miss_cost : undefined,
        hop_cost: retrievalConfig.hop_cost !== 0.05 ? retrievalConfig.hop_cost : undefined,
        full_number_match_bonus: retrievalConfig.full_number_match_bonus !== 0.12 ? retrievalConfig.full_number_match_bonus : undefined,
        enable_adaptive_weights: retrievalConfig.enable_adaptive_weights,
        // Episodic output control
        display_mode: retrievalConfig.display_mode,
        max_facets_per_episode: retrievalConfig.max_facets_per_episode !== 4 ? retrievalConfig.max_facets_per_episode : undefined,
        max_points_per_facet: retrievalConfig.max_points_per_facet !== 8 ? retrievalConfig.max_points_per_facet : undefined,
        // Node name filter
        node_name: nodeNames.length > 0 ? nodeNames : undefined,
        // Coreference parameters
        coref_enabled: corefEnabled || undefined,
        coref_session_id: corefSessionId || undefined,
        coref_new_turn: corefNewTurn || undefined,
        // Additional episodic parameters
        keyword_match_bonus: keywordMatchBonus !== 0.15 ? keywordMatchBonus : undefined,
        direct_episode_penalty: directEpisodePenalty !== 0.3 ? directEpisodePenalty : undefined,
      });
      setResults(response.results || []);
      setSearchTime(Date.now() - startTime);
    } catch (error) {
      console.error("Search failed:", error);
      setResults([]);
    }
  }, [query, recallMode, selectedDatasets, topK, onlyContext, useCombinedContext, systemPrompt, wideSearchTopK, tripletDistancePenalty, verbose, retrievalConfig, searchMutation, nodeNames, corefEnabled, corefSessionId, corefNewTurn, keywordMatchBonus, directEpisodePenalty]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Search</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">Query your knowledge base with multiple retrieval modes.</p>
      </div>

      {/* Mode Selector */}
      <SearchModeSelector value={recallMode} onChange={setRecallMode} />

      {/* Search Input */}
      <div className="space-y-3">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Search size={16} strokeWidth={1.5} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={getPlaceholder(recallMode)}
              className="w-full h-11 pl-10 pr-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[14px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] transition-colors"
            />
          </div>
          <Button onClick={handleSearch} loading={searchMutation.isPending} className="h-11 px-5">
            Search
          </Button>
        </div>

        {/* Advanced Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-[13px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
        >
          <SlidersHorizontal size={14} strokeWidth={1.5} />
          Advanced
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {/* Advanced Options */}
        <AnimatePresence>
          {showAdvanced && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="grid grid-cols-2 gap-4 pt-4 border-t border-[var(--border-subtle)]">
                {/* Top K */}
                <div className="space-y-2">
                  <Label className="text-[12px] text-[var(--text-muted)]">Results (Top K)</Label>
                  <div className="flex items-center gap-3">
                    <Slider value={[topK]} onValueChange={([v]) => setTopK(v)} min={1} max={50} step={1} className="flex-1" />
                    <span className="w-6 text-[13px] text-[var(--text-secondary)]">{topK}</span>
                  </div>
                </div>

                {/* Dataset Filter */}
                <div className="space-y-2">
                  <Label className="text-[12px] text-[var(--text-muted)]">Dataset</Label>
                  <Select value={selectedDatasets[0] || "all"} onValueChange={(v) => setSelectedDatasets(v === "all" ? [] : [v])}>
                    <SelectTrigger className="h-9">
                      <SelectValue placeholder="All datasets" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All datasets</SelectItem>
                      {datasets?.map((ds) => (
                        <SelectItem key={ds.id} value={ds.name}>{ds.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Merge Multi-Dataset Results - only show when searching all datasets */}
                {selectedDatasets.length === 0 && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <Label className="text-[12px] text-[var(--text-muted)]">Merge multi-dataset results</Label>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          Combine results from all datasets before generating LLM response.
                        </div>
                      </div>
                    </div>
                    <Switch checked={useCombinedContext} onCheckedChange={setUseCombinedContext} />
                  </div>
                )}

                {/* Hybrid Search (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="flex items-center justify-between">
                    <Label className="text-[12px] text-[var(--text-muted)]">Hybrid search</Label>
                    <Switch checked={retrievalConfig.enable_hybrid_search} onCheckedChange={(v) => retrievalConfig.setConfig({ enable_hybrid_search: v })} />
                  </div>
                )}

                {/* Time Bonus (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="flex items-center justify-between">
                    <Label className="text-[12px] text-[var(--text-muted)]">Time bonus</Label>
                    <Switch checked={retrievalConfig.enable_time_bonus} onCheckedChange={(v) => retrievalConfig.setConfig({ enable_time_bonus: v })} />
                  </div>
                )}

                {/* Adaptive Weights (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="flex items-center justify-between">
                    <Label className="text-[12px] text-[var(--text-muted)]">Adaptive scoring</Label>
                    <Switch checked={retrievalConfig.enable_adaptive_weights} onCheckedChange={(v) => retrievalConfig.setConfig({ enable_adaptive_weights: v })} />
                  </div>
                )}

                {/* Display Mode (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-1.5">
                      <Label className="text-[12px] text-[var(--text-muted)]">Display mode</Label>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          Summary: Episode summaries only. Detail: Include Facet and Entity details.
                        </div>
                      </div>
                    </div>
                    <Select value={retrievalConfig.display_mode || "summary"} onValueChange={(v) => retrievalConfig.setConfig({ display_mode: v as "summary" | "detail" })}>
                      <SelectTrigger className="h-9">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="summary">Summary</SelectItem>
                        <SelectItem value="detail">Detail</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {/* Max Facets per Episode (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-1.5">
                      <Label className="text-[12px] text-[var(--text-muted)]">Max facets per episode</Label>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          Maximum number of facets returned per episode.
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Slider
                        value={[retrievalConfig.max_facets_per_episode || 4]}
                        onValueChange={([v]) => retrievalConfig.setConfig({ max_facets_per_episode: v })}
                        min={1}
                        max={10}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-6 text-[13px] text-[var(--text-secondary)]">{retrievalConfig.max_facets_per_episode || 4}</span>
                    </div>
                  </div>
                )}

                {/* Max Points per Facet (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-1.5">
                      <Label className="text-[12px] text-[var(--text-muted)]">Max points per facet</Label>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          Maximum number of detail points per facet.
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Slider
                        value={[retrievalConfig.max_points_per_facet || 8]}
                        onValueChange={([v]) => retrievalConfig.setConfig({ max_points_per_facet: v })}
                        min={1}
                        max={20}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-6 text-[13px] text-[var(--text-secondary)]">{retrievalConfig.max_points_per_facet || 8}</span>
                    </div>
                  </div>
                )}

                {/* Node Name Filter (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="col-span-2 space-y-2">
                    <Label className="text-[12px] text-[var(--text-muted)]">Node name filter</Label>
                    <div className="flex items-center gap-2">
                      <input
                        value={nodeNameInput}
                        onChange={(e) => setNodeNameInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && nodeNameInput.trim()) {
                            e.preventDefault();
                            addNodeName(nodeNameInput.trim());
                            setNodeNameInput("");
                          }
                        }}
                        placeholder="Type a name and press Enter..."
                        className="flex-1 h-9 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] transition-colors"
                      />
                    </div>
                    {nodeNames.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {nodeNames.map((name) => (
                          <span key={name} className="inline-flex items-center gap-1 px-2 py-0.5 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[12px] text-[var(--text-secondary)]">
                            {name}
                            <button onClick={() => removeNodeName(name)} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors">
                              <X size={10} />
                            </button>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Coreference Controls (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <>
                    <div className="flex items-center justify-between">
                      <Label className="text-[12px] text-[var(--text-muted)]">Coreference resolution</Label>
                      <Switch checked={corefEnabled} onCheckedChange={setCorefEnabled} />
                    </div>
                    {corefEnabled && (
                      <>
                        <div className="space-y-2">
                          <Label className="text-[12px] text-[var(--text-muted)]">Coref session ID</Label>
                          <input
                            value={corefSessionId}
                            onChange={(e) => setCorefSessionId(e.target.value)}
                            placeholder="Session identifier..."
                            className="w-full h-9 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] transition-colors"
                          />
                        </div>
                        <div className="flex items-center justify-between">
                          <Label className="text-[12px] text-[var(--text-muted)]">New turn</Label>
                          <Switch checked={corefNewTurn} onCheckedChange={setCorefNewTurn} />
                        </div>
                      </>
                    )}
                  </>
                )}

                {/* Edge Miss Cost (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="space-y-2">
                    <Label className="text-[12px] text-[var(--text-muted)]">Edge miss cost</Label>
                    <div className="flex items-center gap-3">
                      <Slider value={[retrievalConfig.edge_miss_cost ?? 0.9]} onValueChange={([v]) => retrievalConfig.setConfig({ edge_miss_cost: v })} min={0} max={2} step={0.1} className="flex-1" />
                      <span className="w-8 text-[13px] text-[var(--text-secondary)] text-right">{(retrievalConfig.edge_miss_cost ?? 0.9).toFixed(1)}</span>
                    </div>
                  </div>
                )}

                {/* Hop Cost (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="space-y-2">
                    <Label className="text-[12px] text-[var(--text-muted)]">Hop cost</Label>
                    <div className="flex items-center gap-3">
                      <Slider value={[retrievalConfig.hop_cost ?? 0.05]} onValueChange={([v]) => retrievalConfig.setConfig({ hop_cost: v })} min={0} max={0.5} step={0.01} className="flex-1" />
                      <span className="w-8 text-[13px] text-[var(--text-secondary)] text-right">{(retrievalConfig.hop_cost ?? 0.05).toFixed(2)}</span>
                    </div>
                  </div>
                )}

                {/* Keyword Match Bonus (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="space-y-2">
                    <Label className="text-[12px] text-[var(--text-muted)]">Keyword match bonus</Label>
                    <div className="flex items-center gap-3">
                      <Slider value={[keywordMatchBonus]} onValueChange={([v]) => setKeywordMatchBonus(v)} min={0} max={0.5} step={0.01} className="flex-1" />
                      <span className="w-8 text-[13px] text-[var(--text-secondary)] text-right">{keywordMatchBonus.toFixed(2)}</span>
                    </div>
                  </div>
                )}

                {/* Direct Episode Penalty (EPISODIC) */}
                {recallMode === "EPISODIC" && (
                  <div className="space-y-2">
                    <Label className="text-[12px] text-[var(--text-muted)]">Direct episode penalty</Label>
                    <div className="flex items-center gap-3">
                      <Slider value={[directEpisodePenalty]} onValueChange={([v]) => setDirectEpisodePenalty(v)} min={0} max={1} step={0.05} className="flex-1" />
                      <span className="w-8 text-[13px] text-[var(--text-secondary)] text-right">{directEpisodePenalty.toFixed(2)}</span>
                    </div>
                  </div>
                )}

                {/* System Prompt - Only shown in TRIPLET_COMPLETION mode */}
                {recallMode === "TRIPLET_COMPLETION" && (
                  <div className="col-span-2 space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-[12px] text-[var(--text-muted)] flex items-center gap-1.5">
                        <MessageSquare size={12} />
                        System Prompt
                      </Label>
                      {systemPrompt && (
                        <button
                          onClick={() => setSystemPrompt("")}
                          className="text-[11px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] flex items-center gap-1 transition-colors"
                        >
                          <RotateCcw size={10} />
                          Reset
                        </button>
                      )}
                    </div>
                    <textarea
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      placeholder="Custom instructions for the LLM response..."
                      rows={3}
                      className="w-full px-3 py-2 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] transition-colors resize-none"
                    />
                    <p className="text-[11px] text-[var(--text-muted)]">
                      Customize how the LLM generates responses based on retrieved context.
                    </p>
                  </div>
                )}

                {/* Advanced Search Parameters */}
                <div className="col-span-2 pt-3 border-t border-[var(--border-subtle)]">
                  <Label className="text-[11px] text-[var(--text-muted)] uppercase tracking-wider">Expert Options</Label>
                </div>

                {/* Wide Search Top K */}
                <div className="space-y-2">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-[12px] text-[var(--text-muted)]">Wide Search K</Label>
                    <div className="group relative">
                      <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Number of candidates in the wide search phase. Higher values improve recall but increase latency.
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Slider 
                      value={[wideSearchTopK]} 
                      onValueChange={([v]) => setWideSearchTopK(v)} 
                      min={20} 
                      max={200} 
                      step={10} 
                      className="flex-1" 
                    />
                    <span className="w-8 text-[13px] text-[var(--text-secondary)] text-right">{wideSearchTopK}</span>
                  </div>
                </div>

                {/* Triplet Distance Penalty - Only shown in TRIPLET_COMPLETION mode */}
                {recallMode === "TRIPLET_COMPLETION" && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-1.5">
                      <Label className="text-[12px] text-[var(--text-muted)]">Distance Penalty</Label>
                      <div className="group relative">
                        <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                        <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                          Penalty for triplet distance in graph traversal. Lower values prefer closer relationships.
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Slider 
                        value={[tripletDistancePenalty]} 
                        onValueChange={([v]) => setTripletDistancePenalty(v)} 
                        min={0.5} 
                        max={10} 
                        step={0.5} 
                        className="flex-1" 
                      />
                      <span className="w-8 text-[13px] text-[var(--text-secondary)] text-right">{tripletDistancePenalty}</span>
                    </div>
                  </div>
                )}

                {/* Verbose Mode */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Eye size={12} className="text-[var(--text-muted)]" />
                    <Label className="text-[12px] text-[var(--text-muted)]">Verbose output</Label>
                    <div className="group relative">
                      <Info size={11} className="text-[var(--text-muted)] cursor-help" />
                      <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                        Include detailed graph representation in results for debugging.
                      </div>
                    </div>
                  </div>
                  <Switch checked={verbose} onCheckedChange={setVerbose} />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Results */}
      <div className="space-y-3">
        {/* Stats */}
        {(results.length > 0 || searchMutation.isError) && (
          <div className="flex items-center gap-2 text-[13px] text-[var(--text-secondary)]">
            {searchMutation.isError ? (
              <>
                <AlertCircle size={14} className="text-[var(--error)]" />
                <span className="text-[var(--error)]">Search failed</span>
              </>
            ) : (
              <>
                <span>{results.length} results</span>
                {searchTime && <span className="text-[var(--text-muted)]">· {searchTime}ms</span>}
              </>
            )}
          </div>
        )}

        {/* Loading */}
        {searchMutation.isPending && (
          <div className="flex flex-col items-center py-12">
            <Loader2 size={20} className="text-[var(--text-muted)] animate-spin mb-3" />
            <p className="text-[13px] text-[var(--text-muted)]">Searching...</p>
          </div>
        )}

        {/* Results List */}
        <AnimatePresence mode="popLayout">
          {results.map((item, index) => (
            <motion.div
              key={item.id || index}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ delay: index * 0.03 }}
            >
              <SearchResultItem item={item} recallMode={recallMode} index={index} />
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Empty State */}
        {!searchMutation.isPending && results.length === 0 && query.trim() !== "" && !searchMutation.isError && (
          <div className="text-center py-12">
            <p className="text-[var(--text-muted)]">No results found</p>
          </div>
        )}
      </div>
    </div>
  );
}

function getPlaceholder(mode: RecallMode): string {
  switch (mode) {
    case "EPISODIC": return "Search episodic memories...";
    case "PROCEDURAL": return "Search procedural knowledge...";
    case "TRIPLET_COMPLETION": return "Search knowledge triplets...";
    case "CHUNKS_LEXICAL": return "Keyword search...";
    case "CYPHER": return "Cypher query...";
    default: return "Search...";
  }
}
