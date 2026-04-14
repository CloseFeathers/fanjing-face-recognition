import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  RecallMode,
  LLMConfig,
  EmbeddingConfig,
  VectorDBConfig,
  GraphDBConfig,
  ChunkConfig,
  IngestionConfig,
  RetrievalConfig,
} from "@/types";

// ============================================================================
// Search Config State
// ============================================================================

interface SearchConfigState {
  recallMode: RecallMode;
  topK: number;
  onlyContext: boolean;
  useCombinedContext: boolean;
  selectedDatasets: string[];
  systemPrompt: string;
  nodeNames: string[];
  // Advanced search parameters
  wideSearchTopK: number;
  tripletDistancePenalty: number;
  verbose: boolean;
  
  // Actions
  setRecallMode: (mode: RecallMode) => void;
  setTopK: (k: number) => void;
  setOnlyContext: (value: boolean) => void;
  setUseCombinedContext: (value: boolean) => void;
  setSelectedDatasets: (datasets: string[]) => void;
  setSystemPrompt: (prompt: string) => void;
  setNodeNames: (names: string[]) => void;
  addNodeName: (name: string) => void;
  removeNodeName: (name: string) => void;
  setWideSearchTopK: (k: number) => void;
  setTripletDistancePenalty: (penalty: number) => void;
  setVerbose: (value: boolean) => void;
  reset: () => void;
}

const defaultSearchConfig = {
  recallMode: "EPISODIC" as RecallMode,
  topK: 10,
  onlyContext: false,
  useCombinedContext: false,
  selectedDatasets: [] as string[],
  systemPrompt: "",
  nodeNames: [] as string[],
  // Advanced search parameters
  wideSearchTopK: 100,
  tripletDistancePenalty: 3.5,
  verbose: false,
};

export const useSearchConfigStore = create<SearchConfigState>()(
  persist(
    (set) => ({
      ...defaultSearchConfig,
      
      setRecallMode: (mode) => set({ recallMode: mode }),
      setTopK: (k) => set({ topK: k }),
      setOnlyContext: (value) => set({ onlyContext: value }),
      setUseCombinedContext: (value) => set({ useCombinedContext: value }),
      setSelectedDatasets: (datasets) => set({ selectedDatasets: datasets }),
      setSystemPrompt: (prompt) => set({ systemPrompt: prompt }),
      setNodeNames: (names) => set({ nodeNames: names }),
      addNodeName: (name) => set((state) => ({
        nodeNames: state.nodeNames.includes(name) ? state.nodeNames : [...state.nodeNames, name],
      })),
      removeNodeName: (name) => set((state) => ({
        nodeNames: state.nodeNames.filter((n) => n !== name),
      })),
      setWideSearchTopK: (k) => set({ wideSearchTopK: k }),
      setTripletDistancePenalty: (penalty) => set({ tripletDistancePenalty: penalty }),
      setVerbose: (value) => set({ verbose: value }),
      reset: () => set(defaultSearchConfig),
    }),
    {
      name: "mflow-search-config",
    }
  )
);

// ============================================================================
// Retrieval Config State (Advanced Parameters)
// ============================================================================

interface RetrievalConfigState extends RetrievalConfig {
  setConfig: (config: Partial<RetrievalConfig>) => void;
  reset: () => void;
}

const defaultRetrievalConfig: RetrievalConfig = {
  top_k: 10,
  wide_search_top_k: 100,
  edge_miss_cost: 0.9,
  hop_cost: 0.05,
  full_number_match_bonus: 0.12,
  enable_adaptive_weights: true,
  enable_time_bonus: true,
  enable_hybrid_search: true,
  // Episodic output control
  display_mode: "summary",
  max_facets_per_episode: 4,
  max_points_per_facet: 8,
};

export const useRetrievalConfigStore = create<RetrievalConfigState>()(
  persist(
    (set) => ({
      ...defaultRetrievalConfig,
      
      setConfig: (config) => set((state) => ({ ...state, ...config })),
      reset: () => set(defaultRetrievalConfig),
    }),
    {
      name: "mflow-retrieval-config",
    }
  )
);

// ============================================================================
// Ingestion Config State
// ============================================================================

interface IngestionConfigState extends IngestionConfig {
  setConfig: (config: Partial<IngestionConfig>) => void;
  reset: () => void;
}

const defaultIngestionConfig: IngestionConfig = {
  extract_concepts: true,
  extract_summaries: true,
  concept_types: [],
  relationship_types: [],
};

export const useIngestionConfigStore = create<IngestionConfigState>()(
  persist(
    (set) => ({
      ...defaultIngestionConfig,
      
      setConfig: (config) => set((state) => ({ ...state, ...config })),
      reset: () => set(defaultIngestionConfig),
    }),
    {
      name: "mflow-ingestion-config",
    }
  )
);

// ============================================================================
// Chunk Config State
// ============================================================================

interface ChunkConfigState extends ChunkConfig {
  setConfig: (config: Partial<ChunkConfig>) => void;
  reset: () => void;
}

const defaultChunkConfig: ChunkConfig = {
  chunk_strategy: "paragraph",
  chunk_size: 512,
  chunk_overlap: 128,
  chunker_encoding_model: "cl100k_base",
};

export const useChunkConfigStore = create<ChunkConfigState>()(
  persist(
    (set) => ({
      ...defaultChunkConfig,
      
      setConfig: (config) => set((state) => ({ ...state, ...config })),
      reset: () => set(defaultChunkConfig),
    }),
    {
      name: "mflow-chunk-config",
    }
  )
);

// ============================================================================
// Global Settings State (Server-side Configuration)
// ============================================================================

interface GlobalSettingsState {
  llm: LLMConfig | null;
  embedding: EmbeddingConfig | null;
  vectorDb: VectorDBConfig | null;
  graphDb: GraphDBConfig | null;
  isLoading: boolean;
  error: string | null;
  
  setLLM: (config: LLMConfig) => void;
  setEmbedding: (config: EmbeddingConfig) => void;
  setVectorDb: (config: VectorDBConfig) => void;
  setGraphDb: (config: GraphDBConfig) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useGlobalSettingsStore = create<GlobalSettingsState>()((set) => ({
  llm: null,
  embedding: null,
  vectorDb: null,
  graphDb: null,
  isLoading: false,
  error: null,
  
  setLLM: (config) => set({ llm: config }),
  setEmbedding: (config) => set({ embedding: config }),
  setVectorDb: (config) => set({ vectorDb: config }),
  setGraphDb: (config) => set({ graphDb: config }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
}));
