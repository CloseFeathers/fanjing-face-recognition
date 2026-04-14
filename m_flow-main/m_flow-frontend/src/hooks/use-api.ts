"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useAuthStore } from "@/lib/store";
import { STORAGE_KEYS } from "@/lib/config";
import {
  SearchOptions,
  MemorizeRequest,
  LoginRequest,
  RegisterRequest,
  ManualIngestRequest,
  IngestTextRequest,
  IngestUploadOptions,
  DeleteDocumentParams,
  SyncRequest,
  PatchNodeRequest,
  PruneSystemRequest,
  ResponsesRequest,
  SearchOptionsWithCoref,
} from "@/types";

// ============================================================================
// Query Keys
// ============================================================================

export const queryKeys = {
  health: ["health"] as const,
  healthDetailed: ["health", "detailed"] as const,
  user: ["user"] as const,
  datasets: ["datasets"] as const,
  datasetsWithCounts: ["datasets", "with-counts"] as const,
  dataset: (id: string) => ["datasets", id] as const,
  graph: ["graph"] as const,
  // Include datasetId in cache key for proper cache isolation in multi-dataset environments
  episodeGraph: (id: string, datasetId?: string) => 
    datasetId ? ["graph", "episode", id, datasetId] as const : ["graph", "episode", id] as const,
  facetGraph: (id: string, datasetId?: string) => 
    datasetId ? ["graph", "facet", id, datasetId] as const : ["graph", "facet", id] as const,
  // Layer 0: Episodes overview
  episodesOverview: (datasetId?: string) =>
    datasetId ? ["graph", "episodes", datasetId] as const : ["graph", "episodes"] as const,
  // Layer 3: Entity network
  entityNetwork: (entityId: string, datasetId?: string) =>
    datasetId ? ["graph", "entity", entityId, datasetId] as const : ["graph", "entity", entityId] as const,
  settings: ["settings"] as const,
  prompts: ["prompts"] as const,
  prompt: (filename: string) => ["prompts", filename] as const,
  // User management (keep existing 'user' for useCurrentUser)
  users: ["users"] as const,
  userById: (id: string) => ["users", id] as const,
};

// ============================================================================
// Health Check
// ============================================================================

/**
 * Simple Health Check Hook
 * 
 * @description Uses /health endpoint to check if backend is alive
 * @returns Query result with HealthCheckResponse
 * 
 * @example
 * const { data: health, isLoading, error } = useHealthCheck();
 * if (health?.status === 'healthy') {
 *   console.log('Backend is running');
 * }
 */
export function useHealthCheck() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 60000, // refresh every 60 seconds
  });
}

// ============================================================================
// Detailed Health Check
// ============================================================================

/**
 * Detailed Health Check Hook configuration options
 */
export interface UseDetailedHealthOptions {
  /**
   * Auto refresh interval (milliseconds)
   * @default 30000 (30 seconds)
   * @description Set to false or 0 to disable auto refresh
   */
  refetchInterval?: number | false;
  
  /**
   * Whether to enable query
   * @default true
   * @description Can be used to conditionally disable query
   */
  enabled?: boolean;
  
  /**
   * Data stale time (milliseconds)
   * @default 10000 (10 seconds)
   * @description Within this time, repeated requests will use cached data
   */
  staleTime?: number;
}

/**
 * Detailed Health Check Hook
 * 
 * @description Uses /health/detailed endpoint to get detailed health status of all system components
 * @param options Configuration options
 * @returns Query result with DetailedHealthResponse
 * 
 * @example
 * // Basic usage
 * const { data, isLoading, error, refetch } = useDetailedHealth();
 * 
 * @example
 * // Custom refresh interval
 * const { data } = useDetailedHealth({ refetchInterval: 10000 }); // 10 seconds
 * 
 * @example
 * // Disable auto refresh, manual control
 * const { data, refetch } = useDetailedHealth({ refetchInterval: false });
 * // Manual refresh
 * await refetch();
 * 
 * @example
 * // Access specific component status
 * const { data } = useDetailedHealth();
 * if (data) {
 *   // Check LLM status
 *   const llmStatus = data.probes.llm_provider;
 *   console.log(`LLM: ${llmStatus.verdict} (${llmStatus.backend})`);
 *   
 *   // Check if any critical components are down
 *   const criticalDown = ['relational_db', 'vector_db', 'graph_db', 'file_storage']
 *     .some(key => data.probes[key].verdict === 'down');
 *   
 *   // Get total latency
 *   const totalLatency = Object.values(data.probes)
 *     .reduce((sum, probe) => sum + probe.latency_ms, 0);
 * }
 */
export function useDetailedHealth(options: UseDetailedHealthOptions = {}) {
  const {
    refetchInterval = 60000,
    enabled = true,
    staleTime = 30000,
  } = options;
  
  return useQuery({
    queryKey: queryKeys.healthDetailed,
    queryFn: () => apiClient.healthCheckDetailed(),
    refetchInterval,
    enabled,
    staleTime,
    // Error retry configuration
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
  });
}

// ============================================================================
// Authentication
// ============================================================================

export function useCurrentUser() {
  return useQuery({
    queryKey: queryKeys.user,
    queryFn: () => apiClient.getCurrentUser(),
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

export function useLogin() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (credentials: LoginRequest) => apiClient.login(credentials),
    onSuccess: (user) => {
      // Sync update React Query cache
      queryClient.setQueryData(queryKeys.user, user);
      // Sync update Zustand auth store to ensure UI state consistency
      useAuthStore.getState().setUser(user);
    },
  });
}

export function useRegister() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data: RegisterRequest) => apiClient.register(data),
    onSuccess: (user) => {
      queryClient.setQueryData(queryKeys.user, user);
    },
  });
}

export function useLogout() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: () => apiClient.logout(),
    onSuccess: () => {
      // Set flag to prevent auto-login after explicit logout
      if (typeof window !== "undefined") {
        sessionStorage.setItem(STORAGE_KEYS.LOGOUT_FLAG, "true");
      }
      // Sync update React Query cache
      queryClient.setQueryData(queryKeys.user, null);
      queryClient.invalidateQueries();
      // Sync update Zustand auth store to ensure UI state consistency
      useAuthStore.getState().logout();
    },
  });
}

// ============================================================================
// Users (Admin)
// ============================================================================

/**
 * Get all users (superuser only)
 *
 * @endpoint GET /api/v1/users
 */
export function useUsers() {
  const { user } = useAuthStore();

  return useQuery({
    queryKey: queryKeys.users,
    queryFn: () => apiClient.getUsers(),
    // Fetch if user is superuser, or if no auth (user is null, allow by default)
    enabled: user === null || !!user?.is_superuser,
  });
}

/**
 * Get user by ID
 *
 * @endpoint GET /api/v1/users/{id}
 */
export function useUser(id: string) {
  return useQuery({
    queryKey: queryKeys.userById(id),
    queryFn: () => apiClient.getUser(id),
    enabled: !!id,
  });
}

/**
 * Create user via public registration endpoint
 *
 * @endpoint POST /api/v1/auth/register
 */
export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: RegisterRequest) => apiClient.createUser(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
    },
  });
}

/**
 * Delete user (superuser only)
 *
 * @endpoint DELETE /api/v1/users/{id}
 */
export function useDeleteUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => apiClient.deleteUser(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
    },
  });
}

// ============================================================================
// Datasets
// ============================================================================

export function useDatasets() {
  return useQuery({
    queryKey: queryKeys.datasets,
    queryFn: () => apiClient.getDatasets(),
  });
}

export function useDatasetsWithCounts(options?: { refetchInterval?: number | false }) {
  return useQuery({
    queryKey: queryKeys.datasetsWithCounts,
    queryFn: () => apiClient.getDatasetsWithCounts(),
    refetchInterval: options?.refetchInterval ?? 5000, // 5 seconds default
  });
}

export function useDataset(id: string) {
  return useQuery({
    queryKey: queryKeys.dataset(id),
    queryFn: () => apiClient.getDataset(id),
    enabled: !!id,
  });
}

export function useDataItems(datasetId: string | null) {
  return useQuery({
    queryKey: ["dataItems", datasetId] as const,
    queryFn: () => apiClient.getDataItems(datasetId!),
    enabled: !!datasetId,
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteDataset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
    },
  });
}

export function useCreateDataset() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (name: string) => apiClient.createDataset(name),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      await queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
    },
  });
}

// ============================================================================
// Data Addition and Ingestion
// ============================================================================

/**
 * Add Text Content Hook
 * 
 * @endpoint POST /api/v1/add
 */
export function useAddTextMemory() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ text, datasetName, datasetId, nodeSet, incrementalLoading }: { 
      text: string; 
      datasetName?: string; 
      datasetId?: string;
      nodeSet?: string[];
      incrementalLoading?: boolean;
    }) => apiClient.addTextMemory(text, datasetName, datasetId, nodeSet, incrementalLoading),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
    },
  });
}

/**
 * Upload File Hook
 * 
 * @endpoint POST /api/v1/add
 */
export function useUploadFile() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ file, datasetName, datasetId, nodeSet, incrementalLoading }: { 
      file: File; 
      datasetName?: string; 
      datasetId?: string;
      nodeSet?: string[];
      incrementalLoading?: boolean;
    }) => apiClient.uploadFile(file, datasetName, datasetId, nodeSet, incrementalLoading),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
    },
  });
}

export function useMemorize() {
  return useMutation({
    mutationFn: (request: MemorizeRequest) => apiClient.memorize(request),
  });
}

export function useManualIngest() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (request: ManualIngestRequest) => apiClient.manualIngest(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.graph });
    },
  });
}

// ============================================================================
// Unified Ingestion (Ingest = Add + Memorize)
// ============================================================================

/**
 * Unified Ingestion Hook (Text Content)
 * 
 * @endpoint POST /api/v1/ingest
 * @description Complete data ingestion and knowledge graph construction in one step
 * 
 * @example
 * const ingestText = useIngestText();
 * await ingestText.mutateAsync({
 *   content: "Document content...",
 *   dataset_name: "my_docs",
 *   enable_episode_routing: false,
 * });
 */
export function useIngestText() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (request: IngestTextRequest) => apiClient.ingestText(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
      queryClient.invalidateQueries({ queryKey: queryKeys.graph });
    },
  });
}

/**
 * Unified Ingestion Hook (Single File Upload)
 * 
 * @endpoint POST /api/v1/ingest/upload
 * @description Upload a single file and complete ingestion and knowledge graph construction in one step
 */
export function useIngestFile() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ file, options }: { file: File; options?: IngestUploadOptions }) =>
      apiClient.ingestFile(file, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
      queryClient.invalidateQueries({ queryKey: queryKeys.graph });
    },
  });
}

/**
 * Unified Ingestion Hook (Multiple File Upload)
 * 
 * @endpoint POST /api/v1/ingest/upload
 * @description Upload multiple files and complete ingestion and knowledge graph construction in one step
 */
export function useIngestFiles() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ files, options }: { files: File[]; options?: IngestUploadOptions }) =>
      apiClient.ingestFiles(files, options),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
      queryClient.invalidateQueries({ queryKey: queryKeys.graph });
    },
  });
}

// ============================================================================
// Search
// ============================================================================

export function useSearch() {
  return useMutation({
    mutationFn: (options: SearchOptions) => apiClient.search(options),
  });
}

/**
 * Get Search History Hook
 * 
 * @endpoint GET /api/v1/search
 * @description Get current user's search history, sorted by time descending
 */
export function useSearchHistory() {
  return useQuery({
    queryKey: ["searchHistory"] as const,
    queryFn: () => apiClient.getSearchHistory(),
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Get Recent Activities Hook
 * 
 * @endpoint GET /api/v1/activity
 * @description Aggregated activity feed for dashboard from multiple sources
 * 
 * Note: Activity feed is non-critical - gracefully returns empty on errors
 */
export function useActivities(limit: number = 20, options?: { refetchInterval?: number | false }) {
  return useQuery({
    queryKey: ["activities", limit] as const,
    queryFn: async () => {
      try {
        return await apiClient.getActivities(limit);
      } catch {
        // Activity feed is non-critical, return empty array on error
        // to avoid showing error state for empty/unavailable activities
        return [];
      }
    },
    staleTime: 30 * 1000, // 30 seconds - activities update frequently
    refetchInterval: options?.refetchInterval ?? 3000, // 3 seconds default for real-time updates
    retry: false, // Don't retry - just show empty state
  });
}

// ============================================================================
// Knowledge Graph
// ============================================================================

export function useGraph(datasetId?: string) {
  return useQuery({
    queryKey: datasetId ? [...queryKeys.graph, datasetId] : queryKeys.graph,
    queryFn: () => apiClient.getGraph(datasetId),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to fetch episode subgraph data.
 * @param datasetId - Optional dataset ID for cache isolation in multi-dataset environments
 */
export function useEpisodeGraph(episodeId: string, datasetId?: string) {
  return useQuery({
    queryKey: queryKeys.episodeGraph(episodeId, datasetId),
    queryFn: () => apiClient.getEpisodeSubgraph(episodeId, datasetId),
    enabled: !!episodeId,
  });
}

/**
 * Hook to fetch facet subgraph data.
 * @param datasetId - Optional dataset ID for cache isolation in multi-dataset environments
 */
export function useFacetGraph(facetId: string, datasetId?: string) {
  return useQuery({
    queryKey: queryKeys.facetGraph(facetId, datasetId),
    queryFn: () => apiClient.getFacetSubgraph(facetId, datasetId),
    enabled: !!facetId,
  });
}

/**
 * Hook to fetch episodes overview for Layer 0 navigation.
 */
export function useEpisodesOverview(options?: {
  datasetId?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: queryKeys.episodesOverview(options?.datasetId),
    queryFn: () => apiClient.getEpisodesOverview(options),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to fetch entity network for Layer 3 navigation.
 */
export function useEntityNetwork(entityId: string, datasetId?: string) {
  return useQuery({
    queryKey: queryKeys.entityNetwork(entityId, datasetId),
    queryFn: () => apiClient.getEntityNetwork(entityId, datasetId),
    enabled: !!entityId,
  });
}

// ============================================================================
// Settings
// ============================================================================

/**
 * Get System Settings Hook
 * 
 * @description Get server-side stored configuration
 * @returns Query result with Settings
 * 
 * @note Backend only returns partial configuration:
 *   - llm: LLM provider configuration
 *   - vector_db: Vector database configuration
 * Other configuration (embedding, graph_db, etc.) is controlled by environment variables and may not be in response
 */
export function useSettings() {
  return useQuery({
    queryKey: queryKeys.settings,
    queryFn: () => apiClient.getSettings(),
  });
}

/**
 * Update System Settings Hook
 * 
 * @description Update server-side configuration
 * @returns Mutation hook for updating settings
 * 
 * @warning Backend only supports updating the following fields:
 *   - llm: { provider, model, api_key }
 *   - vector_db: { provider, url, api_key }
 * 
 * Other fields (embedding, graph_db, storage, security) will be ignored
 * 
 * @example
 * const updateSettings = useUpdateSettings();
 * 
 * // Update LLM configuration
 * await updateSettings.mutateAsync({
 *   llm: {
 *     llm_provider: 'openai',
 *     llm_model: 'gpt-4o-mini',
 *   }
 * });
 */
export function useUpdateSettings() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (settings: Parameters<typeof apiClient.updateSettings>[0]) =>
      apiClient.updateSettings(settings),
    onSuccess: () => {
      // After updating settings, also refresh settings and health status
      queryClient.invalidateQueries({ queryKey: queryKeys.settings });
      queryClient.invalidateQueries({ queryKey: queryKeys.healthDetailed });
    },
  });
}

// ============================================================================
// Prompts Management
// ============================================================================

/**
 * Get All Prompts Hook
 * 
 * @endpoint GET /api/v1/prompts
 * @description Get all LLM prompt templates, grouped by category
 */
export function usePrompts() {
  return useQuery({
    queryKey: queryKeys.prompts,
    queryFn: () => apiClient.getPrompts(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Get Single Prompt Hook
 * 
 * @endpoint GET /api/v1/prompts/{filename}
 */
export function usePrompt(filename: string) {
  return useQuery({
    queryKey: queryKeys.prompt(filename),
    queryFn: () => apiClient.getPrompt(filename),
    enabled: !!filename,
  });
}

/**
 * Update Prompt Hook
 * 
 * @endpoint PUT /api/v1/prompts/{filename}
 */
export function useUpdatePrompt() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ filename, content }: { filename: string; content: string }) =>
      apiClient.updatePrompt(filename, content),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.prompts });
      queryClient.invalidateQueries({ queryKey: queryKeys.prompt(variables.filename) });
    },
  });
}

/**
 * Reset Prompt Hook
 * 
 * @endpoint POST /api/v1/prompts/{filename}/reset
 */
export function useResetPrompt() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (filename: string) => apiClient.resetPrompt(filename),
    onSuccess: (_, filename) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.prompts });
      queryClient.invalidateQueries({ queryKey: queryKeys.prompt(filename) });
    },
  });
}

/**
 * Reset All Prompts Hook
 * 
 * @endpoint POST /api/v1/prompts/reset-all
 */
export function useResetAllPrompts() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: () => apiClient.resetAllPrompts(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.prompts });
    },
  });
}

// ============================================================================
// Procedural Memory Extraction Hooks
// ============================================================================

/**
 * Extract Procedural Memories Hook
 * 
 * @endpoint POST /api/v1/procedural/extract-from-episodic
 * @description Extract procedural memories from existing episodic memories
 */
export function useExtractProcedural() {
  return useMutation({
    mutationFn: (options: {
      dataset_id?: string;
      episode_ids?: string[];
      limit?: number;
      force_reprocess?: boolean;
      dry_run?: boolean;
    } = {}) => apiClient.extractProcedural(options),
  });
}

// ============================================================================
// Document Deletion
// ============================================================================

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: DeleteDocumentParams) => apiClient.deleteDocument(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets });
      queryClient.invalidateQueries({ queryKey: queryKeys.datasetsWithCounts });
      queryClient.invalidateQueries({ queryKey: queryKeys.graph });
    },
  });
}

// ============================================================================
// Data Sync
// ============================================================================

export function useSyncData() {
  return useMutation({
    mutationFn: (request: SyncRequest) => apiClient.syncData(request),
  });
}

export function useSyncStatus() {
  return useQuery({
    queryKey: ["syncStatus"] as const,
    queryFn: () => apiClient.getSyncStatus(),
    refetchInterval: 5000,
  });
}

// ============================================================================
// Dataset Status
// ============================================================================

export function useDatasetsStatus(datasetIds?: string[]) {
  return useQuery({
    queryKey: ["datasetsStatus", datasetIds] as const,
    queryFn: () => apiClient.getDatasetsStatus(datasetIds),
    refetchInterval: 10000,
  });
}

// ============================================================================
// Manual Node Editing
// ============================================================================

export function usePatchNode() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: PatchNodeRequest) => apiClient.patchNode(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.graph });
    },
  });
}

export function useManualSchema() {
  return useQuery({
    queryKey: ["manualSchema"] as const,
    queryFn: () => apiClient.getManualSchema(),
    staleTime: 30 * 60 * 1000,
  });
}

// ============================================================================
// Granular Prune
// ============================================================================

export function usePruneData() {
  return useMutation({
    mutationFn: () => apiClient.pruneData(),
  });
}

export function usePruneSystem() {
  return useMutation({
    mutationFn: (options?: Partial<Pick<PruneSystemRequest, "graph" | "vector" | "metadata" | "cache">>) =>
      apiClient.pruneSystem(options),
  });
}

export function usePruneAll() {
  return useMutation({
    mutationFn: () => apiClient.pruneAll(),
  });
}

// ============================================================================
// Auth Extended
// ============================================================================

export function useForgotPassword() {
  return useMutation({
    mutationFn: (email: string) => apiClient.forgotPassword(email),
  });
}

export function useResetPassword() {
  return useMutation({
    mutationFn: ({ token, password }: { token: string; password: string }) =>
      apiClient.resetPassword(token, password),
  });
}

export function useRequestVerifyToken() {
  return useMutation({
    mutationFn: (email: string) => apiClient.requestVerifyToken(email),
  });
}

export function useVerifyEmail() {
  return useMutation({
    mutationFn: (token: string) => apiClient.verifyEmail(token),
  });
}

// ============================================================================
// Responses API
// ============================================================================

export function useCreateResponse() {
  return useMutation({
    mutationFn: (request: ResponsesRequest) => apiClient.createResponse(request),
  });
}

// ============================================================================
// Search with Coreference
// ============================================================================

export function useSearchWithCoref() {
  return useMutation({
    mutationFn: (options: SearchOptionsWithCoref) => apiClient.searchWithCoref(options),
  });
}

