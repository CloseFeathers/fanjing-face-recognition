// ============================================================================
// M-Flow Frontend Type Definitions
// ============================================================================

/**
 * Search/Recall mode - corresponds to backend RecallMode
 */
export type RecallMode =
  | "EPISODIC"           // Episodic memory search
  | "PROCEDURAL"         // Procedural memory search
  | "TRIPLET_COMPLETION" // Triplet completion search
  | "CHUNKS_LEXICAL"     // Lexical chunk search
  | "CYPHER";            // Cypher query

/**
 * Search parameters configuration
 */
export interface SearchOptions {
  recall_mode: RecallMode;
  query: string;
  datasets?: string[];
  dataset_ids?: string[];
  system_prompt?: string;
  node_name?: string[];
  top_k?: number;
  only_context?: boolean;
  use_combined_context?: boolean;
  // Advanced search parameters
  wide_search_top_k?: number;
  triplet_distance_penalty?: number;
  verbose?: boolean;
  // Episodic retrieval parameters (Phase 0.4)
  enable_hybrid_search?: boolean;
  enable_time_bonus?: boolean;
  edge_miss_cost?: number;
  hop_cost?: number;
  full_number_match_bonus?: number;
  enable_adaptive_weights?: boolean;
  // Episodic output control
  display_mode?: "summary" | "detail";
  max_facets_per_episode?: number;
  max_points_per_facet?: number;
  keyword_match_bonus?: number;
  direct_episode_penalty?: number;
  // Collection filtering
  collections?: string[];
  // Procedural inclusion for TRIPLET mode
  // Coreference parameters
  coref_enabled?: boolean;
  coref_session_id?: string;
  coref_new_turn?: boolean;
}

/**
 * Search result item
 */
export interface SearchResultItem {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
  node_type?: string;
  source_dataset?: string;
  // Triplet result specific
  subject?: string;
  predicate?: string;
  object?: string;
  // Episodic memory result specific
  episode_id?: string;
  timestamp?: string;
}

/**
 * Search response
 */
export interface SearchResponse {
  results: SearchResultItem[];
  total: number;
  query: string;
  recall_mode: RecallMode;
  processing_time_ms?: number;
}

/**
 * Search history entry
 * 
 * @endpoint GET /api/v1/search
 * @note API returns camelCase field names due to OutDTO alias_generator
 */
export interface SearchHistoryEntry {
  id: string;
  text: string;
  user: string;
  createdAt: string;
}

/**
 * Activity item for dashboard activity feed
 * 
 * @endpoint GET /api/v1/activity
 */
export interface ActivityItem {
  id: string;
  type: "search" | "ingest" | "create" | "delete" | "config";
  title: string;
  description?: string;
  status: "success" | "error" | "pending";
  createdAt: string;
}

/**
 * Active pipeline run for real-time progress tracking
 * 
 * @endpoint GET /api/v1/pipeline/active
 */
export interface ActivePipeline {
  workflow_run_id: string;
  dataset_id: string | null;
  dataset_name: string | null;
  workflow_name: string;
  status: string;
  total_items: number | null;
  processed_items: number | null;
  current_step: string | null;
  started_at: string | null;
  updated_at: string | null;
  created_at: string | null;
  // Convenience aliases
  pipelineRunId?: string;
  datasetId?: string | null;
  datasetName?: string | null;
  pipelineName?: string;
  totalItems?: number | null;
  processedItems?: number | null;
  currentStep?: string | null;
  startedAt?: string | null;
  updatedAt?: string | null;
  createdAt?: string | null;
}

/**
 * Dataset
 * 
 * @note Backend returns camelCase fields via OutDTO alias_generator
 */
export interface Dataset {
  id: string;
  name: string;
  description?: string;
  createdAt?: string;
  updatedAt?: string;
  status?: "ready" | "processing" | "error";
}

/**
 * Data item within a dataset
 */
export interface DataItemInfo {
  id: string;
  name: string;
  extension: string;
  mimeType: string;
  rawDataLocation: string;
  datasetId: string;
  createdAt?: string;
  updatedAt?: string;
  dataSize?: number | null;
  tokenCount?: number | null;
  pipelineStatus?: Record<string, Record<string, string>> | null;
}

/**
 * Dataset with counts
 * 
 * @endpoint GET /api/v1/datasets (with include_counts=true)
 * @note Backend DatasetWithCountsDTO returns: dataCount, ownerId (camelCase via OutDTO)
 */
export interface DatasetWithCounts extends Dataset {
  dataCount?: number;
  ownerId?: string;
}

/**
 * Knowledge graph node
 */
export interface GraphNode {
  id: string;
  name: string;
  label?: string;
  type: string;
  description?: string;
  summary?: string;
  properties?: Record<string, unknown>;
  color?: string;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
  dataset_id?: string;  // For multi-dataset support
}

/**
 * Knowledge graph edge
 */
export interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  relationship: string;
  properties?: Record<string, unknown>;
}

/**
 * Alias for backward compatibility
 * @deprecated Use GraphEdge instead
 */
export type GraphLink = GraphEdge;

/**
 * Graph data structure
 */
export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/**
 * Episode overview for Layer 0 navigation
 */
export interface EpisodeOverview {
  id: string;
  name: string;
  summary?: string;
  facetCount: number;
  entityCount: number;
  createdAt?: string;
  datasetId?: string;
}

/**
 * Episodes overview API response
 */
export interface EpisodesOverviewResponse {
  episodes: EpisodeOverview[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Entity network node for Layer 3
 */
export interface EntityNetworkNode {
  id: string;
  name: string;
  type: string;
  relationship: string;
  datasetId?: string;
}

/**
 * Entity network API response
 */
export interface EntityNetworkResponse {
  entityId: string;
  entityName: string;
  entityType: string;
  connectedEpisodes: EntityNetworkNode[];
  connectedFacets: EntityNetworkNode[];
  sameEntities: EntityNetworkNode[];
  datasetId?: string;
}

/**
 * Navigation layer type for hierarchical graph
 */
export type NavigationLayer = "overview" | "episode" | "facet" | "entity" | "procedure";

/**
 * Navigation state for hierarchical graph
 */
export interface NavigationState {
  layer: NavigationLayer;
  episodeId?: string;
  episodeName?: string;
  facetId?: string;
  facetName?: string;
  entityId?: string;
  entityName?: string;
  procedureId?: string;
  procedureName?: string;
  datasetId?: string;
}

/**
 * User authentication info
 */
export interface AuthUser {
  id: string;
  email: string;
  username?: string;
  is_superuser?: boolean;
  tenant_id?: string;
}

/**
 * User list item (from GET /api/v1/users)
 *
 * @note Backend UserRead uses snake_case (fastapi-users standard, NOT OutDTO)
 * @endpoint GET /api/v1/users
 */
export interface UserListItem {
  id: string;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  is_verified: boolean;
  tenant_id?: string;
}

/**
 * Login request
 */
export interface LoginRequest {
  username: string;
  password: string;
}

/**
 * Register request
 */
export interface RegisterRequest {
  email: string;
  password: string;
  username?: string;
}

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * LLM configuration (GET response format)
 * 
 * This is the llm section from GET /api/v1/settings response
 * Note: Backend fields use llm_ prefix (inherited from LLMConfig pydantic model)
 */
export interface LLMConfig {
  llm_provider: string;       // Provider: openai, anthropic, ollama, gemini, mistral, bedrock, custom
  llm_model: string;          // Model name
  llm_endpoint?: string;      // Custom endpoint URL
  llm_api_key?: string;       // API key (partially masked, may not be returned for security)
  llm_temperature?: number;   // Temperature
  llm_max_tokens?: number;    // Max tokens
  llm_streaming?: boolean;    // Enable streaming output
  llm_rate_limit_requests?: number;       // Rate limit: requests/minute
  llm_rate_limit_input_tokens?: number;   // Rate limit: input tokens/minute
  llm_rate_limit_output_tokens?: number;  // Rate limit: output tokens/minute
  structured_output_framework?: string;   // Structured output framework
}

/**
 * LLM configuration update request
 * 
 * This is the llm section accepted by POST /api/v1/settings
 * Note: Uses short field names (provider, model, api_key)
 * Not llm_provider, llm_model etc. from GET response
 */
export interface LLMSettingsUpdate {
  provider: string;       // Not llm_provider
  model: string;          // Not llm_model
  api_key?: string;       // Not llm_api_key (optional, empty means no update)
}

/**
 * Embedding configuration
 */
export interface EmbeddingConfig {
  embedding_provider: string;
  embedding_model: string;
  embedding_endpoint?: string;
  embedding_api_key?: string;
  embedding_dimensions?: number;
}

/**
 * Vector database configuration (runtime read format)
 * 
 * This is the vector_db section from GET /api/v1/settings response
 */
export interface VectorDBConfig {
  provider: string;       // Provider: lancedb, pgvector, chromadb
  url: string;            // Connection URL
  api_key: string;        // API key (partially masked)
  providers: Array<{ value: string; label: string }>;   // Available providers list
}

/**
 * Vector database configuration update request
 * 
 * This is the vector_db section accepted by POST /api/v1/settings
 */
export interface VectorDBSettingsUpdate {
  provider: string;       // lancedb, pgvector, chromadb
  url: string;            // Connection URL
  api_key: string;        // API key
}

/**
 * Vector database configuration (full fields, for internal env vars)
 * 
 * Corresponds to backend VectorConfig Pydantic model (loaded from .env)
 */
export interface VectorDBEnvConfig {
  vector_db_provider: string;
  vector_db_url?: string;
  vector_db_port?: number;
  vector_db_name?: string;
  vector_db_key?: string;
}

/**
 * Graph database configuration
 */
export interface GraphDBConfig {
  graph_db_provider: string;
  graph_db_url?: string;
  graph_db_name?: string;
  graph_db_username?: string;
  graph_db_password?: string;
}

/**
 * Chunking configuration
 */
export interface ChunkConfig {
  chunk_strategy: "paragraph" | "sentence" | "word" | "fixed";
  chunk_size: number;
  chunk_overlap: number;
  chunker_encoding_model?: string;
}

/**
 * Ingestion configuration
 */
export interface IngestionConfig {
  extract_concepts?: boolean;
  extract_summaries?: boolean;
  concept_types?: string[];
  relationship_types?: string[];
}

/**
 * Retrieval configuration
 */
export interface RetrievalConfig {
  top_k: number;
  wide_search_top_k?: number;
  edge_miss_cost?: number;
  hop_cost?: number;
  full_number_match_bonus?: number;
  enable_adaptive_weights?: boolean;
  enable_time_bonus?: boolean;
  enable_hybrid_search?: boolean;
  // Episodic output control
  display_mode?: "summary" | "detail";
  max_facets_per_episode?: number;
  max_points_per_facet?: number;
  // Collection control (TRIPLET / EPISODIC)
  collections?: string[];
}

/**
 * Global settings (from backend API)
 * 
 * Note: Backend GET /api/v1/settings only returns llm and vector_db
 * Other configs (embedding, graph_db, storage) are managed via .env file
 */
export interface Settings {
  llm: LLMConfig;
  vector_db: VectorDBConfig;
  // Note: Backend GET /settings currently doesn't return embedding config
  // Embedding config is only controlled via env vars
  // This field is optional for future backend extension compatibility
  embedding?: EmbeddingConfig;
}

/**
 * Embedding configuration update request
 * 
 * Used for POST /api/v1/settings embedding section
 */
export interface EmbeddingSettingsUpdate {
  provider: string;       // openai, ollama, fastembed, azure
  model: string;          // Model name
  dimensions?: number;    // Vector dimensions
  endpoint?: string;      // Custom endpoint
  api_key?: string;       // API key (optional, empty means no update)
}

/**
 * Settings update request
 * 
 * Used for POST /api/v1/settings
 */
export interface SettingsUpdateRequest {
  llm?: LLMSettingsUpdate;
  vector_db?: VectorDBSettingsUpdate;
  embedding?: EmbeddingSettingsUpdate;
}

/**
 * Session-level configuration (stored in localStorage, passed as API params)
 */
export interface SessionSettings {
  embedding?: EmbeddingConfig;
  chunk?: ChunkConfig;
  ingestion?: IngestionConfig;
  retrieval?: RetrievalConfig;
}

// ============================================================================
// Memorize (Ingestion) Related Types
// ============================================================================

/**
 * Chunking strategy
 */
export type ChunkerType = "TextChunker" | "LangchainChunker";

/**
 * Concurrency conflict handling mode
 */
export type ConflictMode = "warn" | "error" | "ignore";

export interface MemorizeRequest {
  datasets?: string[];
  dataset_ids?: string[];
  run_in_background?: boolean;
  chunk_size?: number;
  chunker?: ChunkerType;
  chunks_per_batch?: number;
  items_per_batch?: number;
  incremental_loading?: boolean;
  conflict_mode?: ConflictMode;
  // Feature toggles (override environment variables)
  enable_episode_routing?: boolean;
  enable_content_routing?: boolean;
  enable_procedural?: boolean;
  enable_semantic_merge?: boolean;
  precise_mode?: boolean;
  enable_facet_points?: boolean;
}

// ============================================================================
// Unified Ingestion Related Types
// ============================================================================

/**
 * Ingestion status enum
 */
export type IngestStatus =
  | "completed"          // Sync completed, data queryable
  | "background_started" // Background processing started
  | "memorize_skipped"   // Only add executed, data not queryable
  | "memorize_failed";   // Add succeeded but memorize failed

/**
 * Unified ingestion request (text content)
 *
 * @endpoint POST /api/v1/ingest
 */
export interface IngestTextRequest {
  content: string;
  dataset_name?: string;
  graph_scope?: string[];
  skip_memorize?: boolean;
  run_in_background?: boolean;
  chunk_size?: number;
  chunks_per_batch?: number;
  // Feature toggles
  enable_episode_routing?: boolean;
  enable_content_routing?: boolean;
  enable_procedural?: boolean;
  enable_facet_points?: boolean;
  precise_mode?: boolean;
  content_type?: string;
  conflict_mode?: ConflictMode;
}

/**
 * Unified ingestion request (file upload)
 *
 * @endpoint POST /api/v1/ingest/upload
 */
export interface IngestUploadOptions {
  datasetName?: string;
  datasetId?: string;
  nodeSet?: string[];
  skipMemorize?: boolean;
  runInBackground?: boolean;
  customPrompt?: string;
  chunkSize?: number;
  chunksPerBatch?: number;
  // Feature toggles
  enableEpisodeRouting?: boolean;
  enableContentRouting?: boolean;
  enableProcedural?: boolean;
  enableFacetPoints?: boolean;
  preciseMode?: boolean;
  conflictMode?: ConflictMode;
}

/**
 * Ingestion response
 *
 * @endpoint POST /api/v1/ingest
 * @endpoint POST /api/v1/ingest/upload
 */
export interface IngestResponse {
  dataset_id: string;
  dataset_name: string;
  status: IngestStatus;
  add_run_id: string;
  memorize_run_id?: string;
  error_message?: string;
}

/**
 * Pipeline run status - matches backend RunStatus enum
 * 
 * @see m_flow/pipeline/models/RunEvent.py
 */
export type RunStatus = 
  | "RunStarted"        // Started
  | "RunYield"          // Intermediate result
  | "RunCompleted"      // Completed
  | "RunAlreadyCompleted"  // Already completed (cached)
  | "RunFailed";       // Error

/**
 * Pipeline run info - matches backend RunEvent model
 * 
 * @see m_flow/pipeline/models/RunEvent.py
 */
export interface RunEvent {
  status: RunStatus;
  workflow_run_id: string;
  dataset_id: string;
  dataset_name: string;
  payload?: unknown;
  processing_results?: unknown[];
}

/**
 * Memorize API response
 * 
 * Backend returns dict[dataset_id -> RunEvent]
 * @endpoint POST /api/v1/memorize
 */
export type MemorizeResponse = Record<string, RunEvent>;

/**
 * WebSocket progress message
 * 
 * @endpoint WS /api/v1/memorize/subscribe/{workflow_run_id}
 */
export interface WebSocketProgress {
  workflow_run_id: string;
  status: RunStatus;
  payload?: unknown;
}

/**
 * @deprecated Use RunEvent instead
 * Kept for backward compatibility, will be removed in future versions
 */
export interface MemorizeProgress {
  task_id: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  current_step?: string;
  error?: string;
}

// ============================================================================
// Manual Ingestion Types
// ============================================================================

/**
 * FacetPoint input - fine-grained information point
 */
export interface ManualFacetPointInput {
  search_text: string;
  aliases?: string[];
  description?: string;
  display_only?: string;
}

/**
 * Facet input - detailed anchor point for Episode
 */
export interface ManualFacetInput {
  facet_type?: string;
  search_text: string;
  aliases?: string[];
  description?: string;
  anchor_text?: string;
  display_only?: string;
  points?: ManualFacetPointInput[];
}

/**
 * Entity/Entity input
 */
export interface ManualConceptInput {
  name: string;
  description: string;
  canonical_name?: string;
  entity_type?: string;
  display_only?: string;
}

/**
 * Episode input - coarse-grained memory anchor
 */
export interface ManualEpisodeInput {
  name: string;
  summary: string;
  signature?: string;
  status?: string;
  memory_type?: string;
  display_only?: string;
  facets?: ManualFacetInput[];
  entities?: ManualConceptInput[];
}

/**
 * Manual ingestion request
 * 
 * @endpoint POST /api/v1/manual/ingest
 */
export interface ManualIngestRequest {
  episodes: ManualEpisodeInput[];
  dataset_name?: string;
  dataset_id?: string;
  embed_triplets?: boolean;
}

/**
 * Manual ingestion result
 */
export interface ManualIngestResult {
  success: boolean;
  episodes_created: number;
  facets_created: number;
  facet_points_created: number;
  entities_created: number;
  errors?: string[];
}

// ============================================================================
// Prompts Management Types
// ============================================================================

/**
 * Prompt template
 * 
 * @endpoint GET /api/v1/prompts/{filename}
 */
export interface Prompt {
  name: string;
  filename: string;
  category: string;
  content: string;
  description?: string;
  is_modified: boolean;
}

/**
 * Prompt list item (without full content)
 */
export interface PromptListItem {
  name: string;
  filename: string;
  category: string;
  description?: string;
  is_modified: boolean;
}

/**
 * Prompt category (grouped by category)
 * 
 * @endpoint GET /api/v1/prompts
 */
export interface PromptCategory {
  category: string;
  prompts: PromptListItem[];
}

// ============================================================================
// System Related Types
// ============================================================================

/**
 * Health check response (simple version, for /health endpoint)
 * 
 * @endpoint GET /health
 * @example
 * {
 *   "status": "ready",
 *   "health": "up",
 *   "version": "0.5.1"
 * }
 */
export interface HealthCheckResponse {
  status: "healthy" | "unhealthy";
  version?: string;
  services?: {
    database: boolean;
    vector_db: boolean;
    graph_db: boolean;
    llm: boolean;
  };
}

// ============================================================================
// Detailed Health Check Types (for /health/detailed endpoint)
// ============================================================================

/**
 * Health probe status enum
 * 
 * Corresponds to backend Verdict enum:
 * - "up"   → Service running normally
 * - "warn" → Service available but with warnings (non-critical service failed)
 * - "down" → Service unavailable
 */
export type HealthVerdict = "up" | "warn" | "down";

/**
 * Single component probe result
 * 
 * @field verdict    - Probe result status
 * @field backend    - Backend provider name (e.g. "openai", "sqlite", "kuzu")
 * @field latency_ms - Probe latency (milliseconds)
 * @field note       - Additional notes (e.g. "responding", "session ok", error messages)
 * 
 * @example
 * {
 *   "verdict": "up",
 *   "backend": "openai",
 *   "latency_ms": 234,
 *   "note": "responding"
 * }
 */
export interface ProbeResult {
  verdict: HealthVerdict;
  backend: string;
  latency_ms: number;
  note: string;
}

/**
 * All component probe results collection
 * 
 * Corresponds to backend _PROBES list:
 * - relational_db    (critical: true)  → SQLite/Postgres
 * - vector_db        (critical: true)  → LanceDB/Qdrant/Milvus
 * - graph_db         (critical: true)  → Kuzu/Neo4j
 * - file_storage     (critical: true)  → Local/S3
 * - llm_provider     (critical: false) → OpenAI/Anthropic/Ollama
 * - embedding_service (critical: false) → Embedding provider
 */
export interface HealthProbes {
  /** Relational database probe (critical) */
  relational_db: ProbeResult;
  /** Vector database probe (critical) */
  vector_db: ProbeResult;
  /** Graph database probe (critical) */
  graph_db: ProbeResult;
  /** File storage probe (critical) */
  file_storage: ProbeResult;
  /** LLM provider probe (non-critical) */
  llm_provider: ProbeResult;
  /** Embedding service probe (non-critical) */
  embedding_service: ProbeResult;
}

/**
 * Detailed health check response
 * 
 * @endpoint GET /health/detailed
 * @description Returns detailed health status of all system components, including latency and error info
 * 
 * @field verdict       - Overall health status (based on all critical components)
 * @field checked_at    - Check time (ISO 8601 format)
 * @field build         - System version
 * @field alive_seconds - Service uptime (seconds)
 * @field probes        - Component probe results
 * 
 * @example
 * {
 *   "verdict": "up",
 *   "checked_at": "2026-03-11T12:54:32.123Z",
 *   "build": "0.5.1",
 *   "alive_seconds": 3600,
 *   "probes": {
 *     "relational_db": { "verdict": "up", "backend": "sqlite", "latency_ms": 5, "note": "session ok" },
 *     "vector_db": { "verdict": "up", "backend": "lancedb", "latency_ms": 12, "note": "index reachable" },
 *     "graph_db": { "verdict": "up", "backend": "kuzu", "latency_ms": 23, "note": "schema valid" },
 *     "file_storage": { "verdict": "up", "backend": "local", "latency_ms": 8, "note": "read/write ok" },
 *     "llm_provider": { "verdict": "up", "backend": "openai", "latency_ms": 234, "note": "responding" },
 *     "embedding_service": { "verdict": "up", "backend": "configured", "latency_ms": 89, "note": "vectors ok" }
 *   }
 * }
 */
export interface DetailedHealthResponse {
  /** Overall health status */
  verdict: HealthVerdict;
  /** Check time (ISO 8601) */
  checked_at: string;
  /** System version */
  build: string;
  /** Service uptime (seconds) */
  alive_seconds: number;
  /** Component probe results */
  probes: HealthProbes;
}

/**
 * Probe result UI display configuration
 * Used to convert backend verdict to UI-friendly display
 */
export interface ProbeDisplayConfig {
  /** Display label */
  label: string;
  /** Color theme */
  color: "green" | "amber" | "red" | "gray";
  /** Icon identifier */
  icon: "check" | "warning" | "error" | "unknown";
  /** Background color variant */
  bgVariant: "success" | "warning" | "error" | "neutral";
}

/**
 * Probe key to display name mapping
 */
export type ProbeKey = keyof HealthProbes;

/**
 * Probe display metadata
 * Used for UI rendering to get friendly name and description of probe items
 */
export interface ProbeMetadata {
  /** Display name */
  displayName: string;
  /** Short description */
  description: string;
  /** Whether it's a critical service */
  isCritical: boolean;
}

/**
 * Prune all request
 * @endpoint POST /api/v1/prune/all
 */
export interface PruneAllRequest {
  confirm: "DELETE_ALL_DATA";
}

/**
 * Prune data only request
 * @endpoint POST /api/v1/prune/data
 */
export interface PruneDataRequest {
  confirm: "DELETE_FILES";
}

/**
 * Prune system request (granular)
 * @endpoint POST /api/v1/prune/system
 */
export interface PruneSystemRequest {
  confirm: "DELETE_SYSTEM";
  graph?: boolean;
  vector?: boolean;
  metadata?: boolean;
  cache?: boolean;
}

/**
 * Prune response
 */
export interface PruneResponse {
  status: string;
  cleared: Record<string, boolean>;
  message: string;
  warnings?: string[];
}

/**
 * Delete document request params
 * @endpoint DELETE /api/v1/delete
 */
export interface DeleteDocumentParams {
  data_id: string;
  dataset_id: string;
  mode?: "soft" | "hard";
}

/**
 * Delete document response
 */
export interface DeleteDocumentResponse {
  status: string;
  message: string;
  graph_deletions?: number;
  data_id: string;
  dataset?: string;
  deleted_node_ids?: string[];
}

/**
 * Sync request
 * @endpoint POST /api/v1/sync
 */
export interface SyncRequest {
  dataset_ids?: string[];
}

/**
 * Sync response
 */
export interface SyncResponse {
  run_id: string;
  status: string;
  dataset_ids: string[];
  dataset_names: string[];
  message: string;
  timestamp: string;
  user_id: string;
}

/**
 * Sync status response
 * @endpoint GET /api/v1/sync/status
 */
export interface SyncStatusResponse {
  has_running_sync: boolean;
  running_sync_count: number;
  latest_running_sync?: {
    run_id: string;
    dataset_ids: string[];
    dataset_names: string[];
    progress_percentage: number;
    created_at: string;
  };
}

/**
 * Dataset status response
 * @endpoint GET /api/v1/datasets/status
 */
export type DatasetStatusResponse = Record<string, string>;

/**
 * Patch node request
 * @endpoint PATCH /api/v1/manual/node
 */
export interface PatchNodeRequest {
  node_id: string;
  node_type: string;
  display_only?: string;
}

/**
 * Patch node result
 */
export interface PatchNodeResult {
  status: string;
  node_id: string;
  node_type: string;
  updated_fields: Record<string, unknown>;
}

/**
 * Responses API request (OpenAI-compatible)
 * @endpoint POST /api/v1/responses/
 */
export interface ResponsesRequest {
  model?: string;
  input: string;
  tools?: unknown[];
  tool_choice?: string;
  user?: string;
  temperature?: number;
}

/**
 * Responses API response
 */
export interface ResponsesResponse {
  id: string;
  created: number;
  model: string;
  object: string;
  status: string;
  tool_calls?: unknown[];
  usage?: Record<string, number>;
}

/**
 * API error response
 */
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * Procedural Extraction Request
 * 
 * @endpoint POST /api/v1/procedural/extract-from-episodic
 * @description Extract procedural memories from existing episodic memories
 */
export interface ExtractProceduralRequest {
  /** Optional list of episode IDs to process */
  episode_ids?: string[];
  /** Maximum number of episodes to process (default: 100, max: 1000) */
  limit?: number;
}

/**
 * Procedural Extraction Response
 */
export interface ExtractProceduralResponse {
  /** Whether the extraction was successful */
  success: boolean;
  /** Number of episodes actually analyzed */
  episodes_analyzed: number;
  /** Number of procedures created */
  procedures_created: number;
  /** Human-readable result message */
  message: string;
}

// ============================================================================
// Episode Quality Check Types
// ============================================================================

/**
 * Episode quality item with issue classification
 */
export interface EpisodeQualityItem {
  id: string;
  name: string;
  facet_count: number;
  issue_type: "empty" | "oversized" | null;
  severity: "high" | "medium" | "low" | null;
  can_size_check: boolean;
}

/**
 * Episode quality statistics
 */
export interface EpisodeQualityStats {
  total_episodes: number;
  empty_count: number;
  oversized_count: number;
  threshold: number;
  threshold_mode: "fixed" | "iqr";
  q1: number;
  q3: number;
  max_facet_count: number;
}

/**
 * Episode quality response from API
 */
export interface EpisodeQualityResponse {
  stats: EpisodeQualityStats;
  problematic_episodes: EpisodeQualityItem[];
  all_episodes: EpisodeQualityItem[];
}

/**
 * Size check result for a single episode
 */
export interface SizeCheckResult {
  episode_id: string;
  episode_name: string | null;
  decision: "SPLIT" | "KEEP" | "SKIPPED" | "ERROR";
  reasoning: string | null;
  new_episodes?: { id: string; name: string; facet_count: number }[];
  adapted_threshold?: number;
}

/**
 * Size check response from API
 */
export interface SizeCheckResponse {
  results: SizeCheckResult[];
  summary: {
    checked: number;
    split: number;
    kept: number;
    skipped: number;
    errors: number;
  };
}

// ============================================================================
// Coreference Resolution Types
// ============================================================================

/**
 * Coreference resolution language mode
 */
export type CorefLanguage = "auto" | "zh" | "en";

/**
 * Coreference configuration
 *
 * @endpoint GET /api/v1/settings/coreference
 * @endpoint POST /api/v1/settings/coreference
 */
export interface CorefConfig {
  enabled: boolean;
  max_history: number;
  session_ttl: number;
  max_sessions: number;
  language: CorefLanguage;
  paragraph_reset: boolean;
}

/**
 * Coreference configuration update request
 */
export interface CorefConfigUpdate {
  enabled?: boolean;
  max_history?: number;
  session_ttl?: number;
  max_sessions?: number;
  language?: CorefLanguage;
  paragraph_reset?: boolean;
}

/**
 * Coreference session information
 */
export interface CorefSessionInfo {
  session_id: string;
  user_id: string;
  turn_count: number;
  last_active: string;
  language: string;
}

/**
 * Coreference module statistics
 *
 * @endpoint GET /api/v1/coreference/stats
 */
export interface CorefStats {
  active_sessions: number;
  max_sessions: number;
  ttl_seconds: number;
  max_history: number;
  sessions?: CorefSessionInfo[];
}

/**
 * Search options extended with coreference parameters
 */
export interface SearchOptionsWithCoref extends SearchOptions {
  coref_enabled?: boolean;
  coref_session_id?: string;
  coref_new_turn?: boolean;
}
