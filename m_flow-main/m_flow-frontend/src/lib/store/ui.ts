import { create } from "zustand";

// ============================================================================
// View Types
// ============================================================================

export type View = 
  // Setup
  | "setup"
  
  // Dashboard  
  | "dashboard"
  
  // Memorize (Ingestion)
  | "memorize"
  | "memorize-ingest"
  | "memorize-add"
  | "memorize-process"
  | "memorize-manual"
  | "memorize-learn"
  
  // Retrieve (Search)
  | "retrieve"
  | "retrieve-episodic"
  | "retrieve-triplet"
  | "retrieve-procedural"
  | "retrieve-cypher"
  | "retrieve-lexical"
  
  // Memories (Graph)
  | "memories"
  
  // Memory Export
  | "export"
  
  // Prompts
  | "prompts"
  
  // User Management
  | "users"
  
  // Admin
  | "permissions"
  | "audit"
  
  // Docs
  | "docs"
  | "integration-examples"
  
  
  // Legacy (backward compatibility)
  | "search"
  | "add-memory"
  | "graph"
  | "datasets"
  | "health"
  | "quick-start"
  | "learn"
  | "prune";

// ============================================================================
// Dataset Context - Currently selected user and dataset
// ============================================================================

interface DatasetContext {
  userId: string | null;
  datasetId: string | null;
  datasetName: string | null;
}

// ============================================================================
// UI State
// ============================================================================

interface UIState {
  // Navigation
  currentView: View;
  sidebarCollapsed: boolean;
  searchQuery: string;
  
  // Dataset Context (header dropdown selection)
  datasetContext: DatasetContext;
  
  // Actions
  setCurrentView: (view: View) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setSearchQuery: (query: string) => void;
  setDatasetContext: (context: Partial<DatasetContext>) => void;
}

export const useUIStore = create<UIState>()((set) => ({
  // Initial state
  currentView: "dashboard",
  sidebarCollapsed: false,
  searchQuery: "",
  datasetContext: {
    userId: null,
    datasetId: null,
    datasetName: null,
  },
  
  // Actions
  setCurrentView: (view) => set({ currentView: view }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  setSearchQuery: (query) => set({ searchQuery: query }),
  setDatasetContext: (context) => set((state) => ({ 
    datasetContext: { ...state.datasetContext, ...context } 
  })),
}));
