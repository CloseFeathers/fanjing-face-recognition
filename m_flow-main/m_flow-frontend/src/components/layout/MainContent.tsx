"use client";

import React from "react";
import dynamic from "next/dynamic";
import { useUIStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { CommandPalette, useCommandPalette, UnavailablePage } from "@/components/common";
import { Loader2 } from "lucide-react";

// Loading component for lazy-loaded pages
const PageLoader = () => (
  <div className="flex items-center justify-center py-20">
    <Loader2 size={20} className="text-[var(--text-muted)] animate-spin" />
  </div>
);

// ============================================================================
// Core Components (always loaded - first screen)
// ============================================================================
import { DashboardPage } from "@/components/dashboard";
import { SetupPage } from "@/components/setup";

// ============================================================================
// Lazy-loaded Components (code splitting for non-first-screen pages)
// ============================================================================

// Search & Prompts
const SearchPanel = dynamic(() => import("@/components/search").then(m => ({ default: m.SearchPanel })), { loading: PageLoader });
const PromptsPage = dynamic(() => import("@/components/settings").then(m => ({ default: m.PromptsPage })), { loading: PageLoader });

// Datasets & Graph
const DatasetsPage = dynamic(() => import("@/components/datasets").then(m => ({ default: m.DatasetsPage })), { loading: PageLoader });
const GraphViewSelector = dynamic(() => import("@/components/graph").then(m => ({ default: m.GraphViewSelector })), { loading: PageLoader });

// Memorize
const IngestPage = dynamic(() => import("@/components/memorize").then(m => ({ default: m.IngestPage })), { loading: PageLoader });
const AddPage = dynamic(() => import("@/components/memorize").then(m => ({ default: m.AddPage })), { loading: PageLoader });
const MemorizePage = dynamic(() => import("@/components/memorize").then(m => ({ default: m.MemorizePage })), { loading: PageLoader });
const ManualIngestPage = dynamic(() => import("@/components/memorize").then(m => ({ default: m.ManualIngestPage })), { loading: PageLoader });
const ExtractProceduresPage = dynamic(() => import("@/components/memorize").then(m => ({ default: m.ExtractProceduresPage })), { loading: PageLoader });

// Retrieve
const RetrievePage = dynamic(() => import("@/components/retrieve").then(m => ({ default: m.RetrievePage })), { loading: PageLoader });
const EpisodicPage = dynamic(() => import("@/components/retrieve").then(m => ({ default: m.EpisodicPage })), { loading: PageLoader });
const TripletPage = dynamic(() => import("@/components/retrieve").then(m => ({ default: m.TripletPage })), { loading: PageLoader });
const ProceduralPage = dynamic(() => import("@/components/retrieve").then(m => ({ default: m.ProceduralPage })), { loading: PageLoader });
const CypherPage = dynamic(() => import("@/components/retrieve").then(m => ({ default: m.CypherPage })), { loading: PageLoader });
const LexicalPage = dynamic(() => import("@/components/retrieve").then(m => ({ default: m.LexicalPage })), { loading: PageLoader });

// Other pages
const ExportPage = dynamic(() => import("@/components/export").then(m => ({ default: m.ExportPage })), { loading: PageLoader });
const UserManagementPage = dynamic(() => import("@/components/users").then(m => ({ default: m.UserManagementPage })), { loading: PageLoader });
const LearnPage = dynamic(() => import("@/components/learn").then(m => ({ default: m.LearnPage })), { loading: PageLoader });


// Legacy Components
const FileUpload = dynamic(() => import("@/components/upload").then(m => ({ default: m.FileUpload })), { loading: PageLoader });
const HealthPage = dynamic(() => import("@/components/system").then(m => ({ default: m.HealthPage })), { loading: PageLoader });
const PrunePage = dynamic(() => import("@/components/system").then(m => ({ default: m.PrunePage })), { loading: PageLoader });
const PermissionsPage = dynamic(() => import("@/components/admin").then(m => ({ default: m.PermissionsPage })), { loading: PageLoader });
const AuditPage = dynamic(() => import("@/components/admin").then(m => ({ default: m.AuditPage })), { loading: PageLoader });

// ============================================================================
// MainContent component
// ============================================================================

export function MainContent() {
  const { currentView, sidebarCollapsed, datasetContext } = useUIStore();
  const commandPalette = useCommandPalette();

  const renderContent = () => {
    switch (currentView) {
      // ===== Setup =====
      case "setup":
        return <SetupPage />;
      
      // ===== Dashboard =====
      case "dashboard":
        return <DashboardPage />;
      
      // ===== Memorize =====
      case "memorize":
      case "memorize-ingest":
        return <IngestPage />;
      case "memorize-add":
        return <AddPage />;
      case "memorize-process":
        return <MemorizePage />;
      case "memorize-manual":
        return <ManualIngestPage />;
      case "memorize-learn":
        return <ExtractProceduresPage />;
      
      // ===== Retrieve =====
      case "retrieve":
        return <RetrievePage />;
      case "retrieve-episodic":
        return <EpisodicPage />;
      case "retrieve-triplet":
        return <TripletPage />;
      case "retrieve-procedural":
        return <ProceduralPage />;
      case "retrieve-cypher":
        return <CypherPage />;
      case "retrieve-lexical":
        return <LexicalPage />;
      
      
      // ===== Memories =====
      case "memories":
        return <GraphViewSelector datasetId={datasetContext.datasetId ?? undefined} />;
      
      // ===== Export =====
      case "export":
        return <ExportPage />;
      
      // ===== Prompts =====
      case "prompts":
        return <PromptsPage />;
      
      // ===== User Management =====
      case "users":
        return <UserManagementPage />;
      
      // ===== Admin =====
      case "permissions":
        return <PermissionsPage />;
      case "audit":
        return <AuditPage />;
      
      // ===== Docs (redirects to external site) =====
      
      // ===== Legacy Routes (backward compatibility) =====
      case "search":
        return <SearchPanel />;
      case "add-memory":
        return <FileUpload />;
      case "datasets":
        return <DatasetsPage />;
      case "graph":
        return <GraphViewSelector datasetId={datasetContext.datasetId ?? undefined} />;
      case "quick-start":
      case "learn":
        return <LearnPage />;
      case "health":
        return <HealthPage />;
      case "prune":
        return <PrunePage />;
      
      default:
        return <DashboardPage />;
    }
  };

  return (
    <>
      <main
        className={cn(
          "min-h-screen pt-14 transition-all duration-200 bg-[var(--bg-base)]",
          sidebarCollapsed ? "pl-14" : "pl-56"
        )}
      >
        <div className="px-8 py-6">{renderContent()}</div>
      </main>

      {/* Global Command Palette */}
      <CommandPalette
        isOpen={commandPalette.isOpen}
        onClose={commandPalette.close}
      />
    </>
  );
}
