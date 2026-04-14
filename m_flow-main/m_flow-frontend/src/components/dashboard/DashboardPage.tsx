"use client";

/**
 * DashboardPage Component
 *
 * Main dashboard view displaying system overview, key metrics,
 * recent activity, and quick actions.
 *
 * Features:
 * - Health status bar
 * - Key statistics with trends
 * - Activity feed
 * - Quick action cards
 * - Onboarding for new users
 * - Recent datasets
 */

import React, { useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { useDatasetsWithCounts, useEpisodesOverview, useDetailedHealth } from "@/hooks/use-api";
import { useUIStore } from "@/lib/store";
import {
  Database,
  ArrowRight,
  Search,
  Plus,
  Upload,
  Network,
  FileText,
} from "lucide-react";

import { HealthBar } from "./HealthBar";
import { ActiveOperations } from "./ActiveOperations";
import { ActivityFeed } from "./ActivityFeed";
import { StatsCard, StatsGrid } from "./StatsCard";
import { OnboardingCard, createDefaultOnboardingSteps } from "./OnboardingCard";

// ============================================================================
// Types
// ============================================================================

interface QuickActionProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  onClick: () => void;
}

// ============================================================================
// Quick Action Card Component
// ============================================================================

function QuickAction({ title, description, icon, onClick }: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-start gap-3 p-4 rounded-lg text-left transition-all w-full",
        "border border-[var(--border-subtle)]",
        "hover:border-[var(--border-default)] hover:bg-[var(--bg-hover)]",
        "focus:outline-none focus:ring-1 focus:ring-[var(--border-default)]",
        "group"
      )}
    >
      <div className="w-9 h-9 rounded-lg bg-[var(--bg-elevated)] flex items-center justify-center text-[var(--text-muted)] group-hover:text-[var(--text-secondary)] transition-colors">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[14px] font-medium text-[var(--text-primary)]">
            {title}
          </span>
          <ArrowRight
            size={14}
            className="text-[var(--text-muted)] opacity-0 group-hover:opacity-100 transition-opacity"
          />
        </div>
        <p className="text-[12px] text-[var(--text-muted)] mt-0.5">
          {description}
        </p>
      </div>
    </button>
  );
}

// ============================================================================
// Recent Dataset Item Component
// ============================================================================

interface DatasetItemProps {
  name: string;
  documentCount: number;
  onClick: () => void;
}

function DatasetItem({ name, documentCount, onClick }: DatasetItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center justify-between p-3 rounded-lg",
        "hover:bg-[var(--bg-hover)] transition-colors",
        "focus:outline-none focus:ring-1 focus:ring-[var(--border-default)]",
        "group"
      )}
    >
      <div className="flex items-center gap-3">
        <Database size={14} className="text-[var(--text-muted)]" />
        <span className="text-[13px] text-[var(--text-primary)]">{name}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[11px] text-[var(--text-muted)]">
          {documentCount} docs
        </span>
        <ArrowRight
          size={12}
          className="text-[var(--text-muted)] opacity-0 group-hover:opacity-100 transition-opacity"
        />
      </div>
    </button>
  );
}

// ============================================================================
// Section Header Component
// ============================================================================

interface SectionHeaderProps {
  title: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

function SectionHeader({ title, action }: SectionHeaderProps) {
  return (
    <div className="flex items-center justify-between mb-3">
      <h2 className="text-[13px] font-medium text-[var(--text-primary)]">
        {title}
      </h2>
      {action && (
        <button
          onClick={action.onClick}
          className="text-[11px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
        >
          {action.label} →
        </button>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function DashboardPage() {
  const [onboardingDismissed, setOnboardingDismissed] = useState(false);
  const { data: datasets, isLoading: isLoadingDatasets } = useDatasetsWithCounts();
  const { data: episodesData } = useEpisodesOverview();
  const { data: healthData } = useDetailedHealth({ refetchInterval: false });
  const { setCurrentView } = useUIStore();

  // Calculate stats from lightweight episodes overview (avoids loading entire graph)
  const stats = useMemo(() => {
    const episodes = episodesData?.episodes || [];
    const totalFacets = episodes.reduce((sum, ep) => sum + (ep.facetCount || 0), 0);
    const totalEntityRefs = episodes.reduce((sum, ep) => sum + (ep.entityCount || 0), 0);
    return {
      datasets: datasets?.length || 0,
      episodes: episodesData?.total || 0,
      facets: totalFacets,
      entityRefs: totalEntityRefs,
    };
  }, [datasets, episodesData]);

  // Stabilize isNewUser: once we've resolved it, don't flicker on refetch
  const hasEverLoaded = useRef(false);
  if (!isLoadingDatasets) hasEverLoaded.current = true;
  const isNewUser = hasEverLoaded.current && (!datasets || datasets.length === 0);

  // Only show loading skeleton on first mount (before any data arrives)
  const showStatsLoading = isLoadingDatasets && !hasEverLoaded.current;

  // Check LLM configuration status
  const isLLMConfigured = healthData?.probes?.llm_provider?.verdict === "up";

  // Onboarding steps
  const onboardingSteps = useMemo(() => {
    return createDefaultOnboardingSteps(
      {
        llmConfigured: isLLMConfigured,
        hasDocuments: stats.episodes > 0,
      },
      {
        onConfigureLLM: () => setCurrentView("setup"),
        onAddDocument: () => setCurrentView("memorize-add"),
      }
    );
  }, [isLLMConfigured, stats.episodes, setCurrentView]);

  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Dashboard</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Your knowledge base at a glance
        </p>
      </div>

      {/* Health Status Bar */}
      <section className="mb-8">
        <HealthBar />
      </section>

      {/* Active Operations */}
      <section className="mb-8">
        <ActiveOperations />
      </section>

      {/* Onboarding Card (for new users) */}
      {isNewUser && !onboardingDismissed && (
        <section className="mb-8">
          <OnboardingCard
            steps={onboardingSteps}
            onDismiss={() => setOnboardingDismissed(true)}
            onStepClick={(stepId) => {
              const step = onboardingSteps.find((s) => s.id === stepId);
              step?.action?.onClick();
            }}
          />
        </section>
      )}

      {/* Stats Grid */}
      <section className="mb-10">
        <StatsGrid columns={4}>
          <StatsCard
            label="Datasets"
            value={stats.datasets}
            trend={stats.datasets > 0 ? { value: stats.datasets, direction: "up", label: "total" } : undefined}
            loading={showStatsLoading}
            onClick={() => setCurrentView("memories")}
          />
          <StatsCard
            label="Episodes"
            value={stats.episodes}
            trend={stats.episodes > 0 ? { value: stats.episodes, direction: "up", label: "total" } : undefined}
            onClick={() => setCurrentView("memories")}
          />
          <StatsCard
            label="Facets"
            value={stats.facets}
            trend={stats.facets > 0 ? { value: stats.facets, direction: "up", label: "total" } : undefined}
            onClick={() => setCurrentView("memories")}
          />
          <StatsCard
            label="Entity Refs"
            value={stats.entityRefs}
            trend={stats.entityRefs > 0 ? { value: stats.entityRefs, direction: "up", label: "total" } : undefined}
            onClick={() => setCurrentView("memories")}
          />
        </StatsGrid>
      </section>

      {/* Two-column layout: Quick Actions + Activity */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
        {/* Quick Actions */}
        <div>
          <SectionHeader title="Quick Actions" />
          <div className="space-y-2">
            <QuickAction
              title="Search"
              description="Query your knowledge base"
              icon={<Search size={18} />}
              onClick={() => setCurrentView("retrieve-episodic")}
            />
            <QuickAction
              title="Add Memory"
              description="Import documents or text"
              icon={<Upload size={18} />}
              onClick={() => setCurrentView("memorize-add")}
            />
            <QuickAction
              title="View Graph"
              description="Explore knowledge relationships"
              icon={<Network size={18} />}
              onClick={() => setCurrentView("memories")}
            />
          </div>
        </div>

        {/* Activity Feed */}
        <div>
          <ActivityFeed
            limit={5}
            onActivityClick={(activity) => {
              // Navigate based on activity type
              if (activity.type === "search") {
                setCurrentView("retrieve-episodic");
              } else if (activity.type === "ingest") {
                setCurrentView("memorize-ingest");
              }
            }}
          />
        </div>
      </section>

      {/* Recent Datasets */}
      {datasets && datasets.length > 0 && (
        <section>
          <SectionHeader
            title="Recent Datasets"
            action={{
              label: "View all",
              onClick: () => setCurrentView("memories"),
            }}
          />
          <div className="rounded-lg border border-[var(--border-subtle)] overflow-hidden">
            {datasets.slice(0, 5).map((ds) => (
              <DatasetItem
                key={ds.id}
                name={ds.name}
                documentCount={ds.dataCount || 0}
                onClick={() => setCurrentView("memories")}
              />
            ))}
          </div>
        </section>
      )}

      {/* Empty State (fallback) */}
      {!isNewUser && datasets && datasets.length === 0 && (
        <section className="text-center py-12">
          <div className="w-12 h-12 rounded-full bg-[var(--bg-elevated)] flex items-center justify-center mx-auto mb-4">
            <FileText size={20} className="text-[var(--text-muted)]" />
          </div>
          <p className="text-[14px] text-[var(--text-secondary)] mb-2">
            No datasets yet
          </p>
          <p className="text-[12px] text-[var(--text-muted)] mb-4">
            Start by adding your first document to build your knowledge base
          </p>
          <button
            onClick={() => setCurrentView("memorize-add")}
            className={cn(
              "inline-flex items-center gap-2 px-4 py-2 rounded-lg",
              "bg-[var(--text-primary)] text-[var(--bg-base)]",
              "text-[13px] font-medium",
              "hover:opacity-90 transition-opacity"
            )}
          >
            <Plus size={16} />
            Add Memory
          </button>
        </section>
      )}
    </div>
  );
}

DashboardPage.displayName = "DashboardPage";

export default DashboardPage;
