"use client";

import React, { useState } from "react";
import { useDetailedHealth, useDatasets, useSearchHistory } from "@/hooks/use-api";
import {
  Activity,
  HardDrive,
  AlertCircle,
  Loader2,
  RefreshCw,
  Database,
  Server,
  Cpu,
  Box,
  Search,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProbeKey, ProbeResult, SearchHistoryEntry } from "@/types";

// ============================================================================
// Tab Configuration
// ============================================================================

type TabId = "logs" | "resources";

const TABS: Array<{ id: TabId; label: string; icon: React.ReactNode }> = [
  { id: "logs", label: "Search History", icon: <Search size={14} strokeWidth={1.5} /> },
  { id: "resources", label: "System Status", icon: <HardDrive size={14} strokeWidth={1.5} /> },
];

// ============================================================================
// Probe Display Configuration
// ============================================================================

const PROBE_CONFIG: Record<ProbeKey, { name: string; icon: React.ReactNode; description: string }> = {
  relational_db: { name: "Relational Database", icon: <Server size={16} />, description: "Metadata storage" },
  vector_db: { name: "Vector Database", icon: <Database size={16} />, description: "Embedding storage" },
  graph_db: { name: "Graph Database", icon: <Box size={16} />, description: "Knowledge graph storage" },
  file_storage: { name: "File Storage", icon: <HardDrive size={16} />, description: "Document file storage" },
  llm_provider: { name: "LLM Provider", icon: <Cpu size={16} />, description: "Language model API connection" },
  embedding_service: { name: "Embedding Service", icon: <Activity size={16} />, description: "Text vectorization" },
};

function getVerdictColor(verdict: string): string {
  switch (verdict) {
    case "up":
      return "text-green-400 bg-green-500/10";
    case "degraded":
      return "text-amber-400 bg-amber-500/10";
    case "down":
      return "text-red-400 bg-red-500/10";
    default:
      return "text-[var(--text-muted)] bg-[var(--bg-elevated)]";
  }
}

// ============================================================================
// Main Component
// ============================================================================

export function AuditPage() {
  const [activeTab, setActiveTab] = useState<TabId>("resources");

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Monitoring & Audit</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">View operation logs, API statistics, and resource usage.</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg w-fit">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2 text-[13px] rounded-md transition-colors",
              activeTab === tab.id
                ? "bg-[var(--text-primary)] text-[var(--bg-base)]"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            )}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeTab === "logs" && <AuditLogsPanel />}
      {activeTab === "resources" && <ResourcesPanel />}
    </div>
  );
}

// =============================================================================
// Search History Panel
// =============================================================================

function AuditLogsPanel() {
  const { data: searchHistory, isLoading, error, refetch } = useSearchHistory();

  const formatDateTime = (dateStr: string): string => {
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  const truncateText = (text: string, maxLength: number = 60): string => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  };

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <AlertCircle size={32} className="text-[var(--error)] mb-3" />
        <p className="text-[var(--error)] text-sm mb-2">Failed to load search history</p>
        <p className="text-[var(--text-muted)] text-xs mb-4">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2 text-xs bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded hover:border-[var(--text-muted)] transition-colors"
        >
          <RefreshCw size={12} />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={24} className="animate-spin text-[var(--text-muted)]" />
        </div>
      )}

      {/* Search History Table */}
      {!isLoading && (
        <div className="border border-[var(--border-subtle)] rounded-lg overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-[var(--bg-surface)] border-b border-[var(--border-subtle)]">
                <th className="px-4 py-3 text-left text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wider">
                  Time
                </th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wider">
                  Query
                </th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wider">
                  User
                </th>
              </tr>
            </thead>
            <tbody>
              {searchHistory && searchHistory.length > 0 ? (
                searchHistory.map((entry: SearchHistoryEntry) => (
                  <tr key={entry.id} className="border-b border-[var(--border-subtle)] last:border-0 hover:bg-[var(--bg-elevated)] transition-colors">
                    <td className="px-4 py-3 text-[12px] text-[var(--text-muted)]">
                      {formatDateTime(entry.createdAt)}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <Search size={12} className="text-[var(--text-muted)] flex-shrink-0" />
                        <span className="text-[13px] text-[var(--text-primary)]" title={entry.text}>
                          {truncateText(entry.text)}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[12px] text-[var(--text-secondary)]">
                      {entry.user}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={3} className="px-4 py-12 text-center text-[var(--text-muted)]">
                    <Search size={24} className="mx-auto mb-2 opacity-50" />
                    <p className="text-[13px]">No search history found</p>
                    <p className="text-[11px] mt-1">Perform a search to see history here</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Refresh Button */}
      {searchHistory && searchHistory.length > 0 && (
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-[var(--text-muted)]">
            {searchHistory.length} search{searchHistory.length !== 1 ? "es" : ""} recorded
          </span>
          <button
            onClick={() => refetch()}
            disabled={isLoading}
            className="flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// System Status Panel (formerly Resources)
// =============================================================================

function ResourcesPanel() {
  const { data: healthData, isLoading, error, refetch } = useDetailedHealth();
  const { data: datasets } = useDatasets();

  const formatUptime = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
  };

  const renderProbeCard = (key: ProbeKey, probe: ProbeResult) => {
    const config = PROBE_CONFIG[key];
    if (!config) return null;

    return (
      <div
        key={key}
        className="p-5 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg"
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className="text-[var(--text-muted)]">{config.icon}</span>
            <div>
              <h3 className="text-[14px] font-medium text-[var(--text-primary)]">{config.name}</h3>
              <p className="text-[11px] text-[var(--text-muted)]">{probe.backend || config.description}</p>
            </div>
          </div>
          <span className={cn("px-2 py-0.5 text-[10px] rounded capitalize", getVerdictColor(probe.verdict))}>
            {probe.verdict}
          </span>
        </div>
        <div className="space-y-2">
          {probe.latency_ms !== undefined && (
            <div className="flex items-center justify-between text-[12px]">
              <span className="text-[var(--text-muted)]">Latency</span>
              <span className="text-[var(--text-secondary)]">{probe.latency_ms}ms</span>
            </div>
          )}
          {probe.note && (
            <div className="flex items-center justify-between text-[12px]">
              <span className="text-[var(--text-muted)]">Note</span>
              <span className="text-[var(--text-secondary)]">{probe.note}</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <AlertCircle size={32} className="text-[var(--error)] mb-3" />
        <p className="text-[var(--error)] text-sm mb-2">Failed to load resource status</p>
        <p className="text-[var(--text-muted)] text-xs mb-4">
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2 text-xs bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded hover:border-[var(--text-muted)] transition-colors"
        >
          <RefreshCw size={12} />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Loading State */}
      {isLoading && !healthData && (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={24} className="animate-spin text-[var(--text-muted)]" />
        </div>
      )}

      {healthData && (
        <>
          {/* System Overview */}
          <div className="p-5 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-[14px] font-medium text-[var(--text-primary)]">System Overview</h3>
              <span className={cn("px-2 py-0.5 text-[10px] rounded capitalize", getVerdictColor(healthData.verdict))}>
                {healthData.verdict}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-6">
              <div>
                <p className="text-[11px] text-[var(--text-muted)] mb-1">Version</p>
                <p className="text-[13px] text-[var(--text-primary)] font-mono">{healthData.build}</p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--text-muted)] mb-1">Uptime</p>
                <p className="text-[13px] text-[var(--text-primary)]">{formatUptime(healthData.alive_seconds)}</p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--text-muted)] mb-1">Datasets</p>
                <p className="text-[13px] text-[var(--text-primary)]">{datasets?.length || 0}</p>
              </div>
              <div>
                <p className="text-[11px] text-[var(--text-muted)] mb-1">Last Check</p>
                <p className="text-[13px] text-[var(--text-primary)]">
                  {new Date(healthData.checked_at).toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>

          {/* Service Status Grid */}
          <div>
            <h3 className="text-[12px] font-medium text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Services
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {(Object.entries(healthData.probes) as Array<[ProbeKey, ProbeResult]>).map(([key, probe]) =>
                renderProbeCard(key, probe)
              )}
            </div>
          </div>

          {/* Refresh Button */}
          <div className="flex justify-end">
            <button
              onClick={() => refetch()}
              disabled={isLoading}
              className="flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
            >
              <RefreshCw size={12} className={isLoading ? "animate-spin" : ""} />
              Refresh
            </button>
          </div>
        </>
      )}
    </div>
  );
}
