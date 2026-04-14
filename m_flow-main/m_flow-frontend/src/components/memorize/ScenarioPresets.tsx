"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Zap,
  BookOpen,
  MessageSquare,
  Shield,
  ClipboardList,
  RefreshCw,
  Microscope,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

// ============================================================================
// Types shared with IngestPage / FileUpload
// ============================================================================

export interface PresetOverrides {
  skipMemorize?: boolean;
  runInBackground?: boolean;
  enableEpisodeRouting?: boolean;
  enableContentRouting?: boolean;
  enableProcedural?: boolean;
  enableFacetPoints?: boolean;
  preciseMode?: boolean;
  contentType?: "text" | "dialog";
  incrementalLoading?: boolean;
}

interface ScenarioPreset {
  id: string;
  label: string;
  Icon: LucideIcon;
  overrides: PresetOverrides;
  summary: string;
}

// ============================================================================
// Preset Definitions
// ============================================================================

const SCENARIO_PRESETS: ScenarioPreset[] = [
  {
    id: "fast-batch",
    label: "Fast Batch Import",
    Icon: Zap,
    overrides: {
      enableEpisodeRouting: false,
      enableContentRouting: false,
      enableFacetPoints: true,
      enableProcedural: false,
      preciseMode: false,
      runInBackground: true,
    },
    summary:
      "Skips episode routing and sentence classification for maximum throughput. " +
      "Episodes are created independently without cross-document merging. " +
      "Best for importing a large batch of unrelated documents quickly.",
  },
  {
    id: "knowledge-base",
    label: "Knowledge Base",
    Icon: BookOpen,
    overrides: {
      enableEpisodeRouting: true,
      enableContentRouting: true,
      enableFacetPoints: true,
      enableProcedural: false,
      preciseMode: false,
      contentType: "text",
    },
    summary:
      "Full episodic pipeline with episode routing, sentence classification, and " +
      "fine-grained FacetPoints. New content is merged into existing episodes when " +
      "related. Best for building a structured knowledge base over time.",
  },
  {
    id: "chat-meeting",
    label: "Chat / Meeting Notes",
    Icon: MessageSquare,
    overrides: {
      enableEpisodeRouting: true,
      enableContentRouting: true,
      enableFacetPoints: true,
      enableProcedural: false,
      preciseMode: false,
      contentType: "dialog",
    },
    summary:
      "Dialog mode splits text by speaker turns instead of sentence boundaries, " +
      "preserving conversational structure. Episode routing merges related discussions " +
      "across sessions. Best for chat logs, meeting transcripts, and interviews.",
  },
  {
    id: "precise-archive",
    label: "Precise Archival",
    Icon: Shield,
    overrides: {
      enableEpisodeRouting: true,
      enableContentRouting: true,
      enableFacetPoints: true,
      enableProcedural: false,
      preciseMode: true,
      contentType: "text",
    },
    summary:
      "Zero-loss summarization preserves ALL factual information \u2014 every date, number, " +
      "name, and constraint. Significantly higher LLM token cost. Best for contracts, " +
      "financial reports, legal documents, and audit-critical content.",
  },
  {
    id: "sop-howto",
    label: "SOP / How-to Guides",
    Icon: ClipboardList,
    overrides: {
      enableEpisodeRouting: true,
      enableContentRouting: true,
      enableFacetPoints: true,
      enableProcedural: true,
      preciseMode: false,
      contentType: "text",
    },
    summary:
      "Enables procedural memory extraction to capture step-by-step processes, " +
      "preferences, and reusable workflows alongside episodic facts. Higher token cost. " +
      "Best for SOPs, tutorials, onboarding docs, and best-practice guides.",
  },
  {
    id: "incremental-daily",
    label: "Daily Incremental",
    Icon: RefreshCw,
    overrides: {
      enableEpisodeRouting: true,
      enableContentRouting: false,
      enableFacetPoints: true,
      enableProcedural: false,
      preciseMode: false,
      incrementalLoading: true,
      runInBackground: true,
    },
    summary:
      "Lightweight pipeline for appending a few new items to an existing dataset daily. " +
      "Episode routing maintains continuity with past content, but sentence classification " +
      "is skipped for speed.",
  },
  {
    id: "full-analysis",
    label: "Full Analysis",
    Icon: Microscope,
    overrides: {
      enableEpisodeRouting: true,
      enableContentRouting: true,
      enableFacetPoints: true,
      enableProcedural: true,
      preciseMode: true,
      contentType: "text",
    },
    summary:
      "Every feature enabled: precise summarization, procedural extraction, FacetPoints, " +
      "episode routing, and content routing. Maximum memory quality at maximum cost. " +
      "Best for small but critical document sets where nothing can be missed.",
  },
];

// ============================================================================
// Component
// ============================================================================

export function ScenarioPresets({
  onApply,
  disabled = false,
}: {
  onApply: (overrides: PresetOverrides) => void;
  disabled?: boolean;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const activePreset = SCENARIO_PRESETS.find((p) => p.id === selected);

  return (
    <div className="space-y-3 pb-3 mb-3 border-b border-[var(--border-subtle)]">
      <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
        Scenario Presets
      </p>
      <div className="grid grid-cols-2 gap-1.5">
        {SCENARIO_PRESETS.map((preset) => {
          const Icon = preset.Icon;
          return (
            <button
              key={preset.id}
              disabled={disabled}
              onClick={() => {
                setSelected(preset.id);
                onApply(preset.overrides);
              }}
              className={cn(
                "flex items-center gap-2 px-2.5 py-2 text-left text-[11px] rounded border transition-colors",
                selected === preset.id
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                  : "bg-[var(--bg-elevated)] border-[var(--border-subtle)] text-[var(--text-secondary)] hover:border-[var(--text-muted)]",
                disabled && "opacity-50 cursor-not-allowed"
              )}
            >
              <Icon
                size={14}
                className={cn(
                  "flex-shrink-0",
                  selected === preset.id
                    ? "text-blue-400"
                    : "text-[var(--text-muted)]"
                )}
              />
              <span className="truncate">{preset.label}</span>
            </button>
          );
        })}
      </div>
      {activePreset && (
        <div className="px-3 py-2 bg-blue-500/5 border border-blue-500/20 rounded text-[10px] text-[var(--text-muted)] leading-relaxed">
          {activePreset.summary}
        </div>
      )}
    </div>
  );
}
