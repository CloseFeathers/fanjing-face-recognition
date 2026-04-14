"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { EpisodicPage } from "./EpisodicPage";
import { TripletPage } from "./TripletPage";
import { ProceduralPage } from "./ProceduralPage";
import { CypherPage } from "./CypherPage";
import { LexicalPage } from "./LexicalPage";

// ============================================================================
// Recall Modes
// ============================================================================

interface RecallMode {
  id: string;
  name: string;
  description: string;
  recommended?: boolean;
}

const RECALL_MODES: RecallMode[] = [
  { id: "episodic", name: "Episodic", description: "Context-aware search using Episode/Facet graph structure", recommended: true },
  { id: "triplet", name: "Triplet Completion", description: "Knowledge graph relation queries" },
  { id: "procedural", name: "Procedural", description: "Step-by-step procedures and guides" },
  { id: "cypher", name: "Cypher Query", description: "Direct graph database queries" },
  { id: "lexical", name: "Lexical Search", description: "Keyword-based chunk retrieval" },
];

// ============================================================================
// Main Page
// ============================================================================

export function RetrievePage() {
  const [selectedMode, setSelectedMode] = useState<string | null>(null);

  if (selectedMode) {
    const backButton = (
      <button
        onClick={() => setSelectedMode(null)}
        className="mb-6 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
      >
        ← Back to methods
      </button>
    );

    switch (selectedMode) {
      case "episodic": return <div>{backButton}<EpisodicPage /></div>;
      case "triplet": return <div>{backButton}<TripletPage /></div>;
      case "procedural": return <div>{backButton}<ProceduralPage /></div>;
      case "cypher": return <div>{backButton}<CypherPage /></div>;
      case "lexical": return <div>{backButton}<LexicalPage /></div>;
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Retrieve</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Select a retrieval method to query the knowledge graph.
        </p>
      </div>

      {/* Methods Grid */}
      <div className="grid grid-cols-1 gap-3">
        {RECALL_MODES.map((mode) => (
          <button
            key={mode.id}
            onClick={() => setSelectedMode(mode.id)}
            className="w-full p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded text-left hover:border-[var(--text-muted)] transition-colors group"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-[var(--text-primary)]">{mode.name}</span>
                  {mode.recommended && (
                    <span className="px-1.5 py-0.5 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded">
                      recommended
                    </span>
                  )}
                </div>
                <p className="text-xs text-[var(--text-muted)] mt-1">{mode.description}</p>
              </div>
              <span className="text-[var(--text-muted)] opacity-0 group-hover:opacity-100 transition-opacity">
                →
              </span>
            </div>
          </button>
        ))}
      </div>

      {/* Quick Guide */}
      <div className="mt-8">
        <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-3">
          Method Selection Guide
        </label>
        <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-4 space-y-2 text-xs text-[var(--text-secondary)]">
          <p><span className="text-[var(--text-primary)]">Episodic</span> — Best for most use cases. Event-based, context-aware, time-sensitive queries.</p>
          <p><span className="text-[var(--text-primary)]">Triplet</span> — For entity relationship queries like "Who founded X?" or "X works at ?".</p>
          <p><span className="text-[var(--text-primary)]">Procedural</span> — For how-to guides, SOPs, and step-by-step instructions.</p>
          <p><span className="text-[var(--text-primary)]">Cypher</span> — Advanced: direct graph queries using Cypher syntax.</p>
          <p><span className="text-[var(--text-primary)]">Lexical</span> — Exact keyword matching, useful for code or technical terms.</p>
        </div>
      </div>
    </div>
  );
}
