"use client";

import React from "react";
import { RecallMode } from "@/types";
import { cn } from "@/lib/utils";
import { Brain, FileText, Network, GitBranch, Search, Code } from "lucide-react";

interface SearchModeSelectorProps {
  value: RecallMode;
  onChange: (mode: RecallMode) => void;
}

interface ModeOption {
  value: RecallMode;
  label: string;
  shortDesc: string;
  icon: React.ReactNode;
}

const modes: ModeOption[] = [
  { value: "EPISODIC", label: "Episodic", shortDesc: "Semantic similarity", icon: <Brain size={16} strokeWidth={1.5} /> },
  { value: "PROCEDURAL", label: "Procedural", shortDesc: "Learned rules", icon: <FileText size={16} strokeWidth={1.5} /> },
  { value: "TRIPLET_COMPLETION", label: "Triplet", shortDesc: "Knowledge triplets", icon: <GitBranch size={16} strokeWidth={1.5} /> },
  { value: "CHUNKS_LEXICAL", label: "Lexical", shortDesc: "Keyword match", icon: <Search size={16} strokeWidth={1.5} /> },
  { value: "CYPHER", label: "Cypher", shortDesc: "Graph query", icon: <Code size={16} strokeWidth={1.5} /> },
];

export function SearchModeSelector({ value, onChange }: SearchModeSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {modes.map((mode) => {
        const isActive = value === mode.value;
        return (
          <button
            key={mode.value}
            onClick={() => onChange(mode.value)}
            className={cn(
              "flex items-center gap-2 px-3 py-2 rounded-md transition-colors text-[13px]",
              isActive
                ? "bg-[var(--text-primary)] text-[var(--bg-base)]"
                : "bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-default)]"
            )}
          >
            {mode.icon}
            <span className="font-medium">{mode.label}</span>
          </button>
        );
      })}
    </div>
  );
}
