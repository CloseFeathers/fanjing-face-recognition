"use client";

import React, { useMemo, useState } from "react";
import { Database, ListOrdered, FileText, ChevronDown, ChevronUp } from "lucide-react";
import type { SearchResultItem } from "@/types";

interface StructuredProcedure {
  title: string;
  summary: string;
  search_text: string;
  context_text: string;
  points_text: string;
  version: number;
  status: string;
  confidence: string;
  steps: string[];
  context_points: { type: string; text: string }[];
}

interface ParsedResult {
  procedures: StructuredProcedure[];
  isStructured: boolean;
  rawText: string;
}

const CONTEXT_LABELS: Record<string, string> = {
  when: "When",
  why: "Why",
  boundary: "Boundary",
  outcome: "Outcome",
  prereq: "Prerequisites",
  exception: "Exceptions",
  context: "Context",
  misc: "Note",
};

function parseContent(raw: string): ParsedResult {
  const cleaned = raw
    .replace(/__node_content_start__/g, "")
    .replace(/__node_content_end__/g, "")
    .trim();

  try {
    const parsed = JSON.parse(cleaned);
    if (parsed?.__procedural_structured__ && Array.isArray(parsed.procedures)) {
      return { procedures: parsed.procedures, isStructured: true, rawText: "" };
    }
  } catch {}

  return { procedures: [], isStructured: false, rawText: cleaned };
}

function ProcedureBlock({ proc, index }: { proc: StructuredProcedure; index: number }) {
  return (
    <div className="space-y-3">
      {/* Summary */}
      {proc.summary && (
        <p className="text-xs text-[var(--text-secondary)] leading-relaxed line-clamp-4">{proc.summary}</p>
      )}

      {/* Context points */}
      {proc.context_points.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <FileText size={11} className="text-[var(--text-muted)]" />
            <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-medium">Context</span>
          </div>
          <div className="pl-4 space-y-1">
            {proc.context_points.map((cp, i) => (
              <div key={i} className="flex gap-2">
                <span className="text-[9px] text-[var(--text-muted)] font-mono uppercase flex-shrink-0 w-16 text-right pt-0.5">
                  {CONTEXT_LABELS[cp.type] || cp.type}
                </span>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">{cp.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Context text fallback */}
      {proc.context_points.length === 0 && proc.context_text && (
        <div>
          <div className="flex items-center gap-1.5 mb-1">
            <FileText size={11} className="text-[var(--text-muted)]" />
            <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-medium">Context</span>
          </div>
          <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed pl-4 whitespace-pre-wrap line-clamp-4">{proc.context_text}</p>
        </div>
      )}

      {/* Steps */}
      {proc.steps.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-1.5">
            <ListOrdered size={11} className="text-[var(--text-muted)]" />
            <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-medium">
              Steps ({proc.steps.length})
            </span>
          </div>
          <div className="pl-4 space-y-1">
            {proc.steps.map((step, i) => (
              <div key={i} className="flex gap-2 items-start">
                <span className="text-[9px] text-[var(--text-muted)] font-mono flex-shrink-0 w-4 text-right pt-0.5">{i + 1}</span>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">{step}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Points text fallback */}
      {proc.steps.length === 0 && proc.points_text && (
        <div>
          <div className="flex items-center gap-1.5 mb-1">
            <ListOrdered size={11} className="text-[var(--text-muted)]" />
            <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-medium">Key Points</span>
          </div>
          <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed pl-4 whitespace-pre-wrap line-clamp-6">{proc.points_text}</p>
        </div>
      )}

      {/* Metadata */}
      <div className="flex gap-3 text-[9px] text-[var(--text-muted)] pt-1">
        {proc.version > 1 && <span>v{proc.version}</span>}
        {proc.status !== "active" && <span>{proc.status}</span>}
        {proc.confidence !== "high" && <span className="text-amber-400/60">{proc.confidence}</span>}
      </div>
    </div>
  );
}

export function ProceduralResultCard({ result, index }: { result: SearchResultItem; index: number }) {
  const parsed = useMemo(() => parseContent(result.content), [result.content]);

  if (!parsed.isStructured) {
    return (
      <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-[var(--border-subtle)] flex items-center gap-3">
          <div className="w-6 h-6 bg-[var(--bg-elevated)] rounded flex items-center justify-center">
            <span className="text-[10px] text-[var(--text-muted)] font-mono">{index + 1}</span>
          </div>
          <span className="text-sm text-[var(--text-primary)]">Procedure</span>
          {result.score !== undefined && (
            <span className="text-[10px] text-[var(--text-muted)] bg-[var(--bg-elevated)] px-1.5 py-0.5 rounded font-mono ml-auto">
              {(result.score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <div className="px-4 py-3">
          <p className="text-xs text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">{parsed.rawText}</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {parsed.procedures.map((proc, pi) => (
        <div key={pi} className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg hover:border-[var(--border-default)] transition-colors overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 border-b border-[var(--border-subtle)]">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-6 h-6 bg-[rgba(255,255,255,0.06)] rounded flex items-center justify-center mt-0.5">
                <span className="text-[10px] text-[var(--text-muted)] font-mono">{pi + 1}</span>
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-[var(--text-primary)] leading-snug">
                  {proc.title || proc.search_text || "Procedure"}
                </h3>
                <div className="flex flex-wrap items-center gap-2 mt-1">
                  {result.score !== undefined && pi === 0 && (
                    <span className="text-[10px] text-[var(--text-muted)] bg-[var(--bg-elevated)] px-1.5 py-0.5 rounded font-mono">
                      {(result.score * 100).toFixed(0)}%
                    </span>
                  )}
                  {result.source_dataset && pi === 0 && (
                    <span className="text-[10px] text-[var(--text-muted)] flex items-center gap-0.5">
                      <Database size={9} /> {result.source_dataset}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Body */}
          <div className="px-4 py-3">
            <ProcedureBlock proc={proc} index={pi} />
          </div>
        </div>
      ))}
    </>
  );
}
