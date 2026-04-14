"use client";

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { HelpCircle, ChevronRight, RotateCcw } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface WizardResult {
  enableContentRouting?: boolean;
  enableEpisodeRouting?: boolean;
  preciseMode?: boolean;
  contentType?: "text" | "dialog";
}

type StepId = "idle" | "q1" | "q1-confirm" | "q2" | "q2-confirm" | "q3" | "q3-confirm" | "q4" | "done";

interface WizardState {
  step: StepId;
  result: WizardResult;
}

// ============================================================================
// Step definitions
// ============================================================================

interface StepDef {
  question: string;
  detail: React.ReactNode;
  yesLabel?: string;
  noLabel?: string;
}

const STEPS: Record<string, StepDef> = {
  q1: {
    question: "Does each ingested ContentFragment contain more than one topic?",
    detail:
      "A ContentFragment is the text chunk that M-Flow processes as one unit. " +
      "If a single fragment covers multiple distinct topics or semantic focuses " +
      "(e.g. a meeting transcript mixing budget and hiring discussions), " +
      "Content Routing can separate them into cleaner memory structures.",
  },
  "q1-confirm": {
    question: "Enable Content Routing?",
    detail: (
      <>
        This classifies each sentence by topic within a fragment, producing cleaner Episode boundaries.{" "}
        <strong className="text-[var(--text-primary)]">
          It costs one additional LLM call per multi-sentence fragment and increases ingestion time.
        </strong>
      </>
    ),
    yesLabel: "Yes, enable",
    noLabel: "No, skip",
  },
  q2: {
    question: "Does your content evolve continuously across multiple ingestion batches?",
    detail:
      "For example: a project discussion spanning several meetings, or daily logs about " +
      "the same ongoing event. If related content arrives in different ContentFragments " +
      "across separate ingestion calls, Episode Routing can merge them into coherent Episodes " +
      "instead of creating isolated ones.",
  },
  "q2-confirm": {
    question: "Enable Episode Routing?",
    detail: (
      <>
        This merges new content into existing related Episodes via vector search + LLM decision.{" "}
        <strong className="text-[var(--text-primary)]">
          It costs extra LLM tokens per document and changes ingestion from concurrent to sequential Episode creation, increasing ingestion time.
        </strong>
      </>
    ),
    yesLabel: "Yes, enable",
    noLabel: "No, skip",
  },
  q3: {
    question: "Must ALL factual details be preserved with zero compression loss?",
    detail:
      "By default, M-Flow summarizes content to build concise memory structures. " +
      "If your content contains critical data points (dates, amounts, clause numbers, KPIs) " +
      "where any omission is unacceptable, Precise Mode keeps everything intact.",
  },
  "q3-confirm": {
    question: "Enable Precise Mode?",
    detail: (
      <>
        This uses a two-step summarization pipeline that preserves every date, number, name, and constraint.{" "}
        <strong className="text-[var(--text-primary)]">
          It increases LLM token usage and ingestion time.
        </strong>
      </>
    ),
    yesLabel: "Yes, enable",
    noLabel: "No, skip",
  },
  q4: {
    question: "Is the content in chat/conversation format?",
    detail:
      "Chat logs, meeting transcripts, IM records, and interview transcripts have a " +
      "\"Speaker: message\" structure. Selecting Dialog mode splits text by speaker turns " +
      "instead of sentence boundaries, preserving conversational context.",
  },
};

// ============================================================================
// Navigation logic
// ============================================================================

function nextStep(current: StepId, answer: boolean, result: WizardResult): { step: StepId; result: WizardResult } {
  switch (current) {
    case "q1":
      if (answer) return { step: "q1-confirm", result };
      return { step: "q2", result: { ...result, enableContentRouting: false } };

    case "q1-confirm":
      return {
        step: "q2",
        result: { ...result, enableContentRouting: answer },
      };

    case "q2":
      if (answer) return { step: "q2-confirm", result };
      return { step: "q3", result: { ...result, enableEpisodeRouting: false } };

    case "q2-confirm":
      return {
        step: "q3",
        result: { ...result, enableEpisodeRouting: answer },
      };

    case "q3":
      if (answer) return { step: "q3-confirm", result };
      return { step: "q4", result: { ...result, preciseMode: false } };

    case "q3-confirm":
      return {
        step: "q4",
        result: { ...result, preciseMode: answer },
      };

    case "q4":
      return {
        step: "done",
        result: { ...result, contentType: answer ? "dialog" : "text" },
      };

    default:
      return { step: "done", result };
  }
}

// ============================================================================
// Summary renderer
// ============================================================================

function resultSummary(r: WizardResult): string {
  const parts: string[] = [];
  if (r.enableContentRouting) parts.push("Content Routing: ON");
  if (r.enableContentRouting === false) parts.push("Content Routing: OFF");
  if (r.enableEpisodeRouting) parts.push("Episode Routing: ON");
  if (r.enableEpisodeRouting === false) parts.push("Episode Routing: OFF");
  if (r.preciseMode) parts.push("Precise Mode: ON");
  if (r.preciseMode === false) parts.push("Precise Mode: OFF");
  if (r.contentType === "dialog") parts.push("Content Type: Dialog");
  if (r.contentType === "text") parts.push("Content Type: Text");
  return parts.join("  \u00B7  ");
}

// ============================================================================
// Component
// ============================================================================

export function ConfigWizard({
  onApply,
  disabled = false,
}: {
  onApply: (result: WizardResult) => void;
  disabled?: boolean;
}) {
  const [state, setState] = useState<WizardState>({ step: "idle", result: {} });

  const handleAnswer = useCallback(
    (answer: boolean) => {
      const next = nextStep(state.step, answer, state.result);
      setState(next);
      if (next.step === "done") {
        onApply(next.result);
      }
    },
    [state, onApply]
  );

  const handleReset = useCallback(() => {
    setState({ step: "idle", result: {} });
  }, []);

  const handleStart = useCallback(() => {
    setState({ step: "q1", result: {} });
  }, []);

  // Idle state — show entry button
  if (state.step === "idle") {
    return (
      <div className="pb-3 mb-3 border-b border-[var(--border-subtle)]">
        <button
          onClick={handleStart}
          disabled={disabled}
          className={cn(
            "flex items-center gap-2 w-full px-3 py-2.5 text-left text-[11px] rounded border transition-colors",
            "bg-[var(--bg-elevated)] border-[var(--border-subtle)] text-[var(--text-muted)]",
            "hover:border-[var(--text-muted)] hover:text-[var(--text-secondary)]",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          <HelpCircle size={14} className="flex-shrink-0" />
          <div>
            <div className="text-[var(--text-secondary)]">Not sure which settings to use?</div>
            <div className="text-[10px] mt-0.5">Answer a few questions to configure automatically</div>
          </div>
          <ChevronRight size={14} className="ml-auto flex-shrink-0" />
        </button>
      </div>
    );
  }

  // Done state — show summary
  if (state.step === "done") {
    return (
      <div className="pb-3 mb-3 border-b border-[var(--border-subtle)]">
        <div className="px-3 py-2.5 bg-emerald-500/5 border border-emerald-500/20 rounded space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-emerald-400 uppercase tracking-wider">Configuration applied</span>
            <button
              onClick={handleReset}
              disabled={disabled}
              className="flex items-center gap-1 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
            >
              <RotateCcw size={10} />
              Redo
            </button>
          </div>
          <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
            {resultSummary(state.result)}
          </p>
        </div>
      </div>
    );
  }

  // Active question
  const stepDef = STEPS[state.step];
  if (!stepDef) return null;

  const isConfirmStep = state.step.endsWith("-confirm");

  return (
    <div className="pb-3 mb-3 border-b border-[var(--border-subtle)]">
      <div className="px-3 py-3 bg-blue-500/5 border border-blue-500/20 rounded space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-blue-400 uppercase tracking-wider">Configuration assistant</span>
          <button
            onClick={handleReset}
            disabled={disabled}
            className="flex items-center gap-1 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            <RotateCcw size={10} />
            Start over
          </button>
        </div>

        {/* Question */}
        <p className="text-[12px] text-[var(--text-primary)] font-medium leading-snug">
          {stepDef.question}
        </p>

        {/* Detail */}
        <p className="text-[10px] text-[var(--text-muted)] leading-relaxed">
          {stepDef.detail}
        </p>

        {/* Buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => handleAnswer(true)}
            disabled={disabled}
            className={cn(
              "flex-1 px-3 py-1.5 text-[11px] rounded border transition-colors",
              isConfirmStep
                ? "bg-blue-500/10 border-blue-500/30 text-blue-400 hover:bg-blue-500/20"
                : "bg-[var(--bg-elevated)] border-[var(--border-subtle)] text-[var(--text-secondary)] hover:border-[var(--text-muted)]",
              disabled && "opacity-50 cursor-not-allowed"
            )}
          >
            {stepDef.yesLabel || "Yes"}
          </button>
          <button
            onClick={() => handleAnswer(false)}
            disabled={disabled}
            className={cn(
              "flex-1 px-3 py-1.5 text-[11px] rounded border transition-colors",
              "bg-[var(--bg-elevated)] border-[var(--border-subtle)] text-[var(--text-secondary)] hover:border-[var(--text-muted)]",
              disabled && "opacity-50 cursor-not-allowed"
            )}
          >
            {stepDef.noLabel || "No"}
          </button>
        </div>
      </div>
    </div>
  );
}
