"use client";

/**
 * WizardContainer Component
 *
 * Main container for the configuration wizard.
 * Orchestrates step rendering, navigation, and state management.
 *
 * Features:
 * - Step progress indicator
 * - Dynamic step content rendering
 * - Navigation controls (back/next/skip)
 * - Form state management via useConfigWizard hook
 * - Completion celebration
 *
 * @example
 * <WizardContainer onComplete={() => console.log("Setup complete!")} />
 */

import React, { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { RefreshCw, CheckCircle2, PartyPopper } from "lucide-react";

import { useConfigWizard } from "@/hooks/use-config-wizard";
import { WizardProgress } from "./WizardProgress";
import { WizardNavigation } from "./WizardNavigation";
import {
  LLMConfigStep,
  EmbeddingConfigStep,
  DatabaseConfigStep,
  StorageConfigStep,
  ReviewStep,
} from "./steps";
import type { ConfigStepId } from "@/types/setup";

// ============================================================================
// Types
// ============================================================================

export interface WizardContainerProps {
  /** Callback when wizard is completed */
  onComplete?: () => void;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Completion Component
// ============================================================================

interface CompletionViewProps {
  onReset: () => void;
}

function CompletionView({ onReset }: CompletionViewProps) {
  return (
    <div className="text-center py-12 space-y-6">
      {/* Celebration Icon */}
      <div className="flex justify-center">
        <div className="relative">
          <div className="w-20 h-20 rounded-full bg-emerald-500/10 border-2 border-emerald-500/30 flex items-center justify-center">
            <CheckCircle2 size={40} className="text-emerald-400" />
          </div>
          <PartyPopper
            size={24}
            className="absolute -top-1 -right-1 text-amber-400"
          />
        </div>
      </div>

      {/* Message */}
      <div className="space-y-2">
        <h3 className="text-[20px] font-semibold text-zinc-100">
          Setup Complete!
        </h3>
        <p className="text-[14px] text-zinc-500 max-w-md mx-auto">
          Your M-Flow instance is configured and ready to use.
          Start ingesting documents to build your knowledge graph.
        </p>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={onReset}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg",
            "text-[13px] font-medium text-zinc-400",
            "bg-zinc-900 border border-zinc-800",
            "hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
          )}
        >
          <RefreshCw size={14} />
          Run Setup Again
        </button>
        <a
          href="/memorize"
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg",
            "text-[13px] font-medium",
            "bg-zinc-100 text-zinc-900",
            "hover:bg-white transition-colors"
          )}
        >
          Start Ingesting
        </a>
      </div>
    </div>
  );
}

// ============================================================================
// Step Content Renderer
// ============================================================================

interface StepContentProps {
  stepId: ConfigStepId;
  wizard: ReturnType<typeof useConfigWizard>;
}

function StepContent({ stepId, wizard }: StepContentProps) {
  switch (stepId) {
    case "llm":
      return (
        <LLMConfigStep
          formData={wizard.llmFormData}
          onFormChange={wizard.setLLMFormData}
          healthData={wizard.healthData}
          isSaving={wizard.isSaving}
          error={wizard.lastError}
        />
      );

    case "embedding":
      return (
        <EmbeddingConfigStep
          healthData={wizard.healthData}
          isLoading={wizard.isLoadingHealth}
          formData={wizard.embeddingFormData}
          onFormChange={wizard.setEmbeddingFormData}
          isSaving={wizard.isSaving}
          error={wizard.lastError}
        />
      );

    case "database":
      return (
        <DatabaseConfigStep
          healthData={wizard.healthData}
          isLoading={wizard.isLoadingHealth}
          formData={wizard.vectorDBFormData}
          onFormChange={wizard.setVectorDBFormData}
          isSaving={wizard.isSaving}
          error={wizard.lastError}
        />
      );

    case "storage":
      return (
        <StorageConfigStep
          healthData={wizard.healthData}
          isLoading={wizard.isLoadingHealth}
        />
      );

    case "review":
      return (
        <ReviewStep
          healthData={wizard.healthData}
          isLoadingHealth={wizard.isLoadingHealth}
          onRunTests={wizard.runTests}
          isTesting={wizard.isTesting}
        />
      );

    default:
      return (
        <div className="p-8 text-center text-zinc-500">
          Unknown step: {stepId}
        </div>
      );
  }
}

// ============================================================================
// Main Component
// ============================================================================

export function WizardContainer({ onComplete, className }: WizardContainerProps) {
  const wizard = useConfigWizard();

  // Handle next button click based on current step
  const handleNext = useCallback(async () => {
    const currentStepId = wizard.currentStep.id;

    if (currentStepId === "llm") {
      if (wizard.llmFormData.provider && wizard.llmFormData.model) {
        const success = await wizard.saveLLMConfig();
        if (!success) return;
      }
    }

    if (currentStepId === "embedding") {
      if (wizard.embeddingFormData.provider && wizard.embeddingFormData.model) {
        const success = await wizard.saveEmbeddingConfig();
        if (!success) return;
      }
    }

    if (currentStepId === "database") {
      if (wizard.vectorDBFormData.provider) {
        const success = await wizard.saveVectorDBConfig();
        if (!success) return;
      }
    }

    // Move to next step
    wizard.goNext();

    // Call onComplete if this was the last step
    if (wizard.isLastStep) {
      onComplete?.();
    }
  }, [wizard, onComplete]);

  // Handle skip (for optional steps)
  const handleSkip = useCallback(() => {
    wizard.skipStep();
  }, [wizard]);

  // If wizard is complete, show completion view
  if (wizard.isComplete) {
    return (
      <div className={cn("", className)}>
        <CompletionView onReset={wizard.reset} />
      </div>
    );
  }

  // Determine if current step is optional
  const canSkip = wizard.currentStep.isOptional;

  // Determine next button label and disabled state
  const getNextConfig = () => {
    const stepId = wizard.currentStep.id;

    if (stepId === "llm") {
      const hasRequiredData =
        wizard.llmFormData.provider && wizard.llmFormData.model;
      return {
        label: hasRequiredData ? "Save & Continue" : "Skip",
        disabled: false,
      };
    }

    if (stepId === "embedding") {
      const hasData = wizard.embeddingFormData.provider && wizard.embeddingFormData.model;
      return {
        label: hasData ? "Save & Continue" : "Skip",
        disabled: false,
      };
    }

    if (stepId === "database") {
      const hasData = wizard.vectorDBFormData.provider;
      return {
        label: hasData ? "Save & Continue" : "Skip",
        disabled: false,
      };
    }

    if (wizard.isLastStep) {
      return {
        label: "Complete Setup",
        disabled: false,
      };
    }

    return {
      label: "Continue",
      disabled: false,
    };
  };

  const nextConfig = getNextConfig();

  const guidePages = [
    {
      title: "LLM Provider",
      description: "M-Flow uses an LLM for knowledge extraction, entity recognition, and summarization.",
      envVars: [
        { key: "LLM_PROVIDER", value: "openai", note: "Options: openai, anthropic, ollama, gemini, mistral, bedrock" },
        { key: "LLM_MODEL", value: "gpt-5-nano", note: "Default model — fastest and cheapest for ingestion" },
        { key: "LLM_API_KEY", value: "sk-your-key-here", note: "Required. Get from platform.openai.com" },
      ],
      statusKey: "llm_provider",
    },
    {
      title: "Embedding Service",
      description: "Embedding models convert text into vectors for semantic search. Uses the same API key as LLM by default.",
      envVars: [
        { key: "EMBEDDING_PROVIDER", value: "openai", note: "Default: openai" },
        { key: "EMBEDDING_MODEL", value: "openai/text-embedding-3-large", note: "Default: text-embedding-3-large (3072d)" },
        { key: "EMBEDDING_DIMENSIONS", value: "3072", note: "Must match model output dimensions" },
        { key: "EMBEDDING_API_KEY", value: "sk-your-key-here", note: "Optional — falls back to LLM_API_KEY" },
      ],
      statusKey: "embedding_service",
    },
    {
      title: "Vector Database",
      description: "Stores vector embeddings for semantic search. LanceDB works locally with zero configuration.",
      envVars: [
        { key: "VECTOR_DB_PROVIDER", value: "lancedb", note: "Default: lancedb (local, no setup needed)" },
      ],
      statusKey: "vector_db",
    },
    {
      title: "Storage & Graph",
      description: "Graph database stores knowledge relationships. SQLite and KuzuDB work locally with zero configuration.",
      envVars: [
        { key: "DB_PROVIDER", value: "sqlite", note: "Default: sqlite (local)" },
        { key: "GRAPH_DATABASE_PROVIDER", value: "kuzu", note: "Default: kuzu (local)" },
      ],
      statusKey: "graph_db",
    },
    {
      title: "Start & Verify",
      description: "After editing .env, start the backend and verify all services are connected.",
      command: "python -m uvicorn m_flow.api.client:app --host localhost --port 8000",
      isVerifyStep: true,
    },
  ];

  const [pageIndex, setPageIndex] = useState(0);
  const page = guidePages[pageIndex];

  const probe = page.statusKey && wizard.healthData?.probes
    ? wizard.healthData.probes[page.statusKey as keyof typeof wizard.healthData.probes]
    : null;

  const allProbes = wizard.healthData?.probes
    ? Object.entries(wizard.healthData.probes)
    : [];

  const envBlock = page.envVars
    ? page.envVars.map((v) => `${v.key}=${v.value}`).join("\n")
    : "";

  return (
    <div className={cn("space-y-6", className)}>
      {/* Progress dots */}
      <div className="flex items-center justify-center gap-2">
        {guidePages.map((g, i) => (
          <button
            key={i}
            onClick={() => setPageIndex(i)}
            className={cn(
              "w-2 h-2 rounded-full transition-all",
              i === pageIndex ? "bg-[var(--text-primary)] w-6" : "bg-[var(--border-default)]"
            )}
          />
        ))}
      </div>

      {/* Page content */}
      <div className="min-h-[350px] p-6 rounded-xl bg-zinc-900/30 border border-zinc-800">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-[16px] font-semibold text-[var(--text-primary)]">{page.title}</h3>
          {probe && (
            <span className={cn(
              "text-[11px] px-2 py-0.5 rounded",
              probe.verdict === "up" ? "bg-emerald-950/20 text-emerald-400" : "bg-amber-950/20 text-amber-400"
            )}>
              {probe.verdict === "up" ? "Connected" : probe.note?.slice(0, 30) || "Not connected"}
            </span>
          )}
        </div>
        <p className="text-[12px] text-[var(--text-muted)] mb-5">{page.description}</p>

        {/* Env vars */}
        {page.envVars && (
          <div className="space-y-3 mb-4">
            {page.envVars.map((v) => (
              <div key={v.key} className="flex items-start gap-3 p-2.5 bg-[var(--bg-elevated)] rounded border border-[var(--border-subtle)]">
                <code className="text-[11px] text-emerald-400 font-mono whitespace-nowrap">{v.key}</code>
                <div className="flex-1 min-w-0">
                  <code className="text-[11px] text-[var(--text-secondary)] font-mono">{v.value}</code>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{v.note}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Copy block */}
        {envBlock && (
          <div className="relative group">
            <pre className="bg-[#0a0a0a] border border-[var(--border-subtle)] rounded p-3 text-[11px] text-[var(--text-secondary)] font-mono overflow-x-auto">
              {envBlock}
            </pre>
            <button
              onClick={() => {
                navigator.clipboard.writeText(envBlock);
                import("sonner").then(({ toast }) => toast.success("Copied"));
              }}
              className="absolute top-2 right-2 px-2 py-1 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded opacity-0 group-hover:opacity-100 transition-opacity"
            >
              Copy
            </button>
          </div>
        )}

        {/* Command */}
        {page.command && (
          <div className="relative group mb-4">
            <pre className="bg-[#0a0a0a] border border-[var(--border-subtle)] rounded p-3 text-[11px] text-[var(--text-secondary)] font-mono">
              {page.command}
            </pre>
            <button
              onClick={() => {
                navigator.clipboard.writeText(page.command!);
                import("sonner").then(({ toast }) => toast.success("Copied"));
              }}
              className="absolute top-2 right-2 px-2 py-1 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded opacity-0 group-hover:opacity-100 transition-opacity"
            >
              Copy
            </button>
          </div>
        )}

        {/* Verify step: show all probes */}
        {page.isVerifyStep && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-4">
            {allProbes.map(([key, p]) => (
              <div
                key={key}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 rounded text-[11px] border",
                  p.verdict === "up"
                    ? "bg-emerald-950/10 border-emerald-900/30 text-emerald-400"
                    : p.verdict === "warn"
                      ? "bg-amber-950/10 border-amber-900/30 text-amber-400"
                      : "bg-red-950/10 border-red-900/30 text-red-400"
                )}
              >
                <span className="font-medium">{key.replace(/_/g, " ")}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
          disabled={pageIndex === 0}
          className={cn(
            "flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors",
            pageIndex === 0 && "opacity-30 cursor-not-allowed"
          )}
        >
          &lt; Back
        </button>
        <span className="text-[10px] text-[var(--text-muted)]">{pageIndex + 1} / {guidePages.length}</span>
        <button
          onClick={() => setPageIndex(Math.min(guidePages.length - 1, pageIndex + 1))}
          disabled={pageIndex === guidePages.length - 1}
          className={cn(
            "flex items-center gap-1 text-xs text-[var(--text-primary)] hover:opacity-80 transition-opacity",
            pageIndex === guidePages.length - 1 && "opacity-30 cursor-not-allowed"
          )}
        >
          Next &gt;
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Display Name
// ============================================================================

WizardContainer.displayName = "WizardContainer";

// ============================================================================
// Default Export
// ============================================================================

export default WizardContainer;
