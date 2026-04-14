"use client";

/**
 * EmbeddingConfigStep Component
 *
 * Configuration step for embedding service settings.
 * Editable form for provider, model, dimensions, endpoint, and API key.
 */

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Eye, EyeOff, AlertCircle, CheckCircle2 } from "lucide-react";
import type { EmbeddingFormData } from "@/hooks/use-config-wizard";
import type { DetailedHealthResponse } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface EmbeddingConfigStepProps {
  healthData?: DetailedHealthResponse;
  isLoading?: boolean;
  formData?: Partial<EmbeddingFormData>;
  onFormChange?: (data: Partial<EmbeddingFormData>) => void;
  isSaving?: boolean;
  error?: string | null;
  className?: string;
}

// ============================================================================
// Embedding Providers
// ============================================================================

interface EmbeddingProvider {
  id: string;
  name: string;
  defaultModel: string;
  defaultDimensions: number;
  requiresApiKey: boolean;
  requiresEndpoint: boolean;
  defaultEndpoint?: string;
  modelHint: string;
}

const EMBEDDING_PROVIDERS: EmbeddingProvider[] = [
  {
    id: "openai",
    name: "OpenAI",
    defaultModel: "text-embedding-3-large",
    defaultDimensions: 3072,
    requiresApiKey: true,
    requiresEndpoint: false,
    modelHint: "text-embedding-3-large (3072d), text-embedding-3-small (1536d)",
  },
  {
    id: "ollama",
    name: "Ollama (Local)",
    defaultModel: "nomic-embed-text",
    defaultDimensions: 768,
    requiresApiKey: false,
    requiresEndpoint: true,
    defaultEndpoint: "http://localhost:11434",
    modelHint: "nomic-embed-text, all-minilm, mxbai-embed-large",
  },
  {
    id: "fastembed",
    name: "FastEmbed (Local)",
    defaultModel: "BAAI/bge-small-en-v1.5",
    defaultDimensions: 384,
    requiresApiKey: false,
    requiresEndpoint: false,
    modelHint: "BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5",
  },
  {
    id: "azure",
    name: "Azure OpenAI",
    defaultModel: "text-embedding-3-large",
    defaultDimensions: 3072,
    requiresApiKey: true,
    requiresEndpoint: true,
    modelHint: "text-embedding-3-large, text-embedding-3-small",
  },
];

// ============================================================================
// Connection Status Component
// ============================================================================

function ConnectionStatus({ healthData }: { healthData?: DetailedHealthResponse }) {
  if (!healthData?.probes?.embedding_service) return null;

  const probe = healthData.probes.embedding_service;
  const isConnected = probe.verdict === "up";
  const isWarning = probe.verdict === "warn";

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-3 py-2 rounded-lg text-[12px]",
        isConnected && "bg-emerald-950/20 text-emerald-400",
        isWarning && "bg-amber-950/20 text-amber-400",
        !isConnected && !isWarning && "bg-red-950/20 text-red-400"
      )}
    >
      {isConnected ? (
        <CheckCircle2 size={14} />
      ) : (
        <AlertCircle size={14} />
      )}
      <span className="font-medium">
        {isConnected ? "Connected" : isWarning ? "Degraded" : "Not connected"}
      </span>
      {probe.backend !== "unknown" && (
        <>
          <span className="text-zinc-600">·</span>
          <span className="text-zinc-400">{probe.backend}</span>
        </>
      )}
      {probe.latency_ms > 0 && (
        <>
          <span className="text-zinc-600">·</span>
          <span className="text-zinc-400">{probe.latency_ms}ms</span>
        </>
      )}
    </div>
  );
}

// ============================================================================
// Form Field Component (matches LLMConfigStep)
// ============================================================================

interface FieldProps {
  label: string;
  children: React.ReactNode;
  hint?: string;
  required?: boolean;
}

function Field({ label, children, hint, required }: FieldProps) {
  return (
    <div className="space-y-1.5">
      <label className="flex items-center gap-1 text-[12px] font-medium text-zinc-300">
        {label}
        {required && <span className="text-red-400">*</span>}
      </label>
      {children}
      {hint && <p className="text-[11px] text-zinc-500">{hint}</p>}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function EmbeddingConfigStep({
  healthData,
  isLoading = false,
  formData,
  onFormChange,
  isSaving = false,
  error,
  className,
}: EmbeddingConfigStepProps) {
  const [showApiKey, setShowApiKey] = useState(false);

  const currentProvider = EMBEDDING_PROVIDERS.find((p) => p.id === formData?.provider);

  const handleProviderChange = useCallback(
    (providerId: string) => {
      const provider = EMBEDDING_PROVIDERS.find((p) => p.id === providerId);
      onFormChange?.({
        ...formData,
        provider: providerId,
        model: provider?.defaultModel || "",
        dimensions: provider?.defaultDimensions,
        endpoint: provider?.defaultEndpoint || "",
        apiKey: provider?.requiresApiKey ? formData?.apiKey : "",
      });
    },
    [formData, onFormChange]
  );

  const handleChange = useCallback(
    (field: keyof EmbeddingFormData, value: string | number) => {
      onFormChange?.({ ...formData, [field]: value });
    },
    [formData, onFormChange]
  );

  const inputClass = cn(
    "w-full px-3 py-2 rounded-lg text-[13px]",
    "bg-zinc-900 border border-zinc-700",
    "text-zinc-100 placeholder-zinc-500",
    "focus:outline-none focus:ring-2 focus:ring-zinc-500 focus:border-transparent",
    "disabled:opacity-50 disabled:cursor-not-allowed"
  );

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div>
        <h3 className="text-[18px] font-semibold text-zinc-100">
          Embedding Configuration
        </h3>
        <p className="text-[13px] text-zinc-500 mt-1">
          Configure the embedding model for vector search and knowledge extraction
        </p>
      </div>

      {/* Connection Status */}
      <ConnectionStatus healthData={healthData} />

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-950/20 text-red-400 text-[12px]">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      {/* Form */}
      <div className="space-y-5">
        {/* Provider & Model */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Field label="Provider" required>
            <select
              value={formData?.provider || "openai"}
              onChange={(e) => handleProviderChange(e.target.value)}
              disabled={isSaving}
              className={inputClass}
            >
              {EMBEDDING_PROVIDERS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </Field>

          <Field
            label="Model"
            required
            hint={currentProvider?.modelHint}
          >
            <input
              type="text"
              value={formData?.model || currentProvider?.defaultModel || "text-embedding-3-large"}
              onChange={(e) => handleChange("model", e.target.value)}
              placeholder={currentProvider?.defaultModel || "text-embedding-3-large"}
              disabled={isSaving}
              className={inputClass}
            />
          </Field>
        </div>

        {/* Dimensions */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Field label="Dimensions" hint="Vector dimensions (auto-detected if left empty)">
            <input
              type="number"
              value={formData?.dimensions ?? currentProvider?.defaultDimensions ?? ""}
              onChange={(e) => handleChange("dimensions", parseInt(e.target.value) || 0)}
              placeholder={String(currentProvider?.defaultDimensions || 3072)}
              disabled={isSaving}
              className={inputClass}
            />
          </Field>

          {/* Endpoint - show for providers that need it */}
          {currentProvider?.requiresEndpoint && (
            <Field label="Endpoint URL" required={currentProvider.requiresEndpoint}>
              <input
                type="text"
                value={formData?.endpoint || ""}
                onChange={(e) => handleChange("endpoint", e.target.value)}
                placeholder={currentProvider.defaultEndpoint || "https://api.example.com/v1"}
                disabled={isSaving}
                className={inputClass}
              />
            </Field>
          )}
        </div>

        {/* API Key */}
        {currentProvider?.requiresApiKey && (
          <Field label="API Key" hint="Leave empty to use the same key as your LLM provider">
            <div className="relative">
              <input
                type={showApiKey ? "text" : "password"}
                value={formData?.apiKey || ""}
                onChange={(e) => handleChange("apiKey", e.target.value)}
                placeholder={`Enter your ${currentProvider.name} API key`}
                disabled={isSaving}
                className={cn(inputClass, "pr-10")}
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-zinc-500 hover:text-zinc-300"
                tabIndex={-1}
              >
                {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </Field>
        )}
      </div>
    </div>
  );
}

EmbeddingConfigStep.displayName = "EmbeddingConfigStep";

export default EmbeddingConfigStep;
