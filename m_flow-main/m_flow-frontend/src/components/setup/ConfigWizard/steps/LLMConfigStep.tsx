"use client";

/**
 * LLMConfigStep Component
 *
 * Configuration step for LLM provider settings.
 * Streamlined interface for configuring language model providers.
 *
 * Features:
 * - Provider selection with intelligent defaults
 * - Manual model input for flexibility
 * - API key with secure input
 * - Real-time connection status
 */

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Eye, EyeOff, AlertCircle, CheckCircle2 } from "lucide-react";
import type { LLMFormData } from "@/hooks/use-config-wizard";
import type { DetailedHealthResponse } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface LLMConfigStepProps {
  formData: Partial<LLMFormData>;
  onFormChange: (data: Partial<LLMFormData>) => void;
  healthData?: DetailedHealthResponse;
  isSaving?: boolean;
  error?: string | null;
  className?: string;
}

// ============================================================================
// Provider Configurations
// ============================================================================

interface ProviderConfig {
  id: string;
  name: string;
  defaultModel: string;
  requiresApiKey: boolean;
  requiresEndpoint: boolean;
  defaultEndpoint?: string;
  modelHint: string;
}

// Default model - fastest and cheapest for knowledge extraction
const BENCHMARK_MODEL = "gpt-5-nano";

const PROVIDERS: ProviderConfig[] = [
  {
    id: "openai",
    name: "OpenAI",
    defaultModel: BENCHMARK_MODEL,
    requiresApiKey: true,
    requiresEndpoint: false,
    modelHint: "gpt-5-nano (default), gpt-5-mini, gpt-4o-mini",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    defaultModel: "claude-haiku-4-5",
    requiresApiKey: true,
    requiresEndpoint: false,
    modelHint: "claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6",
  },
  {
    id: "gemini",
    name: "Google Gemini",
    defaultModel: "gemini-2.5-flash-lite",
    requiresApiKey: true,
    requiresEndpoint: false,
    modelHint: "gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro",
  },
  {
    id: "mistral",
    name: "Mistral AI",
    defaultModel: "ministral-8b-2512",
    requiresApiKey: true,
    requiresEndpoint: false,
    modelHint: "ministral-3b/8b, mistral-small-3.2, mistral-large-2512",
  },
  {
    id: "ollama",
    name: "Ollama (Local)",
    defaultModel: "qwen3:8b",
    requiresApiKey: false,
    requiresEndpoint: true,
    defaultEndpoint: "http://localhost:11434",
    modelHint: "qwen3:8b, phi4, llama3.2:3b, deepseek-r1:8b",
  },
  {
    id: "bedrock",
    name: "AWS Bedrock",
    defaultModel: "anthropic.claude-haiku-4-5-v1:0",
    requiresApiKey: true,
    requiresEndpoint: true,
    modelHint: "anthropic.claude-haiku-4-5-v1:0, amazon.nova-lite-v1:0",
  },
  {
    id: "custom",
    name: "Custom / LiteLLM",
    defaultModel: "deepseek/deepseek-chat",
    requiresApiKey: true,
    requiresEndpoint: true,
    modelHint: "provider/model format (deepseek/deepseek-chat)",
  },
];

const DEFAULT_PROVIDER = "openai";
const DEFAULT_MODEL = BENCHMARK_MODEL;

// ============================================================================
// Connection Status Component
// ============================================================================

function ConnectionStatus({ healthData }: { healthData?: DetailedHealthResponse }) {
  if (!healthData?.probes?.llm_provider) return null;

  const probe = healthData.probes.llm_provider;
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
// Form Fields
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

export function LLMConfigStep({
  formData,
  onFormChange,
  healthData,
  isSaving = false,
  error,
  className,
}: LLMConfigStepProps) {
  const [showApiKey, setShowApiKey] = useState(false);

  const currentProvider = PROVIDERS.find((p) => p.id === formData.provider);

  const handleProviderChange = useCallback(
    (providerId: string) => {
      const provider = PROVIDERS.find((p) => p.id === providerId);
      onFormChange({
        ...formData,
        provider: providerId,
        model: provider?.defaultModel || "",
        endpoint: provider?.defaultEndpoint || "",
        apiKey: provider?.requiresApiKey ? formData.apiKey : "",
      });
    },
    [formData, onFormChange]
  );

  const handleChange = useCallback(
    (field: keyof LLMFormData, value: string) => {
      onFormChange({ ...formData, [field]: value });
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
          LLM Configuration
        </h3>
        <p className="text-[13px] text-zinc-500 mt-1">
          Configure your language model for knowledge extraction
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
        {/* Provider & Model - Side by side on larger screens */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Field label="Provider" required>
            <select
              value={formData.provider || DEFAULT_PROVIDER}
              onChange={(e) => handleProviderChange(e.target.value)}
              disabled={isSaving}
              className={inputClass}
            >
              {PROVIDERS.map((p) => (
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
            <div className="space-y-1.5">
              <div className="relative">
                <input
                  type="text"
                  value={formData.model || currentProvider?.defaultModel || DEFAULT_MODEL}
                  onChange={(e) => handleChange("model", e.target.value)}
                  placeholder={currentProvider?.defaultModel || DEFAULT_MODEL}
                  disabled={isSaving}
                  className={inputClass}
                />
                {(formData.model || currentProvider?.defaultModel) === BENCHMARK_MODEL && (
                  <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    Benchmark
                  </span>
                )}
              </div>
            </div>
          </Field>
        </div>

        {/* API Key */}
        {currentProvider?.requiresApiKey && (
          <Field label="API Key" required hint="Encrypted and stored securely">
            <div className="relative">
              <input
                type={showApiKey ? "text" : "password"}
                value={formData.apiKey || ""}
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

        {/* Endpoint */}
        {currentProvider?.requiresEndpoint && (
          <Field
            label="Endpoint URL"
            required={currentProvider.requiresEndpoint}
          >
            <input
              type="text"
              value={formData.endpoint || ""}
              onChange={(e) => handleChange("endpoint", e.target.value)}
              placeholder={currentProvider.defaultEndpoint || "https://api.example.com/v1"}
              disabled={isSaving}
              className={inputClass}
            />
          </Field>
        )}
      </div>
    </div>
  );
}

LLMConfigStep.displayName = "LLMConfigStep";

export default LLMConfigStep;
