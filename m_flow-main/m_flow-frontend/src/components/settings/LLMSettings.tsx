"use client";

import React, { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod/v4";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Sparkles, Save, RotateCcw, AlertCircle } from "lucide-react";
import { LLMConfig } from "@/types";

const llmConfigSchema = z.object({
  llm_provider: z.string().min(1, "Please select an LLM provider"),
  llm_model: z.string().min(1, "Please enter a model name"),
  llm_endpoint: z.string().optional(),
  llm_api_key: z.string().optional(),
  llm_temperature: z.number().min(0).max(2).optional(),
  llm_max_tokens: z.number().min(1).max(128000).optional(),
  llm_streaming: z.boolean().optional(),
  llm_rate_limit_requests: z.number().min(1).optional(),
  llm_rate_limit_input_tokens: z.number().min(1).optional(),
  llm_rate_limit_output_tokens: z.number().min(1).optional(),
  structured_output_framework: z.enum(["baml", "llm_instructor", "default"]).optional(),
});

type LLMConfigForm = z.infer<typeof llmConfigSchema>;

/**
 * LLM configuration format expected by backend API
 * Used for POST /api/v1/settings
 */
interface LLMSettingsPayload {
  provider: string;
  model: string;
  api_key?: string;
}

interface LLMSettingsProps {
  config?: LLMConfig | null;
  onSave: (config: LLMSettingsPayload) => void;
  isSaving?: boolean;
}

const providers = [
  { value: "openai", label: "OpenAI" },
  { value: "anthropic", label: "Anthropic" },
  { value: "gemini", label: "Google Gemini" },
  { value: "mistral", label: "Mistral AI" },
  { value: "ollama", label: "Ollama (Local)" },
  { value: "bedrock", label: "AWS Bedrock" },
  { value: "custom", label: "Custom" },
];

/**
 * List of configuration parameters not supported by backend
 * These parameters are for local reference only and won't be sent to backend
 */
const UNSUPPORTED_BY_BACKEND = [
  "llm_endpoint",
  "llm_temperature",
  "llm_max_tokens",
  "llm_streaming",
  "llm_rate_limit_requests",
  "llm_rate_limit_input_tokens",
  "llm_rate_limit_output_tokens",
  "structured_output_framework",
];

// Benchmark model for testing and evaluation
const BENCHMARK_MODEL = "gpt-5-mini";

const defaultConfig: LLMConfigForm = {
  llm_provider: "openai",
  llm_model: BENCHMARK_MODEL,
  llm_temperature: 0.7,
  llm_max_tokens: 4096,
  llm_streaming: true,
  llm_rate_limit_requests: 100,
  llm_rate_limit_input_tokens: 100000,
  llm_rate_limit_output_tokens: 50000,
  structured_output_framework: "default",
};

export function LLMSettings({ config, onSave, isSaving }: LLMSettingsProps) {
  const {
    register,
    handleSubmit,
    watch,
    setValue,
    reset,
    formState: { errors, isDirty },
  } = useForm<LLMConfigForm>({
    resolver: zodResolver(llmConfigSchema),
    defaultValues: defaultConfig,
  });

  // Update form values when config loads from backend
  useEffect(() => {
    if (config) {
      // Validate structured_output_framework value
      const validFrameworks = ["default", "baml", "llm_instructor"];
      const rawFramework = config.structured_output_framework;
      const framework = (validFrameworks.includes(rawFramework || "")
        ? rawFramework
        : "default") as "default" | "baml" | "llm_instructor";

      reset({
        llm_provider: config.llm_provider || defaultConfig.llm_provider,
        llm_model: config.llm_model || defaultConfig.llm_model,
        llm_endpoint: config.llm_endpoint || "",
        llm_api_key: "", // API key not shown for security
        llm_temperature: config.llm_temperature ?? defaultConfig.llm_temperature,
        llm_max_tokens: config.llm_max_tokens ?? defaultConfig.llm_max_tokens,
        llm_streaming: config.llm_streaming ?? defaultConfig.llm_streaming,
        llm_rate_limit_requests: config.llm_rate_limit_requests ?? defaultConfig.llm_rate_limit_requests,
        llm_rate_limit_input_tokens: config.llm_rate_limit_input_tokens ?? defaultConfig.llm_rate_limit_input_tokens,
        llm_rate_limit_output_tokens: config.llm_rate_limit_output_tokens ?? defaultConfig.llm_rate_limit_output_tokens,
        structured_output_framework: framework,
      });
    }
  }, [config, reset]);

  const temperature = watch("llm_temperature") ?? 0.7;
  const streaming = watch("llm_streaming") ?? true;

  const onSubmit = (data: LLMConfigForm) => {
    // Important: Backend API uses short field names (provider, model, api_key)
    // Field mapping is required, not passing form data directly
    const payload: LLMSettingsPayload = {
      provider: data.llm_provider,
      model: data.llm_model,
      api_key: data.llm_api_key || undefined,
    };
    onSave(payload);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle>LLM Configuration</CardTitle>
            <CardDescription>Configure large language model parameters</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Provider */}
            <div className="space-y-2">
              <Label>LLM Provider</Label>
              <Select
                value={watch("llm_provider")}
                onValueChange={(v) => setValue("llm_provider", v, { shouldDirty: true })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select provider" />
                </SelectTrigger>
                <SelectContent>
                  {providers.map((p) => (
                    <SelectItem key={p.value} value={p.value}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {errors.llm_provider && (
                <p className="text-xs text-red-400">{errors.llm_provider.message}</p>
              )}
            </div>

            {/* Model */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label>Model Name</Label>
                {watch("llm_model") === BENCHMARK_MODEL && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    Benchmark
                  </span>
                )}
              </div>
              <Input
                {...register("llm_model")}
                placeholder={BENCHMARK_MODEL}
                error={errors.llm_model?.message}
              />
              <p className="text-xs text-zinc-500">
                Recommended: <span className="text-emerald-400 font-medium">{BENCHMARK_MODEL}</span> (benchmark model)
              </p>
            </div>

            {/* API Endpoint */}
            <div className="space-y-2">
              <Label>API Endpoint (optional)</Label>
              <Input
                {...register("llm_endpoint")}
                placeholder="https://api.openai.com/v1"
              />
            </div>

            {/* API Key */}
            <div className="space-y-2">
              <Label>API Key (optional)</Label>
              <Input
                {...register("llm_api_key")}
                type="password"
                placeholder="sk-..."
              />
            </div>
          </div>

          {/* Unsupported parameters notice */}
          <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
            <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
            <div className="text-xs text-amber-500/90">
              <p className="font-medium mb-1">Parameters below are for local reference only</p>
              <p className="text-amber-500/70">
                Temperature, Max Tokens, rate limits are not supported by backend API and must be configured via .env file.
                API support for these parameters will be added in future versions.
              </p>
            </div>
          </div>

          {/* Temperature */}
          <div className="space-y-3 opacity-75">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Label>Temperature (Creativity)</Label>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-400">Local only</span>
              </div>
              <span className="text-sm text-zinc-400">{temperature.toFixed(2)}</span>
            </div>
            <Slider
              value={[temperature]}
              onValueChange={([v]) => setValue("llm_temperature", v, { shouldDirty: true })}
              min={0}
              max={2}
              step={0.01}
            />
            <p className="text-xs text-zinc-500">
              Lower values (0-0.3) generate more deterministic output, higher values (0.7-1.2) generate more diverse output
            </p>
          </div>

          {/* Max Tokens - Local reference only */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 opacity-75">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label>Max Tokens</Label>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-400">Local only</span>
              </div>
              <Input
                type="number"
                {...register("llm_max_tokens", { valueAsNumber: true })}
                placeholder="4096"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label>Structured Output Framework</Label>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-400">Local only</span>
              </div>
              <Select
                value={watch("structured_output_framework") || "default"}
                onValueChange={(v) =>
                  setValue("structured_output_framework", v as "baml" | "llm_instructor" | "default", {
                    shouldDirty: true,
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default</SelectItem>
                  <SelectItem value="baml">BAML</SelectItem>
                  <SelectItem value="llm_instructor">LiteLLM Instructor</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between pt-6">
              <div className="flex items-center gap-2">
                <Label>Enable Streaming</Label>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-400">Local only</span>
              </div>
              <Switch
                checked={streaming}
                onCheckedChange={(v) => setValue("llm_streaming", v, { shouldDirty: true })}
              />
            </div>
          </div>

          {/* Rate Limits - Local reference only */}
          <div className="space-y-3 opacity-75">
            <div className="flex items-center gap-2">
              <Label className="text-zinc-300">Rate Limits</Label>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-400">Local only</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label className="text-xs text-zinc-500">Requests/min</Label>
                <Input
                  type="number"
                  {...register("llm_rate_limit_requests", { valueAsNumber: true })}
                  placeholder="100"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs text-zinc-500">Input Tokens/min</Label>
                <Input
                  type="number"
                  {...register("llm_rate_limit_input_tokens", { valueAsNumber: true })}
                  placeholder="100000"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs text-zinc-500">Output Tokens/min</Label>
                <Input
                  type="number"
                  {...register("llm_rate_limit_output_tokens", { valueAsNumber: true })}
                  placeholder="50000"
                />
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-zinc-800">
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                // Ensure type compatibility when resetting
                if (config) {
                  const validFrameworks = ["default", "baml", "llm_instructor"];
                  const rawFramework = config.structured_output_framework;
                  const framework = (validFrameworks.includes(rawFramework || "")
                    ? rawFramework
                    : "default") as "default" | "baml" | "llm_instructor";
                  reset({
                    llm_provider: config.llm_provider || defaultConfig.llm_provider,
                    llm_model: config.llm_model || defaultConfig.llm_model,
                    llm_endpoint: config.llm_endpoint || "",
                    llm_api_key: "",
                    llm_temperature: config.llm_temperature ?? defaultConfig.llm_temperature,
                    llm_max_tokens: config.llm_max_tokens ?? defaultConfig.llm_max_tokens,
                    llm_streaming: config.llm_streaming ?? defaultConfig.llm_streaming,
                    llm_rate_limit_requests: config.llm_rate_limit_requests ?? defaultConfig.llm_rate_limit_requests,
                    llm_rate_limit_input_tokens: config.llm_rate_limit_input_tokens ?? defaultConfig.llm_rate_limit_input_tokens,
                    llm_rate_limit_output_tokens: config.llm_rate_limit_output_tokens ?? defaultConfig.llm_rate_limit_output_tokens,
                    structured_output_framework: framework,
                  });
                } else {
                  reset(defaultConfig);
                }
              }}
              disabled={!isDirty}
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset
            </Button>
            <Button type="submit" loading={isSaving} disabled={!isDirty}>
              <Save className="h-4 w-4 mr-2" />
              Save
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
