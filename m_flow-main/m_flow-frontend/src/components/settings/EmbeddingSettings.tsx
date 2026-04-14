"use client";

import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod/v4";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Layers, AlertCircle, Info, Loader2, RotateCcw } from "lucide-react";
import { EmbeddingConfig, EmbeddingSettingsUpdate } from "@/types";

const embeddingConfigSchema = z.object({
  provider: z.string().min(1, "Please select an embedding provider"),
  model: z.string().min(1, "Please enter a model name"),
  endpoint: z.string().optional(),
  api_key: z.string().optional(),
  dimensions: z.number().min(1).optional(),
});

type EmbeddingConfigForm = z.infer<typeof embeddingConfigSchema>;

interface EmbeddingSettingsProps {
  config?: EmbeddingConfig | null;
  onSave?: (config: EmbeddingSettingsUpdate) => Promise<void>;
  isSaving?: boolean;
}

const providers = [
  { value: "openai", label: "OpenAI" },
  { value: "azure", label: "Azure OpenAI" },
  { value: "ollama", label: "Ollama (Local)" },
  { value: "fastembed", label: "FastEmbed" },
];

const defaultConfig: EmbeddingConfigForm = {
  provider: "openai",
  model: "text-embedding-3-small",
  dimensions: 1536,
};

export function EmbeddingSettings({ config, onSave, isSaving }: EmbeddingSettingsProps) {
  const [hasChanges, setHasChanges] = useState(false);
  
  const {
    register,
    watch,
    setValue,
    reset,
    handleSubmit,
    formState: { errors },
  } = useForm<EmbeddingConfigForm>({
    resolver: zodResolver(embeddingConfigSchema),
    defaultValues: defaultConfig,
  });

  // Watch form changes
  const formValues = watch();
  
  useEffect(() => {
    if (config) {
      const providerChanged = formValues.provider !== (config.embedding_provider || defaultConfig.provider);
      const modelChanged = formValues.model !== (config.embedding_model || defaultConfig.model);
      const dimensionsChanged = formValues.dimensions !== (config.embedding_dimensions ?? defaultConfig.dimensions);
      const endpointChanged = (formValues.endpoint || "") !== (config.embedding_endpoint || "");
      const apiKeyEntered = Boolean(formValues.api_key && formValues.api_key.length > 0);
      
      setHasChanges(providerChanged || modelChanged || dimensionsChanged || endpointChanged || apiKeyEntered);
    }
  }, [formValues, config]);

  // Update form values when config loads from backend
  useEffect(() => {
    if (config) {
      reset({
        provider: config.embedding_provider || defaultConfig.provider,
        model: config.embedding_model || defaultConfig.model,
        endpoint: config.embedding_endpoint || "",
        api_key: "",
        dimensions: config.embedding_dimensions ?? defaultConfig.dimensions,
      });
    }
  }, [config, reset]);

  const onSubmit = async (data: EmbeddingConfigForm) => {
    if (!onSave) return;
    
    const updateData: EmbeddingSettingsUpdate = {
      provider: data.provider,
      model: data.model,
      dimensions: data.dimensions,
      endpoint: data.endpoint || undefined,
      api_key: data.api_key || undefined,
    };
    
    await onSave(updateData);
    setHasChanges(false);
  };

  const handleReset = () => {
    if (config) {
      reset({
        provider: config.embedding_provider || defaultConfig.provider,
        model: config.embedding_model || defaultConfig.model,
        endpoint: config.embedding_endpoint || "",
        api_key: "",
        dimensions: config.embedding_dimensions ?? defaultConfig.dimensions,
      });
      setHasChanges(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-500 to-green-600">
            <Layers className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle>Embedding Configuration</CardTitle>
            <CardDescription>Configure text embedding model parameters</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Provider */}
              <div className="space-y-2">
                <Label>Embedding Provider</Label>
                <Select
                  value={watch("provider")}
                  onValueChange={(v) => setValue("provider", v)}
                  disabled={isSaving}
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
                {errors.provider && (
                  <p className="text-xs text-red-400">{errors.provider.message}</p>
                )}
              </div>

              {/* Model */}
              <div className="space-y-2">
                <Label>Model Name</Label>
                <Input
                  {...register("model")}
                  placeholder="text-embedding-3-small"
                  error={errors.model?.message}
                  disabled={isSaving}
                />
              </div>

              {/* API Endpoint */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label>API Endpoint</Label>
                  <span className="text-[10px] text-[var(--text-muted)]">(optional)</span>
                </div>
                <Input
                  {...register("endpoint")}
                  placeholder="Use default endpoint"
                  disabled={isSaving}
                />
              </div>

              {/* API Key */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label>API Key</Label>
                  <span className="text-[10px] text-[var(--text-muted)]">(leave empty to keep current)</span>
                </div>
                <Input
                  {...register("api_key")}
                  type="password"
                  placeholder="Enter new API Key or leave empty"
                  disabled={isSaving}
                />
              </div>

              {/* Dimensions */}
              <div className="space-y-2">
                <Label>Embedding Dimensions</Label>
                <Input
                  type="number"
                  {...register("dimensions", { valueAsNumber: true })}
                  placeholder="1536"
                  disabled={isSaving}
                />
                <p className="text-xs text-zinc-500">
                  Different models have different default dimensions, refer to model documentation
                </p>
              </div>
            </div>

            {/* Configuration Note */}
            <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
              <Info className="h-4 w-4 text-blue-400 mt-0.5 shrink-0" />
              <div className="text-xs text-blue-300/80">
                <p>
                  Changes will be saved to <code className="px-1 py-0.5 bg-blue-500/10 rounded">.env</code> file.
                  Some changes may require service restart to take effect.
                </p>
              </div>
            </div>

            {/* Action Buttons */}
            {onSave && (
              <div className="flex items-center justify-between pt-4 border-t border-[var(--border-subtle)]">
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={!hasChanges || isSaving}
                  className="flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <RotateCcw size={12} />
                  Reset
                </button>
                <Button
                  type="submit"
                  disabled={!hasChanges || isSaving}
                  className="flex items-center gap-2"
                >
                  {isSaving && <Loader2 size={14} className="animate-spin" />}
                  Save Changes
                </Button>
              </div>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
