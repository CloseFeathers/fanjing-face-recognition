"use client";

import React, { useEffect } from "react";
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
import { Database, Save, RotateCcw, Info } from "lucide-react";
import { VectorDBConfig } from "@/types";

const vectorDbConfigSchema = z.object({
  provider: z.string().min(1, "Please select a vector database"),
  url: z.string().optional(),
  api_key: z.string().optional(),
});

type VectorDBConfigForm = z.infer<typeof vectorDbConfigSchema>;

interface VectorDBSettingsPayload {
  provider: string;
  url: string;
  api_key: string;
}

interface VectorDBSettingsProps {
  config?: VectorDBConfig | null;
  onSave: (config: VectorDBSettingsPayload) => void;
  isSaving?: boolean;
}

interface ProviderOption {
  value: string;
  label: string;
  description?: string;
}

const defaultProviders: ProviderOption[] = [
  { value: "lancedb", label: "LanceDB", description: "Embedded vector database, zero config" },
  { value: "chromadb", label: "ChromaDB", description: "Open source vector database" },
  { value: "pgvector", label: "PGVector", description: "PostgreSQL vector extension" },
];

const defaultConfig: VectorDBConfigForm = {
  provider: "lancedb",
  url: "",
  api_key: "",
};

export function VectorDBSettings({ config, onSave, isSaving }: VectorDBSettingsProps) {
  const {
    register,
    handleSubmit,
    watch,
    setValue,
    reset,
    formState: { errors, isDirty },
  } = useForm<VectorDBConfigForm>({
    resolver: zodResolver(vectorDbConfigSchema),
    defaultValues: defaultConfig,
  });

  useEffect(() => {
    if (config) {
      reset({
        provider: config.provider || defaultConfig.provider,
        url: config.url || "",
        api_key: "",
      });
    }
  }, [config, reset]);

  const selectedProvider = watch("provider");

  const providers: ProviderOption[] = config?.providers?.length
    ? config.providers.map(p => ({ value: p.value, label: p.label }))
    : defaultProviders;

  const onSubmit = (data: VectorDBConfigForm) => {
    const payload: VectorDBSettingsPayload = {
      provider: data.provider,
      url: data.url || "",
      api_key: data.api_key || "",
    };
    onSave(payload);
  };

  const getProviderHelp = (provider: string) => {
    switch (provider) {
      case "lancedb":
        return "LanceDB is an embedded database requiring no URL or API Key configuration. Data is stored locally.";
      case "chromadb":
        return "ChromaDB supports local mode or remote service. Remote service requires URL and API Key configuration.";
      case "pgvector":
        return "Requires PostgreSQL database connection URL, format: postgresql://user:pass@host:5432/db";
      default:
        return "";
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600">
            <Database className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle>Vector Database Configuration</CardTitle>
            <CardDescription>Configure vector storage and retrieval parameters</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Provider */}
          <div className="space-y-2">
            <Label>Vector Database</Label>
            <Select
              value={selectedProvider}
              onValueChange={(v) => setValue("provider", v, { shouldDirty: true })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select vector database" />
              </SelectTrigger>
              <SelectContent>
                {providers.map((p) => (
                  <SelectItem key={p.value} value={p.value}>
                    <div className="flex flex-col">
                      <span>{p.label}</span>
                      {p.description && (
                        <span className="text-xs text-zinc-500">{p.description}</span>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.provider && (
              <p className="text-xs text-red-400">{errors.provider.message}</p>
            )}
          </div>

          {/* Provider-specific help */}
          {selectedProvider && (
            <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
              <Info className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
              <p className="text-xs text-blue-500/90">
                {getProviderHelp(selectedProvider)}
              </p>
            </div>
          )}

          {/* URL - conditional based on provider */}
          {selectedProvider !== "lancedb" && (
            <div className="space-y-2">
              <Label>
                {selectedProvider === "pgvector" ? "Database Connection URL" : "Service URL"}
              </Label>
              <Input
                {...register("url")}
                placeholder={
                  selectedProvider === "pgvector"
                    ? "postgresql://user:pass@localhost:5432/mflow"
                    : selectedProvider === "chromadb"
                    ? "http://localhost:8000"
                    : ""
                }
              />
              <p className="text-xs text-zinc-500">
                {selectedProvider === "pgvector"
                  ? "PostgreSQL database connection string"
                  : "ChromaDB service address"}
              </p>
            </div>
          )}

          {/* API Key - conditional based on provider */}
          {selectedProvider !== "lancedb" && (
            <div className="space-y-2">
              <Label>API Key (optional)</Label>
              <Input
                {...register("api_key")}
                type="password"
                placeholder={
                  selectedProvider === "pgvector"
                    ? "Not required if password is in URL"
                    : "Service API Key"
                }
              />
              <p className="text-xs text-zinc-500">
                {config?.api_key ? "Configured (hidden for security)" : "Fill in if service requires authentication"}
              </p>
            </div>
          )}

          {/* Current Status */}
          {config && (
            <div className="p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
              <div className="text-xs text-zinc-400">
                <p><span className="text-zinc-500">Current provider:</span> {config.provider}</p>
                {config.url && (
                  <p><span className="text-zinc-500">Current URL:</span> {config.url}</p>
                )}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-zinc-800">
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                if (config) {
                  reset({
                    provider: config.provider || defaultConfig.provider,
                    url: config.url || "",
                    api_key: "",
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
