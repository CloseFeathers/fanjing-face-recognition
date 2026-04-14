"use client";

/**
 * StorageConfigStep Component
 *
 * Read-only configuration step for file storage settings.
 * Shows current configuration and provides .env editing guidance.
 *
 * Features:
 * - Display current file storage configuration from health probe
 * - Show environment variable templates
 * - Support for local and cloud storage providers
 *
 * @example
 * <StorageConfigStep healthData={wizard.healthData} />
 */

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Copy, Check, Info, AlertCircle, CheckCircle2, FolderOpen, Cloud } from "lucide-react";
import { ConfigBadge } from "../ConfigBadge";
import type { DetailedHealthResponse } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface StorageConfigStepProps {
  /** Health check data for status display */
  healthData?: DetailedHealthResponse;
  /** Whether health data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Storage Provider Configurations
// ============================================================================

interface StorageProviderConfig {
  id: string;
  name: string;
  description: string;
  icon: "local" | "cloud";
  envVars: {
    key: string;
    description: string;
    example: string;
    required: boolean;
  }[];
}

const STORAGE_PROVIDERS: StorageProviderConfig[] = [
  {
    id: "local",
    name: "Local File System",
    description: "Store files on the local disk. Best for development and single-server deployments.",
    icon: "local",
    envVars: [
      {
        key: "FILE_STORAGE_PROVIDER",
        description: "Storage provider type",
        example: "local",
        required: true,
      },
      {
        key: "FILE_STORAGE_PATH",
        description: "Base directory for file storage",
        example: "/data/m_flow/files",
        required: true,
      },
    ],
  },
  {
    id: "s3",
    name: "Amazon S3",
    description: "Store files in AWS S3. Best for production and multi-server deployments.",
    icon: "cloud",
    envVars: [
      {
        key: "FILE_STORAGE_PROVIDER",
        description: "Storage provider type",
        example: "s3",
        required: true,
      },
      {
        key: "AWS_S3_BUCKET",
        description: "S3 bucket name",
        example: "my-mflow-bucket",
        required: true,
      },
      {
        key: "AWS_ACCESS_KEY_ID",
        description: "AWS access key",
        example: "AKIAIOSFODNN7EXAMPLE",
        required: true,
      },
      {
        key: "AWS_SECRET_ACCESS_KEY",
        description: "AWS secret key",
        example: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        required: true,
      },
      {
        key: "AWS_REGION",
        description: "AWS region",
        example: "us-east-1",
        required: false,
      },
    ],
  },
  {
    id: "gcs",
    name: "Google Cloud Storage",
    description: "Store files in GCS. Best for GCP-based deployments.",
    icon: "cloud",
    envVars: [
      {
        key: "FILE_STORAGE_PROVIDER",
        description: "Storage provider type",
        example: "gcs",
        required: true,
      },
      {
        key: "GCS_BUCKET",
        description: "GCS bucket name",
        example: "my-mflow-bucket",
        required: true,
      },
      {
        key: "GOOGLE_APPLICATION_CREDENTIALS",
        description: "Path to service account JSON",
        example: "/path/to/service-account.json",
        required: true,
      },
    ],
  },
];

// ============================================================================
// Code Block Component
// ============================================================================

interface CodeBlockProps {
  content: string;
}

function CodeBlock({ content }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  }, [content]);

  return (
    <div className="relative group">
      <pre className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 overflow-x-auto">
        <code className="text-[11px] text-zinc-300 font-mono whitespace-pre">
          {content}
        </code>
      </pre>
      <button
        onClick={handleCopy}
        className={cn(
          "absolute top-2 right-2 p-1 rounded-md transition-all",
          "opacity-0 group-hover:opacity-100",
          copied
            ? "bg-emerald-500/20 text-emerald-400"
            : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
        )}
        title={copied ? "Copied!" : "Copy to clipboard"}
      >
        {copied ? <Check size={12} /> : <Copy size={12} />}
      </button>
    </div>
  );
}

// ============================================================================
// Status Display Component
// ============================================================================

interface StorageStatusProps {
  healthData?: DetailedHealthResponse;
}

function StorageStatus({ healthData }: StorageStatusProps) {
  if (!healthData) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-lg border border-zinc-800 bg-zinc-900/50">
        <div className="w-4 h-4 rounded-full bg-zinc-700 animate-pulse" />
        <div className="text-[13px] text-zinc-500">Loading status...</div>
      </div>
    );
  }

  const probe = healthData.probes.file_storage;

  return (
    <div
      className={cn(
        "flex items-start gap-3 p-3 rounded-lg border",
        probe.verdict === "up" && "border-emerald-900/30 bg-emerald-950/10",
        probe.verdict === "warn" && "border-amber-900/30 bg-amber-950/10",
        probe.verdict === "down" && "border-red-900/30 bg-red-950/10"
      )}
    >
      {probe.verdict === "up" ? (
        <CheckCircle2 size={18} className="text-emerald-400 shrink-0 mt-0.5" />
      ) : probe.verdict === "warn" ? (
        <AlertCircle size={18} className="text-amber-400 shrink-0 mt-0.5" />
      ) : (
        <AlertCircle size={18} className="text-red-400 shrink-0 mt-0.5" />
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between">
          <p
            className={cn(
              "text-[13px] font-medium",
              probe.verdict === "up" && "text-emerald-400",
              probe.verdict === "warn" && "text-amber-400",
              probe.verdict === "down" && "text-red-400"
            )}
          >
            {probe.verdict === "up"
              ? "File Storage Connected"
              : probe.verdict === "warn"
              ? "File Storage Degraded"
              : "File Storage Not Configured"}
          </p>
          {probe.latency_ms > 0 && (
            <span className="text-[10px] text-zinc-600">{probe.latency_ms}ms</span>
          )}
        </div>
        <p className="text-[11px] text-zinc-500 mt-0.5">
          {probe.backend !== "unknown"
            ? `Provider: ${probe.backend}`
            : probe.note || "Not configured"}
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Provider Card Component
// ============================================================================

interface ProviderCardProps {
  config: StorageProviderConfig;
  isActive: boolean;
}

function ProviderCard({ config, isActive }: ProviderCardProps) {
  const [isExpanded, setIsExpanded] = useState(isActive);

  const envTemplate = config.envVars.map((v) => `${v.key}=${v.example}`).join("\n");

  return (
    <div
      className={cn(
        "border rounded-lg overflow-hidden transition-all",
        isActive ? "border-emerald-900/50 bg-emerald-950/10" : "border-zinc-800"
      )}
    >
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-zinc-900/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          {/* Icon */}
          {config.icon === "local" ? (
            <FolderOpen size={18} className={isActive ? "text-emerald-400" : "text-zinc-500"} />
          ) : (
            <Cloud size={18} className={isActive ? "text-emerald-400" : "text-zinc-500"} />
          )}

          {/* Name */}
          <div className="text-left">
            <div className="flex items-center gap-2">
              <p className={cn("text-[13px] font-medium", isActive ? "text-emerald-400" : "text-zinc-200")}>
                {config.name}
              </p>
              {isActive && (
                <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-emerald-500/20 text-emerald-400">
                  ACTIVE
                </span>
              )}
            </div>
            <p className="text-[11px] text-zinc-500">{config.description}</p>
          </div>
        </div>

        {/* Expand Icon */}
        <svg
          className={cn("w-4 h-4 text-zinc-500 transition-transform", isExpanded && "rotate-180")}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="p-4 border-t border-zinc-800 space-y-4">
          {/* Environment Variables */}
          <div className="space-y-2">
            <p className="text-[11px] text-zinc-400 font-medium">Environment Variables:</p>
            <div className="space-y-1">
              {config.envVars.map((v) => (
                <div key={v.key} className="flex items-start gap-2 text-[11px]">
                  <code className="text-emerald-400 font-mono shrink-0">{v.key}</code>
                  <span className="text-zinc-600">-</span>
                  <span className="text-zinc-500">{v.description}</span>
                  {v.required && <span className="text-amber-400 shrink-0">(required)</span>}
                </div>
              ))}
            </div>
          </div>

          {/* Example */}
          <div className="space-y-2">
            <p className="text-[11px] text-zinc-400 font-medium">Example Configuration:</p>
            <CodeBlock content={envTemplate} />
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function StorageConfigStep({ healthData, isLoading = false, className }: StorageConfigStepProps) {
  // Determine active provider from health data
  const activeProvider = healthData?.probes.file_storage?.backend || "local";

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-[16px] font-semibold text-zinc-100">File Storage Configuration</h3>
          <p className="text-[13px] text-zinc-500 mt-1">
            Configure file storage for documents, images, and other assets.
          </p>
        </div>
        <ConfigBadge type="env-only" />
      </div>

      {/* Current Status */}
      <StorageStatus healthData={healthData} />

      {/* Provider Cards */}
      <div className="space-y-3">
        <h4 className="text-[14px] font-medium text-zinc-200">Available Providers</h4>
        {STORAGE_PROVIDERS.map((provider) => (
          <ProviderCard
            key={provider.id}
            config={provider}
            isActive={activeProvider.toLowerCase().includes(provider.id)}
          />
        ))}
      </div>

      {/* Info Box */}
      <div className="flex items-start gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
        <Info size={16} className="text-amber-500 shrink-0 mt-0.5" />
        <div className="text-[12px] text-zinc-500">
          <p className="text-amber-400 font-medium">Requires Restart</p>
          <p className="mt-1">
            Storage configuration is loaded at startup. After modifying the{" "}
            <code className="text-zinc-400">.env</code> file, restart the M-Flow server for changes to
            take effect.
          </p>
        </div>
      </div>

      {/* Default Notice */}
      <div className="flex items-start gap-3 p-3 rounded-lg bg-emerald-950/10 border border-emerald-900/30">
        <CheckCircle2 size={16} className="text-emerald-400 shrink-0 mt-0.5" />
        <div className="text-[12px] text-zinc-400">
          <p className="text-emerald-400 font-medium">Default Configuration</p>
          <p className="mt-1">
            By default, M-Flow uses local file storage in the application data directory. This is
            suitable for development and single-server deployments.
          </p>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Display Name
// ============================================================================

StorageConfigStep.displayName = "StorageConfigStep";

// ============================================================================
// Default Export
// ============================================================================

export default StorageConfigStep;
