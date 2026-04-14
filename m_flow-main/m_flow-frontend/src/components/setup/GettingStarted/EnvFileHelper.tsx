"use client";

/**
 * EnvFileHelper Component
 *
 * Interactive helper for understanding and configuring .env files.
 * Shows environment variables grouped by service with descriptions,
 * examples, and copy functionality.
 *
 * Features:
 * - Grouped variables by service
 * - Required/optional indicators
 * - Sensitive value masking
 * - Copy individual or all variables
 * - Generate complete .env template
 * - Search/filter variables
 *
 * @example
 * <EnvFileHelper
 *   sections={envSections}
 *   showFilter={true}
 * />
 */

import React, { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  Copy,
  Check,
  ChevronDown,
  Search,
  Eye,
  EyeOff,
  Download,
  AlertCircle,
  CheckCircle2,
  Lock,
  FileText,
} from "lucide-react";
import type { EnvSection, EnvVariable, ProbeKey } from "@/types/setup";
import type { DetailedHealthResponse } from "@/types";

// ============================================================================
// Types
// ============================================================================

export interface EnvFileHelperProps {
  /** Environment sections */
  sections: EnvSection[];
  /** Health data for status indicators */
  healthData?: DetailedHealthResponse;
  /** Whether to show search filter */
  showFilter?: boolean;
  /** Whether to show generate button */
  showGenerate?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export interface EnvVariableRowProps {
  variable: EnvVariable;
  healthData?: DetailedHealthResponse;
  showSensitive?: boolean;
  onToggleSensitive?: () => void;
}

// ============================================================================
// Copy Button Component
// ============================================================================

interface CopyButtonProps {
  text: string;
  size?: "sm" | "md";
  className?: string;
}

function CopyButton({ text, size = "sm", className }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  }, [text]);

  const iconSize = size === "sm" ? 12 : 14;

  return (
    <button
      onClick={handleCopy}
      className={cn(
        "p-1 rounded transition-all",
        copied
          ? "bg-emerald-500/20 text-emerald-400"
          : "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-700",
        className
      )}
      title={copied ? "Copied!" : "Copy"}
    >
      {copied ? <Check size={iconSize} /> : <Copy size={iconSize} />}
    </button>
  );
}

// ============================================================================
// Status Indicator Component
// ============================================================================

interface StatusIndicatorProps {
  service?: ProbeKey;
  healthData?: DetailedHealthResponse;
}

function StatusIndicator({ service, healthData }: StatusIndicatorProps) {
  if (!service || !healthData) return null;

  const probe = healthData.probes[service];
  if (!probe) return null;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium",
        probe.verdict === "up" && "bg-emerald-500/10 text-emerald-400",
        probe.verdict === "warn" && "bg-amber-500/10 text-amber-400",
        probe.verdict === "down" && "bg-red-500/10 text-red-400"
      )}
    >
      {probe.verdict === "up" ? (
        <CheckCircle2 size={10} />
      ) : (
        <AlertCircle size={10} />
      )}
      {probe.verdict.toUpperCase()}
    </span>
  );
}

// ============================================================================
// Variable Row Component
// ============================================================================

export function EnvVariableRow({
  variable,
  healthData,
  showSensitive = false,
  onToggleSensitive,
}: EnvVariableRowProps) {
  const displayValue = variable.sensitive && !showSensitive
    ? "••••••••"
    : variable.example;

  const copyValue = `${variable.key}=${variable.example}`;

  return (
    <div
      className={cn(
        "group flex items-start gap-3 p-3 rounded-lg border transition-colors",
        "border-zinc-800/50 hover:border-zinc-700 hover:bg-zinc-900/30"
      )}
    >
      {/* Key & Description */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <code className="text-[12px] font-mono text-emerald-400">
            {variable.key}
          </code>
          {variable.required && (
            <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30">
              REQUIRED
            </span>
          )}
          {variable.sensitive && (
            <span title="Sensitive value">
              <Lock size={10} className="text-zinc-500" aria-label="Sensitive value" />
            </span>
          )}
          <StatusIndicator service={variable.service} healthData={healthData} />
        </div>
        <p className="text-[11px] text-zinc-500 mt-1">{variable.description}</p>

        {/* Default value */}
        {variable.defaultValue && (
          <p className="text-[10px] text-zinc-600 mt-1">
            Default: <code className="text-zinc-500">{variable.defaultValue}</code>
          </p>
        )}
      </div>

      {/* Value & Actions */}
      <div className="flex items-center gap-2 shrink-0">
        <code
          className={cn(
            "px-2 py-1 rounded text-[11px] font-mono",
            "bg-zinc-900 border border-zinc-800",
            variable.sensitive ? "text-zinc-500" : "text-zinc-300"
          )}
        >
          {displayValue}
        </code>

        {/* Show/hide sensitive */}
        {variable.sensitive && onToggleSensitive && (
          <button
            onClick={onToggleSensitive}
            className="p-1 text-zinc-500 hover:text-zinc-200 transition-colors"
            title={showSensitive ? "Hide value" : "Show value"}
          >
            {showSensitive ? <EyeOff size={12} /> : <Eye size={12} />}
          </button>
        )}

        {/* Copy */}
        <CopyButton
          text={copyValue}
          className="opacity-0 group-hover:opacity-100"
        />
      </div>
    </div>
  );
}

// ============================================================================
// Section Component
// ============================================================================

interface EnvSectionCardProps {
  section: EnvSection;
  healthData?: DetailedHealthResponse;
  defaultExpanded?: boolean;
}

function EnvSectionCard({
  section,
  healthData,
  defaultExpanded = false,
}: EnvSectionCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [showSensitive, setShowSensitive] = useState(false);

  // Generate section env content
  const sectionEnvContent = section.variables
    .map((v) => `${v.key}=${v.example}`)
    .join("\n");

  return (
    <div className="border border-zinc-800 rounded-xl overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          "w-full flex items-center justify-between p-4",
          "bg-zinc-900/30 hover:bg-zinc-900/50 transition-colors"
        )}
      >
        <div className="flex items-center gap-3">
          <FileText size={18} className="text-zinc-500" />
          <div className="text-left">
            <h4 className="text-[14px] font-medium text-zinc-200">
              {section.title}
            </h4>
            <p className="text-[11px] text-zinc-500">{section.description}</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Variable count */}
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-zinc-800 text-zinc-400">
            {section.variables.length} variables
          </span>

          <ChevronDown
            size={16}
            className={cn(
              "text-zinc-500 transition-transform",
              isExpanded && "rotate-180"
            )}
          />
        </div>
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-4 border-t border-zinc-800 space-y-3">
          {/* Section actions */}
          <div className="flex items-center justify-end gap-2">
            <button
              onClick={() => setShowSensitive(!showSensitive)}
              className={cn(
                "flex items-center gap-1.5 px-2 py-1 rounded text-[11px]",
                "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
              )}
            >
              {showSensitive ? <EyeOff size={12} /> : <Eye size={12} />}
              {showSensitive ? "Hide all" : "Show all"}
            </button>
            <CopyButton text={sectionEnvContent} size="md" />
          </div>

          {/* Variables */}
          <div className="space-y-2">
            {section.variables.map((variable) => (
              <EnvVariableRow
                key={variable.key}
                variable={variable}
                healthData={healthData}
                showSensitive={showSensitive}
                onToggleSensitive={() => setShowSensitive(!showSensitive)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function EnvFileHelper({
  sections,
  healthData,
  showFilter = true,
  showGenerate = true,
  className,
}: EnvFileHelperProps) {
  const [filter, setFilter] = useState("");
  const [expandAll, setExpandAll] = useState(false);

  // Filter sections based on search
  const filteredSections = useMemo(() => {
    if (!filter) return sections;

    const lowerFilter = filter.toLowerCase();
    return sections
      .map((section) => ({
        ...section,
        variables: section.variables.filter(
          (v) =>
            v.key.toLowerCase().includes(lowerFilter) ||
            v.description.toLowerCase().includes(lowerFilter)
        ),
      }))
      .filter((section) => section.variables.length > 0);
  }, [sections, filter]);

  // Generate complete .env template
  const generateEnvTemplate = useCallback(() => {
    const content = sections
      .map((section) => {
        const header = `# ${section.title}\n# ${section.description}`;
        const vars = section.variables
          .map((v) => {
            const comment = v.required ? "" : "# ";
            const value = v.sensitive ? "" : v.example;
            return `${comment}${v.key}=${value}`;
          })
          .join("\n");
        return `${header}\n${vars}`;
      })
      .join("\n\n");

    return content;
  }, [sections]);

  // Download .env template
  const handleDownload = useCallback(() => {
    const content = generateEnvTemplate();
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = ".env.template";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [generateEnvTemplate]);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header Actions */}
      <div className="flex items-center justify-between gap-4">
        {/* Search Filter */}
        {showFilter && (
          <div className="relative flex-1 max-w-sm">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500"
            />
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Search variables..."
              className={cn(
                "w-full pl-9 pr-3 py-2 rounded-lg text-[12px]",
                "bg-zinc-900 border border-zinc-800",
                "text-zinc-200 placeholder-zinc-500",
                "focus:outline-none focus:ring-1 focus:ring-zinc-600"
              )}
            />
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExpandAll(!expandAll)}
            className={cn(
              "px-3 py-1.5 rounded-lg text-[11px] font-medium",
              "text-zinc-400 hover:text-zinc-200",
              "bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
            )}
          >
            {expandAll ? "Collapse All" : "Expand All"}
          </button>

          {showGenerate && (
            <button
              onClick={handleDownload}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium",
                "text-zinc-200 bg-zinc-800 hover:bg-zinc-700 transition-colors"
              )}
            >
              <Download size={12} />
              Download .env
            </button>
          )}
        </div>
      </div>

      {/* Sections */}
      <div className="space-y-3">
        {filteredSections.length > 0 ? (
          filteredSections.map((section) => (
            <EnvSectionCard
              key={section.id}
              section={section}
              healthData={healthData}
              defaultExpanded={expandAll}
            />
          ))
        ) : (
          <div className="py-8 text-center">
            <p className="text-[13px] text-zinc-500">
              No variables match "{filter}"
            </p>
          </div>
        )}
      </div>

      {/* Info Box */}
      <div className="flex items-start gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
        <AlertCircle size={16} className="text-amber-500 shrink-0 mt-0.5" />
        <div className="text-[12px] text-zinc-500">
          <p className="text-amber-400 font-medium">Configuration Notes</p>
          <ul className="mt-1 space-y-1">
            <li>• Create a <code className="text-zinc-400">.env</code> file in the project root</li>
            <li>• Required variables must be set before starting the server</li>
            <li>• Restart the server after changing environment variables</li>
            <li>• Never commit <code className="text-zinc-400">.env</code> files to version control</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

EnvVariableRow.displayName = "EnvVariableRow";
EnvFileHelper.displayName = "EnvFileHelper";

// ============================================================================
// Default Export
// ============================================================================

export default EnvFileHelper;
