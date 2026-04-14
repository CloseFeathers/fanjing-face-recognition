"use client";

/**
 * IssueCard Component
 *
 * Displays a common issue card with problem description,
 * detection method, and solution steps.
 *
 * Features:
 * - Collapsible solution steps
 * - Related service badges
 * - Copy solution steps
 * - Search highlighting
 *
 * @example
 * <IssueCard
 *   issue={issueData}
 *   isExpanded={false}
 *   onToggle={() => setExpanded(!expanded)}
 * />
 */

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  ChevronDown,
  Copy,
  Check,
  Search,
  ExternalLink,
  CheckCircle2,
} from "lucide-react";
import type { CommonIssue, ProbeKey } from "@/types/setup";
import { PROBE_METADATA } from "@/lib/utils/health";

// ============================================================================
// Types
// ============================================================================

export interface IssueCardProps {
  /** Issue data */
  issue: CommonIssue;
  /** Whether card is expanded */
  isExpanded?: boolean;
  /** Toggle handler */
  onToggle?: () => void;
  /** Search query for highlighting */
  searchQuery?: string;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Service Badge Component
// ============================================================================

interface ServiceBadgeProps {
  service: ProbeKey;
}

function ServiceBadge({ service }: ServiceBadgeProps) {
  const metadata = PROBE_METADATA[service];

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium",
        "bg-zinc-800 text-zinc-400 border border-zinc-700"
      )}
      title={metadata.description}
    >
      {metadata.displayName}
    </span>
  );
}

// ============================================================================
// Solution Step Component
// ============================================================================

interface SolutionStepProps {
  step: string;
  index: number;
  searchQuery?: string;
}

function SolutionStep({ step, index, searchQuery }: SolutionStepProps) {
  const [copied, setCopied] = useState(false);

  // Check if step contains code (backticks)
  const hasCode = step.includes("`");

  const handleCopy = useCallback(async () => {
    // Extract code from backticks if present
    const codeMatch = step.match(/`([^`]+)`/);
    const textToCopy = codeMatch ? codeMatch[1] : step;

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  }, [step]);

  // Highlight matching text
  const highlightText = (text: string) => {
    if (!searchQuery) return text;

    const regex = new RegExp(`(${searchQuery})`, "gi");
    const parts = text.split(regex);

    return parts.map((part, i) =>
      regex.test(part) ? (
        <mark key={i} className="bg-amber-500/30 text-amber-200 rounded px-0.5">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  // Render step with code formatting
  const renderStep = () => {
    if (!hasCode) return highlightText(step);

    const parts = step.split(/(`[^`]+`)/);
    return parts.map((part, i) => {
      if (part.startsWith("`") && part.endsWith("`")) {
        const code = part.slice(1, -1);
        return (
          <code
            key={i}
            className="px-1.5 py-0.5 rounded bg-zinc-900 text-emerald-400 font-mono text-[11px]"
          >
            {highlightText(code)}
          </code>
        );
      }
      return <span key={i}>{highlightText(part)}</span>;
    });
  };

  return (
    <div className="flex gap-3 group">
      {/* Step number */}
      <div
        className={cn(
          "w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5",
          "bg-zinc-800 text-zinc-500 text-[10px] font-medium"
        )}
      >
        {index + 1}
      </div>

      {/* Step content */}
      <div className="flex-1 min-w-0">
        <p className="text-[12px] text-zinc-300 leading-relaxed">
          {renderStep()}
        </p>
      </div>

      {/* Copy button (for code steps) */}
      {hasCode && (
        <button
          onClick={handleCopy}
          className={cn(
            "p-1 rounded shrink-0 opacity-0 group-hover:opacity-100 transition-opacity",
            copied
              ? "text-emerald-400"
              : "text-zinc-500 hover:text-zinc-200"
          )}
          title={copied ? "Copied!" : "Copy command"}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
        </button>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function IssueCard({
  issue,
  isExpanded = false,
  onToggle,
  searchQuery,
  className,
}: IssueCardProps) {
  const isClickable = !!onToggle;

  // Highlight matching text in title
  const highlightTitle = () => {
    if (!searchQuery) return issue.title;

    const regex = new RegExp(`(${searchQuery})`, "gi");
    const parts = issue.title.split(regex);

    return parts.map((part, i) =>
      regex.test(part) ? (
        <mark key={i} className="bg-amber-500/30 text-amber-200 rounded px-0.5">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <div
      className={cn(
        "rounded-xl border transition-all duration-200",
        isExpanded
          ? "border-amber-900/40 bg-amber-950/10"
          : "border-zinc-800 hover:border-zinc-700",
        className
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-start gap-3 p-4",
          isClickable && "cursor-pointer"
        )}
        onClick={onToggle}
        role={isClickable ? "button" : undefined}
        tabIndex={isClickable ? 0 : undefined}
        onKeyDown={
          isClickable
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onToggle?.();
                }
              }
            : undefined
        }
      >
        {/* Icon */}
        <div
          className={cn(
            "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
            isExpanded ? "bg-amber-500/20" : "bg-zinc-800"
          )}
        >
          <AlertTriangle
            size={16}
            className={isExpanded ? "text-amber-400" : "text-zinc-500"}
          />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <h4 className="text-[14px] font-medium text-zinc-200">
            {highlightTitle()}
          </h4>
          <p className="text-[12px] text-zinc-500 mt-0.5 line-clamp-2">
            {issue.problem}
          </p>

          {/* Service badges */}
          {issue.relatedServices.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {issue.relatedServices.map((service) => (
                <ServiceBadge key={service} service={service} />
              ))}
            </div>
          )}
        </div>

        {/* Expand icon */}
        {isClickable && (
          <ChevronDown
            size={16}
            className={cn(
              "text-zinc-500 shrink-0 transition-transform",
              isExpanded && "rotate-180"
            )}
          />
        )}
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 pb-4 pt-0 space-y-4 border-t border-zinc-800/50">
          {/* Detection */}
          <div className="pt-4">
            <div className="flex items-center gap-2 mb-2">
              <Search size={12} className="text-zinc-500" />
              <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
                How to Detect
              </span>
            </div>
            <p className="text-[12px] text-zinc-400 pl-5">{issue.detection}</p>
          </div>

          {/* Solution Steps */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 size={12} className="text-emerald-500" />
              <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
                Solution Steps
              </span>
            </div>
            <div className="space-y-3 pl-5">
              {issue.solution.map((step, idx) => (
                <SolutionStep
                  key={idx}
                  step={step}
                  index={idx}
                  searchQuery={searchQuery}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Issue List Component
// ============================================================================

export interface IssueListProps {
  issues: CommonIssue[];
  searchQuery?: string;
  className?: string;
}

export function IssueList({ issues, searchQuery, className }: IssueListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const handleToggle = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  if (issues.length === 0) {
    return (
      <div className="py-8 text-center">
        <AlertTriangle size={24} className="text-zinc-600 mx-auto mb-2" />
        <p className="text-[13px] text-zinc-500">No issues found</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-3", className)}>
      {issues.map((issue) => (
        <IssueCard
          key={issue.id}
          issue={issue}
          isExpanded={expandedId === issue.id}
          onToggle={() => handleToggle(issue.id)}
          searchQuery={searchQuery}
        />
      ))}
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

IssueCard.displayName = "IssueCard";
IssueList.displayName = "IssueList";

// ============================================================================
// Default Export
// ============================================================================

export default IssueCard;
