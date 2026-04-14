"use client";

/**
 * CodeExample Component
 *
 * Displays code snippets with syntax highlighting, copy functionality,
 * and optional output display.
 *
 * Features:
 * - Multiple language support (Python, Bash, TypeScript, JSON, env)
 * - Copy to clipboard
 * - Line numbers (optional)
 * - Line highlighting
 * - Output display
 * - Collapsible code blocks
 *
 * @example
 * <CodeExample
 *   example={codeExampleData}
 *   showCopy={true}
 * />
 */

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Copy, Check, ChevronDown, Terminal, Code as CodeIcon } from "lucide-react";
import type { CodeExample as CodeExampleType, CodeLanguage } from "@/types/setup";

// ============================================================================
// Types
// ============================================================================

export interface CodeExampleProps {
  /** Code example data */
  example: CodeExampleType;
  /** Whether to show copy button */
  showCopy?: boolean;
  /** Whether to show title */
  showTitle?: boolean;
  /** Whether collapsible */
  collapsible?: boolean;
  /** Default collapsed state */
  defaultCollapsed?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export interface CodeBlockProps {
  /** Code content */
  code: string;
  /** Language */
  language: CodeLanguage;
  /** Show line numbers */
  showLineNumbers?: boolean;
  /** Highlighted lines (1-indexed) */
  highlightLines?: number[];
  /** Show copy button */
  showCopy?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Language Configuration
// ============================================================================

interface LanguageConfig {
  label: string;
  icon: React.ReactNode;
  bgClass: string;
  borderClass: string;
}

const LANGUAGE_CONFIG: Record<CodeLanguage, LanguageConfig> = {
  python: {
    label: "Python",
    icon: <CodeIcon size={12} />,
    bgClass: "bg-blue-500/10",
    borderClass: "border-blue-500/30",
  },
  bash: {
    label: "Bash",
    icon: <Terminal size={12} />,
    bgClass: "bg-green-500/10",
    borderClass: "border-green-500/30",
  },
  typescript: {
    label: "TypeScript",
    icon: <CodeIcon size={12} />,
    bgClass: "bg-blue-500/10",
    borderClass: "border-blue-500/30",
  },
  json: {
    label: "JSON",
    icon: <CodeIcon size={12} />,
    bgClass: "bg-amber-500/10",
    borderClass: "border-amber-500/30",
  },
  env: {
    label: ".env",
    icon: <CodeIcon size={12} />,
    bgClass: "bg-zinc-500/10",
    borderClass: "border-zinc-500/30",
  },
};

// ============================================================================
// Copy Button Component
// ============================================================================

interface CopyButtonProps {
  text: string;
  className?: string;
}

function CopyButton({ text, className }: CopyButtonProps) {
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

  return (
    <button
      onClick={handleCopy}
      className={cn(
        "p-1.5 rounded-md transition-all",
        copied
          ? "bg-emerald-500/20 text-emerald-400"
          : "bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700",
        className
      )}
      title={copied ? "Copied!" : "Copy to clipboard"}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

// ============================================================================
// Language Badge Component
// ============================================================================

interface LanguageBadgeProps {
  language: CodeLanguage;
}

function LanguageBadge({ language }: LanguageBadgeProps) {
  const config = LANGUAGE_CONFIG[language];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border",
        config.bgClass,
        config.borderClass,
        "text-zinc-300"
      )}
    >
      {config.icon}
      {config.label}
    </span>
  );
}

// ============================================================================
// Code Block Component (standalone)
// ============================================================================

export function CodeBlock({
  code,
  language,
  showLineNumbers = false,
  highlightLines = [],
  showCopy = true,
  className,
}: CodeBlockProps) {
  const lines = code.split("\n");

  return (
    <div className={cn("relative group", className)}>
      <pre
        className={cn(
          "p-4 rounded-lg overflow-x-auto",
          "bg-zinc-950 border border-zinc-800",
          "text-[12px] leading-relaxed font-mono"
        )}
      >
        <code>
          {lines.map((line, idx) => {
            const lineNum = idx + 1;
            const isHighlighted = highlightLines.includes(lineNum);

            return (
              <div
                key={idx}
                className={cn(
                  "flex",
                  isHighlighted && "bg-amber-500/10 -mx-4 px-4"
                )}
              >
                {showLineNumbers && (
                  <span
                    className={cn(
                      "select-none pr-4 text-right min-w-[2.5rem]",
                      isHighlighted ? "text-amber-400" : "text-zinc-600"
                    )}
                  >
                    {lineNum}
                  </span>
                )}
                <span className="text-zinc-300 whitespace-pre">{line}</span>
              </div>
            );
          })}
        </code>
      </pre>

      {/* Copy Button */}
      {showCopy && (
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <CopyButton text={code} />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Output Block Component
// ============================================================================

interface OutputBlockProps {
  output: string;
}

function OutputBlock({ output }: OutputBlockProps) {
  return (
    <div className="mt-2">
      <div className="flex items-center gap-2 mb-1">
        <Terminal size={12} className="text-zinc-500" />
        <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
          Output
        </span>
      </div>
      <pre
        className={cn(
          "p-3 rounded-lg overflow-x-auto",
          "bg-zinc-900/50 border border-zinc-800/50",
          "text-[11px] leading-relaxed font-mono text-emerald-400"
        )}
      >
        {output}
      </pre>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function CodeExample({
  example,
  showCopy = true,
  showTitle = true,
  collapsible = false,
  defaultCollapsed = false,
  className,
}: CodeExampleProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  return (
    <div
      className={cn(
        "rounded-xl border border-zinc-800 overflow-hidden",
        className
      )}
    >
      {/* Header */}
      {showTitle && (
        <div
          className={cn(
            "flex items-center justify-between px-4 py-3",
            "bg-zinc-900/50 border-b border-zinc-800",
            collapsible && "cursor-pointer hover:bg-zinc-900/70"
          )}
          onClick={collapsible ? () => setIsCollapsed(!isCollapsed) : undefined}
        >
          <div className="flex items-center gap-3">
            <LanguageBadge language={example.language} />
            <div>
              <h4 className="text-[13px] font-medium text-zinc-200">
                {example.title}
              </h4>
              {example.description && (
                <p className="text-[11px] text-zinc-500 mt-0.5">
                  {example.description}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {showCopy && !collapsible && <CopyButton text={example.code} />}
            {collapsible && (
              <ChevronDown
                size={16}
                className={cn(
                  "text-zinc-500 transition-transform",
                  isCollapsed && "-rotate-90"
                )}
              />
            )}
          </div>
        </div>
      )}

      {/* Code Content */}
      {(!collapsible || !isCollapsed) && (
        <div className="p-4 bg-zinc-950">
          <CodeBlock
            code={example.code}
            language={example.language}
            showLineNumbers={example.showLineNumbers}
            highlightLines={example.highlightLines}
            showCopy={!showTitle && showCopy}
          />

          {/* Output */}
          {example.output && <OutputBlock output={example.output} />}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Code Example Tabs Component
// ============================================================================

export interface CodeExampleTabsProps {
  examples: CodeExampleType[];
  defaultTab?: string;
  className?: string;
}

export function CodeExampleTabs({
  examples,
  defaultTab,
  className,
}: CodeExampleTabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || examples[0]?.id);
  const activeExample = examples.find((e) => e.id === activeTab) || examples[0];

  if (examples.length === 0) return null;

  return (
    <div className={cn("rounded-xl border border-zinc-800 overflow-hidden", className)}>
      {/* Tabs */}
      <div className="flex border-b border-zinc-800 bg-zinc-900/50 overflow-x-auto">
        {examples.map((example) => (
          <button
            key={example.id}
            onClick={() => setActiveTab(example.id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 text-[12px] font-medium",
              "border-b-2 transition-colors whitespace-nowrap",
              activeTab === example.id
                ? "border-zinc-100 text-zinc-100 bg-zinc-900/50"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            )}
          >
            {LANGUAGE_CONFIG[example.language]?.icon}
            {example.title}
          </button>
        ))}
      </div>

      {/* Active Example */}
      <div className="p-4 bg-zinc-950">
        <CodeBlock
          code={activeExample.code}
          language={activeExample.language}
          showLineNumbers={activeExample.showLineNumbers}
          highlightLines={activeExample.highlightLines}
          showCopy={true}
        />
        {activeExample.output && <OutputBlock output={activeExample.output} />}
      </div>
    </div>
  );
}

// ============================================================================
// Simple Code Snippet (inline use)
// ============================================================================

export interface CodeSnippetProps {
  code: string;
  language?: CodeLanguage;
  showCopy?: boolean;
  className?: string;
}

export function CodeSnippet({
  code,
  language = "bash",
  showCopy = true,
  className,
}: CodeSnippetProps) {
  return (
    <div className={cn("relative group", className)}>
      <pre
        className={cn(
          "px-3 py-2 rounded-lg overflow-x-auto",
          "bg-zinc-900 border border-zinc-800",
          "text-[12px] font-mono text-zinc-300"
        )}
      >
        <code>{code}</code>
      </pre>
      {showCopy && (
        <div className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <CopyButton text={code} />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

CodeBlock.displayName = "CodeBlock";
CodeExample.displayName = "CodeExample";
CodeExampleTabs.displayName = "CodeExampleTabs";
CodeSnippet.displayName = "CodeSnippet";

// ============================================================================
// Default Export
// ============================================================================

export default CodeExample;
