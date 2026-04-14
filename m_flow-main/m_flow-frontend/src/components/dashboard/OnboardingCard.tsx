"use client";

/**
 * OnboardingCard Component
 *
 * Displays a welcome card for first-time users with guided steps.
 * Shows when dashboard has no data.
 *
 * Features:
 * - Welcome message
 * - 2-step getting started guide
 * - Progress tracking
 * - Skip option
 */

import React from "react";
import { cn } from "@/lib/utils";
import { Settings, Upload, ArrowRight, X } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  completed: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface OnboardingCardProps {
  steps: OnboardingStep[];
  onDismiss?: () => void;
  onStepClick?: (stepId: string) => void;
  className?: string;
}

// ============================================================================
// Step Item Component
// ============================================================================

interface StepItemProps {
  step: OnboardingStep;
  index: number;
  onClick?: () => void;
}

function StepItem({ step, index, onClick }: StepItemProps) {
  return (
    <button
      onClick={onClick}
      disabled={step.completed}
      className={cn(
        "flex items-start gap-3 p-3 rounded-lg text-left transition-all w-full",
        "border",
        step.completed
          ? "border-emerald-900/30 bg-emerald-950/10"
          : "border-[var(--border-subtle)] hover:border-[var(--border-default)] hover:bg-[var(--bg-hover)]",
        "focus:outline-none focus:ring-1 focus:ring-[var(--border-default)]"
      )}
    >
      {/* Step number / icon */}
      <div
        className={cn(
          "flex items-center justify-center w-8 h-8 rounded-lg shrink-0",
          step.completed
            ? "bg-emerald-500/10 text-emerald-400"
            : "bg-[var(--bg-elevated)] text-[var(--text-muted)]"
        )}
      >
        {step.completed ? (
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
          >
            <path d="M20 6L9 17l-5-5" />
          </svg>
        ) : (
          <span className="text-[13px] font-semibold">{index + 1}</span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p
          className={cn(
            "text-[13px] font-medium",
            step.completed
              ? "text-emerald-400"
              : "text-[var(--text-primary)]"
          )}
        >
          {step.title}
        </p>
        <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
          {step.description}
        </p>
      </div>

      {/* Arrow */}
      {!step.completed && (
        <ArrowRight
          size={14}
          className="text-[var(--text-muted)] shrink-0 mt-1"
        />
      )}
    </button>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function OnboardingCard({
  steps,
  onDismiss,
  onStepClick,
  className,
}: OnboardingCardProps) {
  const completedCount = steps.filter((s) => s.completed).length;
  const progress = Math.round((completedCount / steps.length) * 100);

  return (
    <div
      className={cn(
        "relative rounded-xl border border-[var(--border-subtle)]",
        "bg-[var(--bg-elevated)]",
        "overflow-hidden",
        className
      )}
    >
      {/* Dismiss button */}
      {onDismiss && (
        <button
          onClick={onDismiss}
          className={cn(
            "absolute top-3 right-3 p-1 rounded",
            "text-[var(--text-muted)] hover:text-[var(--text-secondary)]",
            "hover:bg-[var(--bg-hover)] transition-colors"
          )}
          aria-label="Dismiss"
        >
          <X size={14} />
        </button>
      )}

      {/* Header */}
      <div className="pt-8 px-5 pb-4">
        <div className="flex items-center gap-3 mb-3">
          <div>
            <h3 className="text-[15px] font-semibold text-[var(--text-primary)]">
              Welcome to M-Flow
            </h3>
            <p className="text-[12px] text-[var(--text-muted)]">
              Get started in 2 simple steps
            </p>
          </div>
        </div>

        {/* Progress bar */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-1 rounded-full bg-[var(--bg-elevated)] overflow-hidden">
            <div
              className="h-full bg-emerald-400 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="text-[11px] text-[var(--text-muted)] tabular-nums">
            {completedCount}/{steps.length}
          </span>
        </div>
      </div>

      {/* Steps */}
      <div className="px-5 pb-5 space-y-2">
        {steps.map((step, index) => (
          <StepItem
            key={step.id}
            step={step}
            index={index}
            onClick={() => onStepClick?.(step.id)}
          />
        ))}
      </div>
    </div>
  );
}

OnboardingCard.displayName = "OnboardingCard";

// ============================================================================
// Default Steps Configuration
// ============================================================================

export function createDefaultOnboardingSteps(
  config: {
    llmConfigured: boolean;
    hasDocuments: boolean;
  },
  actions: {
    onConfigureLLM: () => void;
    onAddDocument: () => void;
  }
): OnboardingStep[] {
  return [
    {
      id: "configure-llm",
      title: "Configure LLM",
      description: "Set LLM_API_KEY in .env, or configure via Setup page",
      icon: <Settings size={14} />,
      completed: config.llmConfigured,
      action: {
        label: "Configure",
        onClick: actions.onConfigureLLM,
      },
    },
    {
      id: "add-document",
      title: "Add your first document",
      description: "Upload a file or paste text to build your knowledge base",
      icon: <Upload size={14} />,
      completed: config.hasDocuments,
      action: {
        label: "Add",
        onClick: actions.onAddDocument,
      },
    },
  ];
}

export default OnboardingCard;
