"use client";

/**
 * SolutionGuide Component
 *
 * Displays step-by-step solution guides for common tasks.
 * Includes code snippets, warnings, and tips.
 *
 * Features:
 * - Step-by-step navigation
 * - Code blocks with syntax highlighting
 * - Copy functionality
 * - Progress tracking
 * - Warnings and tips
 *
 * @example
 * <SolutionGuide
 *   guide={solutionGuideData}
 *   onComplete={() => console.log("Done!")}
 * />
 */

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
  Copy,
  Check,
  Clock,
  ExternalLink,
  X,
} from "lucide-react";
import type { SolutionGuide, SolutionStep } from "@/content/troubleshooting";
import { PROBE_METADATA } from "@/lib/utils/health";
import { CodeBlock } from "../GettingStarted/CodeExample";

// ============================================================================
// Types
// ============================================================================

export interface SolutionGuideProps {
  /** Solution guide data */
  guide: SolutionGuide;
  /** Callback when guide is completed */
  onComplete?: () => void;
  /** Callback to close the guide */
  onClose?: () => void;
  /** Additional CSS classes */
  className?: string;
}

export interface SolutionStepCardProps {
  step: SolutionStep;
  isActive: boolean;
  isCompleted: boolean;
  onComplete?: () => void;
}

// ============================================================================
// Difficulty Badge Component
// ============================================================================

interface DifficultyBadgeProps {
  difficulty: SolutionGuide["difficulty"];
}

function DifficultyBadge({ difficulty }: DifficultyBadgeProps) {
  const config = {
    easy: {
      label: "Easy",
      bgClass: "bg-emerald-500/10",
      textClass: "text-emerald-400",
      borderClass: "border-emerald-500/30",
    },
    medium: {
      label: "Medium",
      bgClass: "bg-amber-500/10",
      textClass: "text-amber-400",
      borderClass: "border-amber-500/30",
    },
    hard: {
      label: "Advanced",
      bgClass: "bg-red-500/10",
      textClass: "text-red-400",
      borderClass: "border-red-500/30",
    },
  };

  const c = config[difficulty];

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium border",
        c.bgClass,
        c.textClass,
        c.borderClass
      )}
    >
      {c.label}
    </span>
  );
}

// ============================================================================
// Progress Indicator Component
// ============================================================================

interface ProgressIndicatorProps {
  currentStep: number;
  totalSteps: number;
  completedSteps: Set<number>;
}

function ProgressIndicator({
  currentStep,
  totalSteps,
  completedSteps,
}: ProgressIndicatorProps) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: totalSteps }, (_, i) => {
        const stepNum = i + 1;
        const isCompleted = completedSteps.has(stepNum);
        const isCurrent = stepNum === currentStep;

        return (
          <div
            key={i}
            className={cn(
              "w-8 h-1.5 rounded-full transition-colors",
              isCompleted && "bg-emerald-500",
              isCurrent && !isCompleted && "bg-zinc-100",
              !isCompleted && !isCurrent && "bg-zinc-700"
            )}
          />
        );
      })}
    </div>
  );
}

// ============================================================================
// Step Content Component
// ============================================================================

interface StepContentProps {
  step: SolutionStep;
  onComplete?: () => void;
}

function StepContent({ step, onComplete }: StepContentProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!step.code) return;
    try {
      await navigator.clipboard.writeText(step.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  }, [step.code]);

  const getLanguage = () => {
    switch (step.codeLanguage) {
      case "bash":
        return "bash";
      case "python":
        return "python";
      case "env":
        return "env";
      default:
        return "bash";
    }
  };

  return (
    <div className="space-y-4">
      {/* Step Header */}
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
            "bg-zinc-100 text-zinc-900 text-[14px] font-bold"
          )}
        >
          {step.step}
        </div>
        <div>
          <h4 className="text-[16px] font-semibold text-zinc-100">
            {step.title}
          </h4>
          <p className="text-[13px] text-zinc-400 mt-1">{step.description}</p>
        </div>
      </div>

      {/* Warning */}
      {step.warning && (
        <div className="flex items-start gap-3 p-3 rounded-lg bg-red-950/30 border border-red-900/50">
          <AlertTriangle size={16} className="text-red-400 shrink-0 mt-0.5" />
          <p className="text-[12px] text-red-300">{step.warning}</p>
        </div>
      )}

      {/* Code Block */}
      {step.code && (
        <div className="relative group">
          <CodeBlock
            code={step.code}
            language={getLanguage()}
            showLineNumbers={false}
            showCopy={false}
          />
          <button
            onClick={handleCopy}
            className={cn(
              "absolute top-2 right-2 p-1.5 rounded transition-all",
              "opacity-0 group-hover:opacity-100",
              copied
                ? "bg-emerald-500/20 text-emerald-400"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            )}
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
        </div>
      )}

      {/* Tip */}
      {step.tip && (
        <div className="flex items-start gap-3 p-3 rounded-lg bg-blue-950/30 border border-blue-900/50">
          <Lightbulb size={16} className="text-blue-400 shrink-0 mt-0.5" />
          <p className="text-[12px] text-blue-300">{step.tip}</p>
        </div>
      )}

      {/* Mark as Done */}
      <button
        onClick={onComplete}
        className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] font-medium",
          "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 transition-colors"
        )}
      >
        <CheckCircle2 size={14} />
        Mark Step as Done
      </button>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function SolutionGuideDisplay({
  guide,
  onComplete,
  onClose,
  className,
}: SolutionGuideProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  const totalSteps = guide.steps.length;
  const currentStepData = guide.steps[currentStep - 1];
  const isLastStep = currentStep === totalSteps;
  const allCompleted = completedSteps.size === totalSteps;

  const handleStepComplete = useCallback(() => {
    setCompletedSteps((prev) => new Set(prev).add(currentStep));

    if (isLastStep) {
      onComplete?.();
    } else {
      setCurrentStep((prev) => prev + 1);
    }
  }, [currentStep, isLastStep, onComplete]);

  const handleNext = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <div className={cn("rounded-xl border border-zinc-800 overflow-hidden", className)}>
      {/* Header */}
      <div className="p-4 bg-zinc-900/50 border-b border-zinc-800">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0 pr-4">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-[16px] font-semibold text-zinc-100">
                {guide.title}
              </h3>
              <DifficultyBadge difficulty={guide.difficulty} />
            </div>
            <p className="text-[12px] text-zinc-500">{guide.description}</p>

            {/* Metadata */}
            <div className="flex items-center gap-4 mt-3">
              <span className="flex items-center gap-1.5 text-[11px] text-zinc-500">
                <Clock size={12} />
                {guide.estimatedTime}
              </span>
              <span className="text-[11px] text-zinc-500">
                {totalSteps} steps
              </span>
              {guide.relatedServices.length > 0 && (
                <div className="flex items-center gap-1">
                  {guide.relatedServices.slice(0, 2).map((service) => (
                    <span
                      key={service}
                      className="px-1.5 py-0.5 rounded text-[9px] bg-zinc-800 text-zinc-500"
                    >
                      {PROBE_METADATA[service].displayName}
                    </span>
                  ))}
                  {guide.relatedServices.length > 2 && (
                    <span className="text-[9px] text-zinc-600">
                      +{guide.relatedServices.length - 2}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Close button */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
            >
              <X size={16} />
            </button>
          )}
        </div>

        {/* Progress */}
        <div className="flex items-center justify-between mt-4">
          <ProgressIndicator
            currentStep={currentStep}
            totalSteps={totalSteps}
            completedSteps={completedSteps}
          />
          <span className="text-[11px] text-zinc-500">
            Step {currentStep} of {totalSteps}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <StepContent step={currentStepData} onComplete={handleStepComplete} />
      </div>

      {/* Footer Navigation */}
      <div className="flex items-center justify-between p-4 border-t border-zinc-800 bg-zinc-900/30">
        <button
          onClick={handlePrev}
          disabled={currentStep === 1}
          className={cn(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors",
            currentStep === 1
              ? "text-zinc-600 cursor-not-allowed"
              : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
          )}
        >
          <ChevronLeft size={14} />
          Previous
        </button>

        {allCompleted && (
          <span className="flex items-center gap-1.5 text-[12px] text-emerald-400">
            <CheckCircle2 size={14} />
            All steps completed!
          </span>
        )}

        <button
          onClick={handleNext}
          disabled={isLastStep}
          className={cn(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors",
            isLastStep
              ? "text-zinc-600 cursor-not-allowed"
              : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
          )}
        >
          Next
          <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Guide Card Component (for list view)
// ============================================================================

export interface GuideCardProps {
  guide: SolutionGuide;
  onClick?: () => void;
  className?: string;
}

export function GuideCard({ guide, onClick, className }: GuideCardProps) {
  return (
    <div
      className={cn(
        "group p-4 rounded-xl border transition-all duration-200 cursor-pointer",
        "border-zinc-800 hover:border-zinc-700 bg-zinc-900/30 hover:bg-zinc-900/50",
        className
      )}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick?.();
        }
      }}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0 pr-4">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-[14px] font-medium text-zinc-200 group-hover:text-white">
              {guide.title}
            </h4>
            <DifficultyBadge difficulty={guide.difficulty} />
          </div>
          <p className="text-[12px] text-zinc-500 line-clamp-2">
            {guide.description}
          </p>

          <div className="flex items-center gap-3 mt-2">
            <span className="flex items-center gap-1 text-[11px] text-zinc-600">
              <Clock size={10} />
              {guide.estimatedTime}
            </span>
            <span className="text-[11px] text-zinc-600">
              {guide.steps.length} steps
            </span>
          </div>
        </div>

        <ChevronRight
          size={16}
          className="text-zinc-600 group-hover:text-zinc-400 shrink-0 mt-1"
        />
      </div>
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

SolutionGuideDisplay.displayName = "SolutionGuideDisplay";
GuideCard.displayName = "GuideCard";

// ============================================================================
// Default Export
// ============================================================================

export default SolutionGuideDisplay;
