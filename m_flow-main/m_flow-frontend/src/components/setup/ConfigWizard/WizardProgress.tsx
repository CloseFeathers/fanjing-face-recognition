"use client";

/**
 * WizardProgress Component
 *
 * Displays the step indicator for the configuration wizard.
 * Shows all steps with their current status and allows navigation
 * to completed/current steps.
 *
 * Features:
 * - Visual step indicator with status icons
 * - Progress line connecting steps
 * - Click to navigate to completed steps
 * - Responsive layout (horizontal on desktop, vertical on mobile)
 *
 * @example
 * <WizardProgress
 *   steps={wizardSteps}
 *   currentStepIndex={2}
 *   onStepClick={(index) => goToStep(index)}
 * />
 */

import React from "react";
import { cn } from "@/lib/utils";
import { Check, AlertCircle, Circle, SkipForward } from "lucide-react";
import type { ConfigStep, StepStatus } from "@/types/setup";

// ============================================================================
// Types
// ============================================================================

export interface WizardProgressProps {
  /** All wizard steps */
  steps: ConfigStep[];
  /** Current active step index */
  currentStepIndex: number;
  /** Handler for step navigation */
  onStepClick?: (index: number) => void;
  /** Layout orientation */
  orientation?: "horizontal" | "vertical";
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Status Icon Component
// ============================================================================

interface StatusIconProps {
  status: StepStatus;
  stepNumber: number;
  size?: "sm" | "md";
}

function StatusIcon({ status, stepNumber, size = "md" }: StatusIconProps) {
  const sizeClasses = size === "sm" ? "w-6 h-6" : "w-8 h-8";
  const iconSize = size === "sm" ? 12 : 16;
  const textSize = size === "sm" ? "text-[10px]" : "text-[12px]";

  const baseClasses = cn(
    "flex items-center justify-center rounded-full border-2 transition-all duration-200",
    sizeClasses
  );

  switch (status) {
    case "completed":
      return (
        <div
          className={cn(
            baseClasses,
            "bg-emerald-500 border-emerald-500 text-white"
          )}
        >
          <Check size={iconSize} strokeWidth={3} />
        </div>
      );
    case "current":
      return (
        <div
          className={cn(
            baseClasses,
            "bg-zinc-900 border-zinc-100 text-zinc-100"
          )}
        >
          <span className={cn("font-semibold", textSize)}>{stepNumber}</span>
        </div>
      );
    case "error":
      return (
        <div
          className={cn(baseClasses, "bg-red-500/20 border-red-500 text-red-400")}
        >
          <AlertCircle size={iconSize} />
        </div>
      );
    case "skipped":
      return (
        <div
          className={cn(
            baseClasses,
            "bg-zinc-800 border-zinc-600 text-zinc-500"
          )}
        >
          <SkipForward size={iconSize - 2} />
        </div>
      );
    case "pending":
    default:
      return (
        <div
          className={cn(
            baseClasses,
            "bg-zinc-900 border-zinc-700 text-zinc-500"
          )}
        >
          <span className={cn("font-medium", textSize)}>{stepNumber}</span>
        </div>
      );
  }
}

// ============================================================================
// Progress Line Component
// ============================================================================

interface ProgressLineProps {
  isComplete: boolean;
  orientation: "horizontal" | "vertical";
}

function ProgressLine({ isComplete, orientation }: ProgressLineProps) {
  if (orientation === "horizontal") {
    return (
      <div className="flex-1 h-0.5 mx-2">
        <div
          className={cn(
            "h-full transition-all duration-300",
            isComplete ? "bg-emerald-500" : "bg-zinc-700"
          )}
        />
      </div>
    );
  }

  return (
    <div className="w-0.5 h-8 mx-auto">
      <div
        className={cn(
          "w-full h-full transition-all duration-300",
          isComplete ? "bg-emerald-500" : "bg-zinc-700"
        )}
      />
    </div>
  );
}

// ============================================================================
// Step Item Component
// ============================================================================

interface StepItemProps {
  step: ConfigStep;
  index: number;
  isClickable: boolean;
  onClick?: () => void;
  orientation: "horizontal" | "vertical";
}

function StepItem({
  step,
  index,
  isClickable,
  onClick,
  orientation,
}: StepItemProps) {
  const stepNumber = index + 1;

  if (orientation === "horizontal") {
    return (
      <div
        className={cn(
          "flex flex-col items-center gap-2",
          isClickable && "cursor-pointer group"
        )}
        onClick={isClickable ? onClick : undefined}
        role={isClickable ? "button" : undefined}
        tabIndex={isClickable ? 0 : undefined}
        onKeyDown={
          isClickable
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onClick?.();
                }
              }
            : undefined
        }
      >
        {/* Status Icon */}
        <div
          className={cn(
            "transition-transform duration-200",
            isClickable && "group-hover:scale-110"
          )}
        >
          <StatusIcon status={step.status} stepNumber={stepNumber} size="md" />
        </div>

        {/* Step Info */}
        <div className="text-center">
          <p
            className={cn(
              "text-[12px] font-medium transition-colors",
              step.status === "current"
                ? "text-zinc-100"
                : step.status === "completed"
                ? "text-emerald-400"
                : step.status === "error"
                ? "text-red-400"
                : "text-zinc-500",
              isClickable && "group-hover:text-zinc-200"
            )}
          >
            {step.title}
          </p>
          {step.isOptional && (
            <p className="text-[10px] text-zinc-600">Has defaults</p>
          )}
        </div>
      </div>
    );
  }

  // Vertical layout
  return (
    <div
      className={cn(
        "flex items-start gap-4 py-2",
        isClickable && "cursor-pointer group"
      )}
      onClick={isClickable ? onClick : undefined}
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
      onKeyDown={
        isClickable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
    >
      {/* Status Icon */}
      <div
        className={cn(
          "shrink-0 transition-transform duration-200",
          isClickable && "group-hover:scale-110"
        )}
      >
        <StatusIcon status={step.status} stepNumber={stepNumber} size="sm" />
      </div>

      {/* Step Info */}
      <div className="min-w-0">
        <p
          className={cn(
            "text-[13px] font-medium transition-colors",
            step.status === "current"
              ? "text-zinc-100"
              : step.status === "completed"
              ? "text-emerald-400"
              : step.status === "error"
              ? "text-red-400"
              : "text-zinc-500",
            isClickable && "group-hover:text-zinc-200"
          )}
        >
          {step.title}
          {step.isOptional && (
            <span className="ml-2 text-[10px] text-zinc-600">(has defaults)</span>
          )}
        </p>
        <p className="text-[11px] text-zinc-600 mt-0.5 truncate">
          {step.description}
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function WizardProgress({
  steps,
  currentStepIndex,
  onStepClick,
  orientation = "horizontal",
  className,
}: WizardProgressProps) {
  // Determine if a step is clickable (completed or current)
  const isStepClickable = (index: number, status: StepStatus): boolean => {
    if (!onStepClick) return false;
    // Can only click on completed steps or the current step
    return status === "completed" || index <= currentStepIndex;
  };

  if (orientation === "horizontal") {
    return (
      <div className={cn("w-full", className)}>
        {/* Desktop: Horizontal layout */}
        <div className="hidden sm:flex items-start justify-between">
          {steps.map((step, index) => (
            <React.Fragment key={step.id}>
              {/* Step Item */}
              <StepItem
                step={step}
                index={index}
                isClickable={isStepClickable(index, step.status)}
                onClick={() => onStepClick?.(index)}
                orientation="horizontal"
              />

              {/* Progress Line (between steps) */}
              {index < steps.length - 1 && (
                <ProgressLine
                  isComplete={
                    step.status === "completed" || step.status === "skipped"
                  }
                  orientation="horizontal"
                />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Mobile: Compact indicator */}
        <div className="sm:hidden">
          <div className="flex items-center justify-center gap-2 mb-4">
            {steps.map((step, index) => (
              <div
                key={step.id}
                className={cn(
                  "w-2 h-2 rounded-full transition-all duration-200",
                  step.status === "completed" && "bg-emerald-500",
                  step.status === "current" && "bg-zinc-100 scale-125",
                  step.status === "error" && "bg-red-500",
                  step.status === "skipped" && "bg-zinc-600",
                  step.status === "pending" && "bg-zinc-700"
                )}
              />
            ))}
          </div>
          <div className="text-center">
            <p className="text-[14px] font-medium text-zinc-100">
              Step {currentStepIndex + 1} of {steps.length}
            </p>
            <p className="text-[12px] text-zinc-500">
              {steps[currentStepIndex]?.title}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Vertical layout
  return (
    <div className={cn("space-y-1", className)}>
      {steps.map((step, index) => (
        <React.Fragment key={step.id}>
          <StepItem
            step={step}
            index={index}
            isClickable={isStepClickable(index, step.status)}
            onClick={() => onStepClick?.(index)}
            orientation="vertical"
          />

          {/* Progress Line (between steps) */}
          {index < steps.length - 1 && (
            <div className="pl-3">
              <ProgressLine
                isComplete={
                  step.status === "completed" || step.status === "skipped"
                }
                orientation="vertical"
              />
            </div>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

// ============================================================================
// Display Name
// ============================================================================

WizardProgress.displayName = "WizardProgress";

// ============================================================================
// Default Export
// ============================================================================

export default WizardProgress;
