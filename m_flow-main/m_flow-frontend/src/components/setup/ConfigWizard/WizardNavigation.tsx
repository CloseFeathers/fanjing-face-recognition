"use client";

/**
 * WizardNavigation Component
 *
 * Navigation buttons for the configuration wizard.
 * Provides Back, Skip, and Continue/Save actions.
 *
 * Features:
 * - Consistent button styling
 * - Loading states for async operations
 * - Skip option for optional steps
 * - Keyboard accessible
 *
 * @example
 * <WizardNavigation
 *   onBack={() => wizard.goBack()}
 *   onNext={() => wizard.goNext()}
 *   onSkip={() => wizard.skipStep()}
 *   isFirstStep={false}
 *   isLastStep={false}
 *   canSkip={true}
 *   isLoading={false}
 * />
 */

import React from "react";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight, SkipForward, Loader2, Check } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface WizardNavigationProps {
  /** Handler for back navigation */
  onBack?: () => void;
  /** Handler for next/continue */
  onNext?: () => void;
  /** Handler for skip (optional steps) */
  onSkip?: () => void;
  /** Whether this is the first step */
  isFirstStep?: boolean;
  /** Whether this is the last step */
  isLastStep?: boolean;
  /** Whether the current step can be skipped */
  canSkip?: boolean;
  /** Whether an operation is in progress */
  isLoading?: boolean;
  /** Custom label for the next button */
  nextLabel?: string;
  /** Whether the next button should be disabled */
  disableNext?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Button Component
// ============================================================================

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

function Button({
  variant = "secondary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  children,
  className,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        // Base styles
        "inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200",
        "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-900",
        // Size variants
        size === "sm" && "px-3 py-1.5 text-[12px] gap-1.5",
        size === "md" && "px-4 py-2 text-[13px] gap-2",
        // Variant styles
        variant === "primary" &&
          "bg-zinc-100 text-zinc-900 hover:bg-white focus:ring-zinc-400 disabled:bg-zinc-700 disabled:text-zinc-400",
        variant === "secondary" &&
          "bg-zinc-800 text-zinc-200 border border-zinc-700 hover:bg-zinc-700 hover:border-zinc-600 focus:ring-zinc-500 disabled:bg-zinc-900 disabled:text-zinc-600 disabled:border-zinc-800",
        variant === "ghost" &&
          "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 focus:ring-zinc-500 disabled:text-zinc-600",
        // Disabled state
        (disabled || isLoading) && "cursor-not-allowed opacity-60",
        // Custom classes
        className
      )}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 size={14} className="animate-spin" />
      ) : (
        leftIcon
      )}
      {children}
      {!isLoading && rightIcon}
    </button>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function WizardNavigation({
  onBack,
  onNext,
  onSkip,
  isFirstStep = false,
  isLastStep = false,
  canSkip = false,
  isLoading = false,
  nextLabel,
  disableNext = false,
  className,
}: WizardNavigationProps) {
  // Determine the next button label
  const getNextLabel = (): string => {
    if (nextLabel) return nextLabel;
    if (isLastStep) return "Complete";
    return "Continue";
  };

  return (
    <div
      className={cn(
        "flex items-center justify-between pt-6 border-t border-zinc-800",
        className
      )}
    >
      {/* Left side: Back button */}
      <div>
        {!isFirstStep && (
          <Button
            variant="ghost"
            onClick={onBack}
            disabled={isLoading}
            leftIcon={<ChevronLeft size={16} />}
          >
            Back
          </Button>
        )}
      </div>

      {/* Right side: Skip and Next buttons */}
      <div className="flex items-center gap-3">
        {/* Skip button (for optional steps) */}
        {canSkip && !isLastStep && (
          <Button
            variant="ghost"
            onClick={onSkip}
            disabled={isLoading}
            rightIcon={<SkipForward size={14} />}
          >
            Skip
          </Button>
        )}

        {/* Next/Continue/Complete button */}
        <Button
          variant="primary"
          onClick={onNext}
          isLoading={isLoading}
          disabled={disableNext}
          rightIcon={
            isLastStep ? (
              <Check size={16} />
            ) : (
              <ChevronRight size={16} />
            )
          }
        >
          {getNextLabel()}
        </Button>
      </div>
    </div>
  );
}

// ============================================================================
// Compact Navigation (for inline use)
// ============================================================================

export interface CompactNavigationProps {
  onBack?: () => void;
  onNext?: () => void;
  showBack?: boolean;
  showNext?: boolean;
  nextLabel?: string;
  isLoading?: boolean;
  className?: string;
}

export function CompactNavigation({
  onBack,
  onNext,
  showBack = true,
  showNext = true,
  nextLabel = "Next",
  isLoading = false,
  className,
}: CompactNavigationProps) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {showBack && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          disabled={isLoading}
          leftIcon={<ChevronLeft size={14} />}
        >
          Back
        </Button>
      )}
      {showNext && (
        <Button
          variant="primary"
          size="sm"
          onClick={onNext}
          isLoading={isLoading}
          rightIcon={<ChevronRight size={14} />}
        >
          {nextLabel}
        </Button>
      )}
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

WizardNavigation.displayName = "WizardNavigation";
CompactNavigation.displayName = "CompactNavigation";

// ============================================================================
// Default Export
// ============================================================================

export default WizardNavigation;
