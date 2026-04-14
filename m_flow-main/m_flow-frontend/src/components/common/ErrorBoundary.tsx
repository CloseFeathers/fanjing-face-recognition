"use client";

/**
 * ErrorBoundary Component
 *
 * Catches JavaScript errors in child component trees, logs them,
 * and displays a fallback UI instead of crashing the whole app.
 *
 * Features:
 * - Graceful error handling
 * - Customizable fallback UI
 * - Error reporting callback
 * - Recovery options
 * - Detailed error display (dev mode)
 *
 * @example
 * <ErrorBoundary fallback={<ErrorFallback />}>
 *   <MyComponent />
 * </ErrorBoundary>
 */

import React, { Component, ErrorInfo, ReactNode } from "react";
import { cn } from "@/lib/utils";
import { AlertTriangle, RefreshCw, Copy, Check, ChevronDown } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface ErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Custom fallback component */
  fallback?: ReactNode;
  /** Callback when error occurs */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Whether to show error details */
  showDetails?: boolean;
  /** Component name for error messages */
  componentName?: string;
  /** Additional CSS classes */
  className?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

// ============================================================================
// Error Fallback Component
// ============================================================================

export interface ErrorFallbackProps {
  /** The error that was caught */
  error: Error | null;
  /** Error info from React */
  errorInfo: ErrorInfo | null;
  /** Reset the error boundary */
  onReset?: () => void;
  /** Whether to show error details */
  showDetails?: boolean;
  /** Component name */
  componentName?: string;
  /** Additional CSS classes */
  className?: string;
}

export function ErrorFallback({
  error,
  errorInfo,
  onReset,
  showDetails = false,
  componentName,
  className,
}: ErrorFallbackProps) {
  const [copied, setCopied] = React.useState(false);
  const [detailsOpen, setDetailsOpen] = React.useState(false);

  const handleCopy = async () => {
    const errorText = `
Error: ${error?.message || "Unknown error"}
Component: ${componentName || "Unknown"}
Stack: ${error?.stack || "No stack trace"}
Component Stack: ${errorInfo?.componentStack || "No component stack"}
    `.trim();

    try {
      await navigator.clipboard.writeText(errorText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error("Failed to copy error:", e);
    }
  };

  return (
    <div
      className={cn(
        "p-6 rounded-xl border border-red-900/50 bg-red-950/20",
        className
      )}
      role="alert"
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center shrink-0">
          <AlertTriangle size={20} className="text-red-400" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-[16px] font-semibold text-red-300">
            Something went wrong
          </h3>
          <p className="text-[13px] text-red-400/80 mt-1">
            {componentName
              ? `An error occurred in ${componentName}`
              : "An unexpected error occurred"}
          </p>
        </div>
      </div>

      {/* Error Message */}
      <div className="mt-4 p-3 rounded-lg bg-red-950/50 border border-red-900/30">
        <p className="text-[12px] font-mono text-red-300 break-all">
          {error?.message || "Unknown error"}
        </p>
      </div>

      {/* Actions */}
      <div className="mt-4 flex items-center gap-3">
        {onReset && (
          <button
            onClick={onReset}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-lg text-[13px] font-medium",
              "bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-colors"
            )}
          >
            <RefreshCw size={14} />
            Try Again
          </button>
        )}
        <button
          onClick={handleCopy}
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg text-[12px]",
            "text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors"
          )}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "Copied!" : "Copy Error"}
        </button>
      </div>

      {/* Details Toggle */}
      {showDetails && (error?.stack || errorInfo?.componentStack) && (
        <div className="mt-4">
          <button
            onClick={() => setDetailsOpen(!detailsOpen)}
            className={cn(
              "flex items-center gap-2 text-[11px] text-red-400 hover:text-red-300"
            )}
          >
            <ChevronDown
              size={12}
              className={cn("transition-transform", detailsOpen && "rotate-180")}
            />
            {detailsOpen ? "Hide Details" : "Show Details"}
          </button>

          {detailsOpen && (
            <div className="mt-3 space-y-3">
              {error?.stack && (
                <div>
                  <p className="text-[10px] text-red-500 uppercase tracking-wider mb-1">
                    Stack Trace
                  </p>
                  <pre className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-[10px] text-red-300/80 overflow-x-auto whitespace-pre-wrap">
                    {error.stack}
                  </pre>
                </div>
              )}
              {errorInfo?.componentStack && (
                <div>
                  <p className="text-[10px] text-red-500 uppercase tracking-wider mb-1">
                    Component Stack
                  </p>
                  <pre className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-[10px] text-red-300/80 overflow-x-auto whitespace-pre-wrap">
                    {errorInfo.componentStack}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Inline Error Component (for smaller areas)
// ============================================================================

export interface InlineErrorProps {
  message: string;
  onRetry?: () => void;
  className?: string;
}

export function InlineError({ message, onRetry, className }: InlineErrorProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 p-3 rounded-lg",
        "bg-red-950/20 border border-red-900/50",
        className
      )}
      role="alert"
    >
      <AlertTriangle size={16} className="text-red-400 shrink-0" />
      <p className="text-[12px] text-red-300 flex-1">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="p-1.5 rounded text-red-400 hover:text-red-300 hover:bg-red-500/20 transition-colors"
          title="Retry"
        >
          <RefreshCw size={14} />
        </button>
      )}
    </div>
  );
}

// ============================================================================
// Main ErrorBoundary Class Component
// ============================================================================

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console
    console.error("ErrorBoundary caught an error:", error, errorInfo);

    // Update state with error info
    this.setState({ errorInfo });

    // Call optional error callback
    this.props.onError?.(error, errorInfo);
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const {
      children,
      fallback,
      showDetails = process.env.NODE_ENV === "development",
      componentName,
      className,
    } = this.props;

    if (hasError) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Default error fallback
      return (
        <ErrorFallback
          error={error}
          errorInfo={errorInfo}
          onReset={this.handleReset}
          showDetails={showDetails}
          componentName={componentName}
          className={className}
        />
      );
    }

    return children;
  }
}

// ============================================================================
// HOC for wrapping components with error boundary
// ============================================================================

export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, "children">
) {
  const displayName =
    WrappedComponent.displayName || WrappedComponent.name || "Component";

  const ComponentWithErrorBoundary = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps} componentName={displayName}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  ComponentWithErrorBoundary.displayName = `withErrorBoundary(${displayName})`;

  return ComponentWithErrorBoundary;
}

// ============================================================================
// Display Names
// ============================================================================

ErrorFallback.displayName = "ErrorFallback";
InlineError.displayName = "InlineError";

// ============================================================================
// Default Export
// ============================================================================

export default ErrorBoundary;
