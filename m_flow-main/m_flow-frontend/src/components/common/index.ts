/**
 * Common Components Barrel Export
 *
 * Shared components used across the M-Flow frontend.
 *
 * Components:
 * - ErrorBoundary: Error handling wrapper
 * - Skeleton: Loading state indicators
 */

// Error Boundary
export {
  ErrorBoundary,
  ErrorFallback,
  InlineError,
  withErrorBoundary,
} from "./ErrorBoundary";

export type {
  ErrorBoundaryProps,
  ErrorFallbackProps,
  InlineErrorProps,
} from "./ErrorBoundary";

// Skeleton
export {
  Skeleton,
  SkeletonText,
  SkeletonAvatar,
  SkeletonButton,
  SkeletonCard,
  SkeletonStatusCard,
  SkeletonGrid,
  SkeletonList,
  SkeletonTable,
  SkeletonSection,
  SkeletonCompound,
} from "./Skeleton";

export type {
  SkeletonProps,
  SkeletonTextProps,
  SkeletonAvatarProps,
  SkeletonButtonProps,
  SkeletonCardProps,
  SkeletonStatusCardProps,
  SkeletonGridProps,
  SkeletonListProps,
  SkeletonTableProps,
  SkeletonSectionProps,
} from "./Skeleton";

// Command Palette
export { CommandPalette, useCommandPalette } from "./CommandPalette";

// Unavailable Page
export { UnavailablePage } from "./UnavailablePage";

// Progress Tracker
export { ProgressTracker, MultiProgressTracker } from "./ProgressTracker";
export type { ProgressTrackerProps, MultiProgressTrackerProps } from "./ProgressTracker";

// Ingestion Result
export { 
  IngestionResult, 
  PipelineStatusBadge, 
  getActionableErrorMessage 
} from "./IngestionResult";
export type { IngestionResultProps, PipelineStatusProps } from "./IngestionResult";

// Glossary Tooltip
export { 
  GlossaryTooltip, 
  Term, 
  GLOSSARY, 
  getTermDefinition 
} from "./GlossaryTooltip";
export type { GlossaryTooltipProps, TermProps, GlossaryKey } from "./GlossaryTooltip";

// Visual Feedback
export {
  FileTypeIcon,
  AnimatedProgress,
  SuccessAnimation,
  ErrorAnimation,
  WarningBanner,
  DropZone,
  LoadingDots,
  PulseRing,
} from "./VisualFeedback";
export type {
  FileTypeIconProps,
  AnimatedProgressProps,
  SuccessAnimationProps,
  ErrorAnimationProps,
  WarningBannerProps,
  DropZoneProps,
} from "./VisualFeedback";

// Prompt Editor
export { PromptEditor, InlinePromptEditor } from "./PromptEditor";

// Tag Input
export { TagInput, CompactTagInput } from "./TagInput";
