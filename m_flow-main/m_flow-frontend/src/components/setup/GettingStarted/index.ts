/**
 * GettingStarted Module
 *
 * Exports all components for the Getting Started section.
 *
 * Main Components:
 * - GettingStartedPanel: Main container with tabs
 * - TutorialCard: Individual tutorial display
 * - CodeExample: Code snippet with syntax highlighting
 * - EnvFileHelper: Environment variable reference
 * - QuickActions: Navigation shortcuts
 *
 * Usage:
 * ```tsx
 * import { GettingStartedPanel } from "@/components/setup/GettingStarted";
 *
 * <GettingStartedPanel onTutorialSelect={(t) => console.log(t)} />
 * ```
 */

// Main container
export { GettingStartedPanel } from "./GettingStartedPanel";
export type { GettingStartedPanelProps } from "./GettingStartedPanel";

// Tutorial components
export { TutorialCard, TutorialGrid } from "./TutorialCard";
export type { TutorialCardProps, TutorialGridProps } from "./TutorialCard";

// Code example components
export {
  CodeBlock,
  CodeExample,
  CodeExampleTabs,
  CodeSnippet,
} from "./CodeExample";
export type {
  CodeBlockProps,
  CodeExampleProps,
  CodeExampleTabsProps,
  CodeSnippetProps,
} from "./CodeExample";

// Environment helper
export { EnvFileHelper, EnvVariableRow } from "./EnvFileHelper";
export type { EnvFileHelperProps, EnvVariableRowProps } from "./EnvFileHelper";

// Quick actions
export { QuickActions, ActionCard } from "./QuickActions";
export type { QuickActionsProps, ActionCardProps } from "./QuickActions";
