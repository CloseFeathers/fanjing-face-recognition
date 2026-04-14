/**
 * Setup Components Barrel Export
 * 
 * This module provides all components for the Setup page:
 * 
 * Main Components:
 * - SetupPage: The main setup interface
 * 
 * System Status:
 * - StatusIndicator: Small dot indicator
 * - StatusCard: Individual service health card
 * - StatusDashboard: Full health dashboard
 * 
 * Config Wizard:
 * - WizardContainer: Main wizard orchestrator
 * - WizardProgress: Step progress indicator
 * - WizardNavigation: Navigation buttons
 * - ConfigBadge: Configuration type indicator
 * - Step components: LLM, Embedding, Database, Storage, Review
 * 
 * Quick Test:
 * - TestRunner: Main test panel with run controls
 * - TestItem: Individual test result display
 * - TestResults: Test summary and statistics
 * 
 * Getting Started:
 * - GettingStartedPanel: Main container with tabs
 * - TutorialCard: Tutorial display cards
 * - CodeExample: Code snippets with highlighting
 * - EnvFileHelper: Environment configuration reference
 * - QuickActions: Navigation shortcuts
 * 
 * Troubleshooting:
 * - TroubleshootingPanel: Main container with tabs
 * - IssueCard: Common issue display
 * - DiagnosticTool: System diagnostics runner
 * - SolutionGuideDisplay: Step-by-step guides
 * - FAQSection: Frequently asked questions
 */

// Main page component
export { SetupPage } from "./SetupPage";
export type { SetupPageProps } from "./SetupPage";

// System Status components
export {
  StatusIndicator,
  StatusCard,
  StatusCardSkeleton,
  StatusDashboard,
} from "./SystemStatus";

export type {
  StatusIndicatorProps,
  IndicatorSize,
  StatusCardProps,
  StatusDashboardProps,
} from "./SystemStatus";

// Config Wizard components
export {
  WizardContainer,
  WizardProgress,
  WizardNavigation,
  CompactNavigation,
  ConfigBadge,
  ConfigBadgeInline,
  LLMConfigStep,
  EmbeddingConfigStep,
  DatabaseConfigStep,
  StorageConfigStep,
  ReviewStep,
} from "./ConfigWizard";

export type {
  WizardContainerProps,
  WizardProgressProps,
  WizardNavigationProps,
  CompactNavigationProps,
  ConfigBadgeProps,
  BadgeType,
  LLMConfigStepProps,
  EmbeddingConfigStepProps,
  DatabaseConfigStepProps,
  StorageConfigStepProps,
  ReviewStepProps,
} from "./ConfigWizard";

// Quick Test components
export {
  TestRunner,
  TestRunnerSkeleton,
  TestItem,
  TestItemSkeleton,
  TestResults,
  CompactResults,
} from "./QuickTest";

export type {
  TestRunnerProps,
  TestItemProps,
  TestResultsProps,
  CompactResultsProps,
} from "./QuickTest";

// Getting Started components
export {
  GettingStartedPanel,
  TutorialCard,
  TutorialGrid,
  CodeBlock,
  CodeExample,
  CodeExampleTabs,
  CodeSnippet,
  EnvFileHelper,
  EnvVariableRow,
  QuickActions,
  ActionCard,
} from "./GettingStarted";

export type {
  GettingStartedPanelProps,
  TutorialCardProps,
  TutorialGridProps,
  CodeBlockProps,
  CodeExampleProps,
  CodeExampleTabsProps,
  CodeSnippetProps,
  EnvFileHelperProps,
  EnvVariableRowProps,
  QuickActionsProps,
  ActionCardProps,
} from "./GettingStarted";

// Troubleshooting components
export {
  TroubleshootingPanel,
  IssueCard,
  IssueList,
  DiagnosticTool,
  SolutionGuideDisplay,
  GuideCard,
  FAQSection,
  FAQItem,
} from "./Troubleshooting";

export type {
  TroubleshootingPanelProps,
  IssueCardProps,
  IssueListProps,
  DiagnosticToolProps,
  DiagnosticResultExtended,
  SolutionGuideProps,
  GuideCardProps,
  FAQSectionProps,
  FAQItemProps,
} from "./Troubleshooting";
