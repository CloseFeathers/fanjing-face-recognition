/**
 * Setup Page Type Definitions
 * 
 * This file contains all type definitions specific to the Setup page.
 * For health-related types, see index.ts (HealthVerdict, ProbeResult, etc.)
 */

import type { HealthVerdict, ProbeKey, ProbeResult, HealthProbes } from "./index";

// ============================================================================
// Setup Page State
// ============================================================================

/**
 * Overall setup completion status
 */
export type SetupStatus = "not_started" | "in_progress" | "completed" | "error";

/**
 * Individual setup step status
 */
export type StepStatus = "pending" | "current" | "completed" | "skipped" | "error";

/**
 * Configuration step identifier
 */
export type ConfigStepId = 
  | "llm"        // LLM Provider (editable)
  | "embedding"  // Embedding Service (read-only)
  | "database"   // Database Configuration (read-only)
  | "storage"    // Storage Configuration (read-only)
  | "review";    // Review & Test

/**
 * Configuration step definition
 */
export interface ConfigStep {
  /** Unique step identifier */
  id: ConfigStepId;
  /** Display title */
  title: string;
  /** Brief description */
  description: string;
  /** Whether this step's config is editable via API */
  isEditable: boolean;
  /** Current step status */
  status: StepStatus;
  /** Whether this step is optional */
  isOptional: boolean;
}

/**
 * Setup wizard state
 */
export interface SetupWizardState {
  /** Current active step index (0-based) */
  currentStepIndex: number;
  /** All configuration steps */
  steps: ConfigStep[];
  /** Whether the wizard is currently saving */
  isSaving: boolean;
  /** Whether the wizard is currently testing */
  isTesting: boolean;
  /** Last error message, if any */
  lastError: string | null;
}

// ============================================================================
// System Status Display
// ============================================================================

/**
 * Aggregated system status for dashboard display
 */
export interface SystemStatusSummary {
  /** Overall system health verdict */
  overallVerdict: HealthVerdict;
  /** Count of healthy services */
  healthyCount: number;
  /** Count of degraded services */
  degradedCount: number;
  /** Count of offline services */
  offlineCount: number;
  /** Total number of services */
  totalCount: number;
  /** System version */
  version: string;
  /** Uptime in seconds */
  uptimeSeconds: number;
  /** Last check timestamp */
  lastCheckedAt: string;
}

/**
 * Individual service status for display
 */
export interface ServiceStatus {
  /** Service key (matches ProbeKey) */
  key: ProbeKey;
  /** Display name */
  displayName: string;
  /** Brief description */
  description: string;
  /** Whether this is a critical service */
  isCritical: boolean;
  /** Current probe result */
  probe: ProbeResult;
}

// ============================================================================
// Quick Test
// ============================================================================

/**
 * Test status for a single test item
 */
export type TestStatus = "idle" | "running" | "passed" | "failed" | "skipped";

/**
 * Individual test result
 */
export interface TestResult {
  /** Test identifier (matches ProbeKey) */
  id: ProbeKey;
  /** Display name */
  name: string;
  /** Test status */
  status: TestStatus;
  /** Execution time in milliseconds */
  durationMs: number | null;
  /** Result message or error */
  message: string | null;
  /** Backend/provider name */
  backend: string | null;
}

/**
 * Quick test runner state
 */
export interface QuickTestState {
  /** All test results */
  results: TestResult[];
  /** Whether tests are currently running */
  isRunning: boolean;
  /** Last run timestamp */
  lastRunAt: Date | null;
  /** Current test being executed (index) */
  currentTestIndex: number | null;
}

// ============================================================================
// Getting Started
// ============================================================================

/**
 * Tutorial category
 */
export type TutorialCategory = 
  | "quickstart"
  | "api"
  | "configuration"
  | "advanced";

/**
 * Tutorial option
 */
export interface TutorialOption {
  /** Tutorial identifier */
  id: string;
  /** Display title */
  title: string;
  /** Brief description */
  description: string;
  /** Estimated duration */
  durationMinutes: number;
  /** Difficulty level */
  difficulty: "beginner" | "intermediate" | "advanced";
  /** Whether this tutorial is interactive */
  isInteractive: boolean;
  /** Category */
  category: TutorialCategory;
  /** Icon name (lucide icon) */
  icon?: string;
  /** External documentation link */
  docLink?: string;
}

/**
 * Code example language
 */
export type CodeLanguage = "python" | "bash" | "typescript" | "json" | "env";

/**
 * Code example with metadata
 */
export interface CodeExample {
  /** Example identifier */
  id: string;
  /** Display title */
  title: string;
  /** Brief description */
  description: string;
  /** Programming language */
  language: CodeLanguage;
  /** Code content */
  code: string;
  /** Whether to show line numbers */
  showLineNumbers?: boolean;
  /** Highlighted lines (1-indexed) */
  highlightLines?: number[];
  /** Output example (if applicable) */
  output?: string;
}

/**
 * Environment variable definition
 */
export interface EnvVariable {
  /** Variable name */
  key: string;
  /** Description */
  description: string;
  /** Example value */
  example: string;
  /** Whether required */
  required: boolean;
  /** Related service */
  service?: ProbeKey;
  /** Default value (if any) */
  defaultValue?: string;
  /** Sensitive (should be masked) */
  sensitive?: boolean;
}

/**
 * Environment configuration section
 */
export interface EnvSection {
  /** Section identifier */
  id: string;
  /** Section title */
  title: string;
  /** Section description */
  description: string;
  /** Variables in this section */
  variables: EnvVariable[];
}

/**
 * Quick action for getting started
 */
export interface QuickAction {
  /** Action identifier */
  id: string;
  /** Display title */
  title: string;
  /** Brief description */
  description: string;
  /** Icon name */
  icon: string;
  /** Navigation link */
  href: string;
  /** Whether external link */
  external?: boolean;
}

// ============================================================================
// Troubleshooting
// ============================================================================

/**
 * Common issue definition
 */
export interface CommonIssue {
  /** Issue identifier */
  id: string;
  /** Issue title */
  title: string;
  /** Problem description */
  problem: string;
  /** How to detect this issue */
  detection: string;
  /** Solution steps */
  solution: string[];
  /** Related service keys */
  relatedServices: ProbeKey[];
}

/**
 * Diagnostic result
 */
export interface DiagnosticResult {
  /** Diagnostic name */
  name: string;
  /** Pass/fail status */
  passed: boolean;
  /** Details or error message */
  details: string;
  /** Suggested action if failed */
  suggestion?: string;
}

// ============================================================================
// Setup Page Props
// ============================================================================

/**
 * Props for StatusCard component
 */
export interface StatusCardProps {
  /** Service key */
  serviceKey: ProbeKey;
  /** Probe result from health check */
  probe: ProbeResult;
  /** Whether to show expanded details */
  showDetails?: boolean;
  /** Click handler */
  onClick?: () => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Props for StatusDashboard component
 */
export interface StatusDashboardProps {
  /** Optional callback when refresh is clicked */
  onRefresh?: () => void;
  /** Whether to enable auto-refresh */
  autoRefresh?: boolean;
  /** Auto-refresh interval in milliseconds */
  autoRefreshInterval?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Props for ConfigWizard component
 */
export interface ConfigWizardProps {
  /** Callback when wizard is completed */
  onComplete?: () => void;
  /** Callback when wizard is skipped */
  onSkip?: () => void;
  /** Initial step index */
  initialStepIndex?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Props for QuickTest component
 */
export interface QuickTestProps {
  /** Callback when all tests complete */
  onComplete?: (results: TestResult[]) => void;
  /** Whether to auto-run tests on mount */
  autoRun?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Section visibility state for SetupPage
 */
export interface SectionVisibility {
  status: boolean;
  wizard: boolean;
  quickTest: boolean;
  gettingStarted: boolean;
  troubleshooting: boolean;
}

/**
 * Setup page section identifier
 */
export type SetupSection = 
  | "status"
  | "wizard"
  | "quickTest"
  | "gettingStarted"
  | "troubleshooting";

/**
 * Re-export health types for convenience
 */
export type {
  HealthVerdict,
  ProbeKey,
  ProbeResult,
  HealthProbes,
} from "./index";
