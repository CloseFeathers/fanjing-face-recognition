/**
 * QuickTest Module
 *
 * Exports all components for the Quick Test panel.
 *
 * Main Components:
 * - TestRunner: Main test panel with run controls
 * - TestItem: Individual test result display
 * - TestResults: Test summary and statistics
 *
 * Usage:
 * ```tsx
 * import { TestRunner } from "@/components/setup/QuickTest";
 *
 * <TestRunner
 *   autoRun={false}
 *   onComplete={(results) => console.log(results)}
 * />
 * ```
 */

// Main component
export { TestRunner, TestRunnerSkeleton } from "./TestRunner";
export type { TestRunnerProps } from "./TestRunner";

// Individual test item
export { TestItem, TestItemSkeleton } from "./TestItem";
export type { TestItemProps } from "./TestItem";

// Results summary
export { TestResults, CompactResults } from "./TestResults";
export type { TestResultsProps, CompactResultsProps } from "./TestResults";
