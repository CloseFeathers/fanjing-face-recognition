/**
 * Troubleshooting Module
 *
 * Exports all components for the Troubleshooting section.
 *
 * Main Components:
 * - TroubleshootingPanel: Main container with tabs
 * - IssueCard: Common issue display
 * - DiagnosticTool: System diagnostics
 * - SolutionGuideDisplay: Step-by-step guides
 * - FAQSection: Frequently asked questions
 *
 * Usage:
 * ```tsx
 * import { TroubleshootingPanel } from "@/components/setup/Troubleshooting";
 *
 * <TroubleshootingPanel onIssueResolved={(id) => console.log(id)} />
 * ```
 */

// Main container
export { TroubleshootingPanel } from "./TroubleshootingPanel";
export type { TroubleshootingPanelProps } from "./TroubleshootingPanel";

// Issue components
export { IssueCard, IssueList } from "./IssueCard";
export type { IssueCardProps, IssueListProps } from "./IssueCard";

// Diagnostic components
export { DiagnosticTool } from "./DiagnosticTool";
export type { DiagnosticToolProps, DiagnosticResultExtended } from "./DiagnosticTool";

// Solution guide components
export { SolutionGuideDisplay, GuideCard } from "./SolutionGuide";
export type { SolutionGuideProps, GuideCardProps } from "./SolutionGuide";

// FAQ components
export { FAQSection, FAQItem } from "./FAQSection";
export type { FAQSectionProps, FAQItemProps } from "./FAQSection";
