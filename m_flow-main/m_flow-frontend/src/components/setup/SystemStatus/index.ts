/**
 * SystemStatus Components Barrel Export
 * 
 * Provides components for displaying system health status:
 * - StatusIndicator: Small dot indicator for health verdict
 * - StatusCard: Card displaying single service health
 * - StatusDashboard: Full dashboard with all services
 */

export { StatusIndicator } from "./StatusIndicator";
export type { StatusIndicatorProps, IndicatorSize } from "./StatusIndicator";

export { StatusCard, StatusCardSkeleton } from "./StatusCard";
export type { StatusCardProps } from "./StatusCard";

export { StatusDashboard } from "./StatusDashboard";
export type { StatusDashboardProps } from "./StatusDashboard";
