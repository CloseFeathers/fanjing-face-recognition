/**
 * React Query hook for fetching active pipeline operations.
 * 
 * Provides real-time progress updates via polling.
 */

import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { ActivePipeline } from "@/types";

interface UseActivePipelinesOptions {
  /** Polling interval in milliseconds (default: 2000ms) */
  refetchInterval?: number;
  /** Enable/disable the query (default: true) */
  enabled?: boolean;
}

/**
 * Hook to fetch and poll active pipeline operations.
 * 
 * @example
 * // Basic usage - polls every 2 seconds
 * const { data: pipelines, isLoading } = useActivePipelines();
 * 
 * @example
 * // Custom polling interval
 * const { data: pipelines } = useActivePipelines({ refetchInterval: 5000 });
 * 
 * @example
 * // Conditional polling
 * const { data: pipelines } = useActivePipelines({ enabled: isMonitoringEnabled });
 */
export function useActivePipelines(options?: UseActivePipelinesOptions) {
  return useQuery<ActivePipeline[], Error>({
    queryKey: ["pipelines", "active"],
    queryFn: () => apiClient.getActivePipelines(),
    refetchInterval: options?.refetchInterval ?? 2000,
    staleTime: 1000,
    enabled: options?.enabled ?? true,
    retry: false,
    placeholderData: [],
  });
}
