"use client";

/**
 * Configuration Wizard State Management Hook
 *
 * Manages the state for the Setup page configuration wizard.
 * Handles step navigation, form validation, and persistence.
 *
 * Features:
 * - Step navigation (forward/back/skip)
 * - Form state per step
 * - Validation state tracking
 * - LocalStorage persistence for progress
 * - Integration with Settings API
 *
 * @example
 * // Basic usage
 * const wizard = useConfigWizard();
 *
 * // Navigation
 * wizard.goNext();
 * wizard.goBack();
 * wizard.goToStep(2);
 *
 * // Step completion
 * wizard.markStepComplete('llm');
 *
 * // Reset
 * wizard.reset();
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useDetailedHealth, useSettings, useUpdateSettings } from "./use-api";
import type {
  ConfigStep,
  ConfigStepId,
  StepStatus,
  SetupWizardState,
} from "@/types/setup";
import type { DetailedHealthResponse } from "@/types";
import { STORAGE_KEYS } from "@/lib/config";

/** Default step definitions */
const DEFAULT_STEPS: ConfigStep[] = [
  {
    id: "llm",
    title: "LLM Provider",
    description: "Configure your language model provider and API key",
    isEditable: true,
    status: "current",
    isOptional: false,
  },
  {
    id: "embedding",
    title: "Embedding Service",
    description: "Configure embedding model for vector search",
    isEditable: true,
    status: "pending",
    isOptional: true,
  },
  {
    id: "database",
    title: "Vector Database",
    description: "Configure vector database for semantic search",
    isEditable: true,
    status: "pending",
    isOptional: true,
  },
  {
    id: "storage",
    title: "Storage",
    description: "View file storage configuration (requires .env)",
    isEditable: false,
    status: "pending",
    isOptional: true,
  },
  {
    id: "review",
    title: "Review & Test",
    description: "Verify all services and run connectivity tests",
    isEditable: false,
    status: "pending",
    isOptional: false,
  },
];

// ============================================================================
// Types
// ============================================================================

/** LLM Configuration Form Data */
export interface LLMFormData {
  provider: string;
  model: string;
  apiKey: string;
  endpoint?: string;
}

/** Vector DB Configuration Form Data */
export interface VectorDBFormData {
  provider: string;
  url: string;
  apiKey: string;
}

/** Embedding Configuration Form Data */
export interface EmbeddingFormData {
  provider: string;
  model: string;
  dimensions?: number;
  endpoint?: string;
  apiKey?: string;
}

/** Persisted wizard state */
interface PersistedState {
  currentStepIndex: number;
  completedSteps: ConfigStepId[];
  llmFormData?: Partial<LLMFormData>;
  completedAt?: string;
}

/** Wizard hook return type */
export interface UseConfigWizardReturn {
  // State
  steps: ConfigStep[];
  currentStep: ConfigStep;
  currentStepIndex: number;
  totalSteps: number;
  isFirstStep: boolean;
  isLastStep: boolean;
  isComplete: boolean;
  isSaving: boolean;
  isTesting: boolean;
  lastError: string | null;

  // Health data (from API)
  healthData: DetailedHealthResponse | undefined;
  isLoadingHealth: boolean;

  // Settings data (from API)
  settings: ReturnType<typeof useSettings>["data"];
  isLoadingSettings: boolean;

  // Navigation
  goNext: () => void;
  goBack: () => void;
  goToStep: (index: number) => void;
  skipStep: () => void;

  // Step management
  markStepComplete: (stepId: ConfigStepId) => void;
  markStepError: (stepId: ConfigStepId, error: string) => void;
  setStepStatus: (stepId: ConfigStepId, status: StepStatus) => void;

  // LLM Form
  llmFormData: Partial<LLMFormData>;
  setLLMFormData: (data: Partial<LLMFormData>) => void;
  saveLLMConfig: () => Promise<boolean>;

  // VectorDB Form
  vectorDBFormData: Partial<VectorDBFormData>;
  setVectorDBFormData: (data: Partial<VectorDBFormData>) => void;
  saveVectorDBConfig: () => Promise<boolean>;

  // Embedding Form
  embeddingFormData: Partial<EmbeddingFormData>;
  setEmbeddingFormData: (data: Partial<EmbeddingFormData>) => void;
  saveEmbeddingConfig: () => Promise<boolean>;

  // Actions
  runTests: () => Promise<void>;
  reset: () => void;
  clearError: () => void;
}

// ============================================================================
// Helper Functions
// ============================================================================

/** Load persisted state from localStorage */
function loadPersistedState(): PersistedState | null {
  if (typeof window === "undefined") return null;

  try {
    const saved = localStorage.getItem(STORAGE_KEYS.WIZARD_PROGRESS);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (e) {
    console.warn("Failed to load wizard state:", e);
  }
  return null;
}

/** Save state to localStorage */
function savePersistedState(state: PersistedState): void {
  if (typeof window === "undefined") return;

  try {
    localStorage.setItem(STORAGE_KEYS.WIZARD_PROGRESS, JSON.stringify(state));
  } catch (e) {
    console.warn("Failed to save wizard state:", e);
  }
}

/** Clear persisted state */
function clearPersistedState(): void {
  if (typeof window === "undefined") return;

  try {
    localStorage.removeItem(STORAGE_KEYS.WIZARD_PROGRESS);
  } catch (e) {
    console.warn("Failed to clear wizard state:", e);
  }
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useConfigWizard(): UseConfigWizardReturn {
  // -------------------------------------------------------------------------
  // API Hooks
  // -------------------------------------------------------------------------
  const {
    data: healthData,
    isLoading: isLoadingHealth,
    refetch: refetchHealth,
  } = useDetailedHealth({ refetchInterval: false });

  const {
    data: settings,
    isLoading: isLoadingSettings,
  } = useSettings();

  const updateSettings = useUpdateSettings();

  // -------------------------------------------------------------------------
  // Local State
  // -------------------------------------------------------------------------
  const [steps, setSteps] = useState<ConfigStep[]>(DEFAULT_STEPS);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<ConfigStepId>>(
    new Set()
  );
  const [llmFormData, setLLMFormData] = useState<Partial<LLMFormData>>({});
  const [vectorDBFormData, setVectorDBFormData] = useState<Partial<VectorDBFormData>>({});
  const [embeddingFormData, setEmbeddingFormData] = useState<Partial<EmbeddingFormData>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const [isComplete, setIsComplete] = useState(false);

  // -------------------------------------------------------------------------
  // Derived State
  // -------------------------------------------------------------------------
  const currentStep = steps[currentStepIndex];
  const totalSteps = steps.length;
  const isFirstStep = currentStepIndex === 0;
  const isLastStep = currentStepIndex === totalSteps - 1;

  // -------------------------------------------------------------------------
  // Persistence: Load on mount
  // -------------------------------------------------------------------------
  useEffect(() => {
    const persisted = loadPersistedState();
    if (persisted) {
      setCurrentStepIndex(persisted.currentStepIndex);
      setCompletedSteps(new Set(persisted.completedSteps));
      if (persisted.llmFormData) {
        setLLMFormData(persisted.llmFormData);
      }
      if (persisted.completedAt) {
        setIsComplete(true);
      }

      // Update step statuses based on completed steps
      setSteps((prev) =>
        prev.map((step, idx) => ({
          ...step,
          status: persisted.completedSteps.includes(step.id)
            ? "completed"
            : idx === persisted.currentStepIndex
            ? "current"
            : idx < persisted.currentStepIndex
            ? "skipped"
            : "pending",
        }))
      );
    }
  }, []);

  // -------------------------------------------------------------------------
  // Persistence: Save on state change
  // -------------------------------------------------------------------------
  useEffect(() => {
    const state: PersistedState = {
      currentStepIndex,
      completedSteps: Array.from(completedSteps),
      // Security: Filter out apiKey to prevent storing sensitive data in localStorage
      // API Key will need to be re-entered after page refresh, but backend will not
      // overwrite existing key if empty value is sent (verified in save_llm_config.py)
      llmFormData: {
        ...llmFormData,
        apiKey: undefined,
      },
      completedAt: isComplete ? new Date().toISOString() : undefined,
    };
    savePersistedState(state);
  }, [currentStepIndex, completedSteps, llmFormData, isComplete]);

  // -------------------------------------------------------------------------
  // Initialize LLM form from settings
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (settings?.llm && !llmFormData.provider) {
      setLLMFormData({
        provider: settings.llm.llm_provider || "",
        model: settings.llm.llm_model || "",
        endpoint: settings.llm.llm_endpoint || "",
        // API key is not returned from server for security
      });
    }
  }, [settings, llmFormData.provider]);

  // -------------------------------------------------------------------------
  // Initialize VectorDB form from settings
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (settings?.vector_db && !vectorDBFormData.provider) {
      setVectorDBFormData({
        provider: settings.vector_db.provider || "lancedb",
        url: settings.vector_db.url || "",
        apiKey: "",
      });
    }
  }, [settings, vectorDBFormData.provider]);

  // -------------------------------------------------------------------------
  // Initialize Embedding form from health probe data
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (healthData?.probes?.embedding_service && !embeddingFormData.provider) {
      const probe = healthData.probes.embedding_service;
      setEmbeddingFormData({
        provider: probe.backend !== "unknown" ? probe.backend : "openai",
        model: "text-embedding-3-small",
        dimensions: 1536,
        endpoint: "",
        apiKey: "",
      });
    }
  }, [healthData, embeddingFormData.provider]);

  // -------------------------------------------------------------------------
  // Step Status Management
  // -------------------------------------------------------------------------
  const updateStepStatus = useCallback(
    (stepId: ConfigStepId, status: StepStatus) => {
      setSteps((prev) =>
        prev.map((step) =>
          step.id === stepId ? { ...step, status } : step
        )
      );
    },
    []
  );

  const setStepStatus = useCallback(
    (stepId: ConfigStepId, status: StepStatus) => {
      updateStepStatus(stepId, status);
    },
    [updateStepStatus]
  );

  const markStepComplete = useCallback(
    (stepId: ConfigStepId) => {
      setCompletedSteps((prev) => new Set([...prev, stepId]));
      updateStepStatus(stepId, "completed");
    },
    [updateStepStatus]
  );

  const markStepError = useCallback(
    (stepId: ConfigStepId, error: string) => {
      updateStepStatus(stepId, "error");
      setLastError(error);
    },
    [updateStepStatus]
  );

  // -------------------------------------------------------------------------
  // Navigation
  // -------------------------------------------------------------------------
  const updateCurrentStep = useCallback(
    (newIndex: number) => {
      // Update previous step status
      setSteps((prev) =>
        prev.map((step, idx) => {
          if (idx === currentStepIndex && step.status === "current") {
            // Mark as completed if moving forward, skipped otherwise
            return {
              ...step,
              status:
                newIndex > currentStepIndex && completedSteps.has(step.id)
                  ? "completed"
                  : newIndex > currentStepIndex
                  ? "skipped"
                  : step.status,
            };
          }
          if (idx === newIndex) {
            return { ...step, status: "current" };
          }
          return step;
        })
      );
      setCurrentStepIndex(newIndex);
    },
    [currentStepIndex, completedSteps]
  );

  const goNext = useCallback(() => {
    if (!isLastStep) {
      updateCurrentStep(currentStepIndex + 1);
    } else {
      // Wizard complete
      setIsComplete(true);
    }
  }, [isLastStep, currentStepIndex, updateCurrentStep]);

  const goBack = useCallback(() => {
    if (!isFirstStep) {
      updateCurrentStep(currentStepIndex - 1);
    }
  }, [isFirstStep, currentStepIndex, updateCurrentStep]);

  const goToStep = useCallback(
    (index: number) => {
      if (index >= 0 && index < totalSteps) {
        updateCurrentStep(index);
      }
    },
    [totalSteps, updateCurrentStep]
  );

  const skipStep = useCallback(() => {
    if (currentStep.isOptional) {
      updateStepStatus(currentStep.id, "skipped");
      goNext();
    }
  }, [currentStep, updateStepStatus, goNext]);

  // -------------------------------------------------------------------------
  // LLM Config Save
  // -------------------------------------------------------------------------
  const saveLLMConfig = useCallback(async (): Promise<boolean> => {
    if (!llmFormData.provider || !llmFormData.model) {
      setLastError("Provider and model are required");
      return false;
    }

    setIsSaving(true);
    setLastError(null);

    try {
      // Backend API uses short field names: provider, model, api_key
      // Not llm_provider, llm_model, llm_api_key
      await updateSettings.mutateAsync({
        llm: {
          provider: llmFormData.provider,
          model: llmFormData.model,
          api_key: llmFormData.apiKey || "",
        },
      });

      markStepComplete("llm");
      return true;
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to save LLM configuration";
      setLastError(message);
      markStepError("llm", message);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [llmFormData, updateSettings, markStepComplete, markStepError]);

  // -------------------------------------------------------------------------
  // VectorDB Config Save
  // -------------------------------------------------------------------------
  const saveVectorDBConfig = useCallback(async (): Promise<boolean> => {
    if (!vectorDBFormData.provider) {
      setLastError("Vector DB provider is required");
      return false;
    }

    setIsSaving(true);
    setLastError(null);

    try {
      await updateSettings.mutateAsync({
        vector_db: {
          provider: vectorDBFormData.provider,
          url: vectorDBFormData.url || "",
          api_key: vectorDBFormData.apiKey || "",
        },
      });

      markStepComplete("database");
      return true;
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to save Vector DB configuration";
      setLastError(message);
      markStepError("database", message);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [vectorDBFormData, updateSettings, markStepComplete, markStepError]);

  // -------------------------------------------------------------------------
  // Embedding Config Save
  // -------------------------------------------------------------------------
  const saveEmbeddingConfig = useCallback(async (): Promise<boolean> => {
    if (!embeddingFormData.provider || !embeddingFormData.model) {
      setLastError("Embedding provider and model are required");
      return false;
    }

    setIsSaving(true);
    setLastError(null);

    try {
      await updateSettings.mutateAsync({
        embedding: {
          provider: embeddingFormData.provider,
          model: embeddingFormData.model,
          dimensions: embeddingFormData.dimensions,
          endpoint: embeddingFormData.endpoint || "",
          api_key: embeddingFormData.apiKey || "",
        },
      });

      markStepComplete("embedding");
      return true;
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to save Embedding configuration";
      setLastError(message);
      markStepError("embedding", message);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [embeddingFormData, updateSettings, markStepComplete, markStepError]);

  // -------------------------------------------------------------------------
  // Test Runner
  // -------------------------------------------------------------------------
  const runTests = useCallback(async () => {
    setIsTesting(true);
    setLastError(null);

    try {
      await refetchHealth();
      // Mark review step as complete if health is good
      if (healthData?.verdict === "up") {
        markStepComplete("review");
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Test execution failed";
      setLastError(message);
    } finally {
      setIsTesting(false);
    }
  }, [refetchHealth, healthData, markStepComplete]);

  // -------------------------------------------------------------------------
  // Reset
  // -------------------------------------------------------------------------
  const reset = useCallback(() => {
    setSteps(DEFAULT_STEPS);
    setCurrentStepIndex(0);
    setCompletedSteps(new Set());
    setLLMFormData({});
    setVectorDBFormData({});
    setEmbeddingFormData({});
    setIsSaving(false);
    setIsTesting(false);
    setLastError(null);
    setIsComplete(false);
    clearPersistedState();
  }, []);

  const clearError = useCallback(() => {
    setLastError(null);
  }, []);

  // -------------------------------------------------------------------------
  // Return
  // -------------------------------------------------------------------------
  return useMemo(
    () => ({
      // State
      steps,
      currentStep,
      currentStepIndex,
      totalSteps,
      isFirstStep,
      isLastStep,
      isComplete,
      isSaving,
      isTesting,
      lastError,

      // Health data
      healthData,
      isLoadingHealth,

      // Settings data
      settings,
      isLoadingSettings,

      // Navigation
      goNext,
      goBack,
      goToStep,
      skipStep,

      // Step management
      markStepComplete,
      markStepError,
      setStepStatus,

      // LLM Form
      llmFormData,
      setLLMFormData,
      saveLLMConfig,

      // VectorDB Form
      vectorDBFormData,
      setVectorDBFormData,
      saveVectorDBConfig,

      // Embedding Form
      embeddingFormData,
      setEmbeddingFormData,
      saveEmbeddingConfig,

      // Actions
      runTests,
      reset,
      clearError,
    }),
    [
      steps,
      currentStep,
      currentStepIndex,
      totalSteps,
      isFirstStep,
      isLastStep,
      isComplete,
      isSaving,
      isTesting,
      lastError,
      healthData,
      isLoadingHealth,
      settings,
      isLoadingSettings,
      goNext,
      goBack,
      goToStep,
      skipStep,
      markStepComplete,
      markStepError,
      setStepStatus,
      llmFormData,
      saveLLMConfig,
      vectorDBFormData,
      saveVectorDBConfig,
      embeddingFormData,
      saveEmbeddingConfig,
      runTests,
      reset,
      clearError,
    ]
  );
}

// ============================================================================
// Display Name for DevTools
// ============================================================================

useConfigWizard.displayName = "useConfigWizard";

// ============================================================================
// Default Export
// ============================================================================

export default useConfigWizard;
