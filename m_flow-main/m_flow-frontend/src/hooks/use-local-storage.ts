"use client";

/**
 * useLocalStorage Hook
 *
 * Persist state in localStorage with type safety and SSR support.
 *
 * Features:
 * - Type-safe storage
 * - SSR compatible
 * - Cross-tab synchronization
 * - Default values
 * - JSON serialization
 *
 * @example
 * const [theme, setTheme] = useLocalStorage("theme", "dark");
 * const [settings, setSettings] = useLocalStorage("settings", { notifications: true });
 */

import { useState, useEffect, useCallback, useRef } from "react";

// ============================================================================
// Types
// ============================================================================

export type SetValue<T> = T | ((prevValue: T) => T);

export interface UseLocalStorageOptions<T> {
  /** Serializer function */
  serializer?: (value: T) => string;
  /** Deserializer function */
  deserializer?: (value: string) => T;
  /** Whether to sync across tabs */
  syncTabs?: boolean;
  /** Error handler */
  onError?: (error: Error) => void;
}

// ============================================================================
// Default Serializers
// ============================================================================

const defaultSerializer = <T>(value: T): string => JSON.stringify(value);

const defaultDeserializer = <T>(value: string): T => JSON.parse(value);

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Check if we're in a browser environment
 */
function isBrowser(): boolean {
  return typeof window !== "undefined";
}

/**
 * Safely get from localStorage
 */
function getStorageValue<T>(
  key: string,
  defaultValue: T,
  deserializer: (value: string) => T
): T {
  if (!isBrowser()) return defaultValue;

  try {
    const item = localStorage.getItem(key);
    if (item === null) return defaultValue;
    return deserializer(item);
  } catch (error) {
    console.warn(`Error reading localStorage key "${key}":`, error);
    return defaultValue;
  }
}

/**
 * Safely set to localStorage
 */
function setStorageValue<T>(
  key: string,
  value: T,
  serializer: (value: T) => string,
  onError?: (error: Error) => void
): boolean {
  if (!isBrowser()) return false;

  try {
    localStorage.setItem(key, serializer(value));
    return true;
  } catch (error) {
    const e = error instanceof Error ? error : new Error(String(error));
    console.warn(`Error setting localStorage key "${key}":`, e);
    onError?.(e);
    return false;
  }
}

/**
 * Remove from localStorage
 */
function removeStorageValue(key: string): void {
  if (!isBrowser()) return;

  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.warn(`Error removing localStorage key "${key}":`, error);
  }
}

// ============================================================================
// Main Hook
// ============================================================================

/**
 * Hook for persisting state in localStorage
 */
export function useLocalStorage<T>(
  key: string,
  defaultValue: T,
  options: UseLocalStorageOptions<T> = {}
): [T, (value: SetValue<T>) => void, () => void] {
  const {
    serializer = defaultSerializer,
    deserializer = defaultDeserializer,
    syncTabs = true,
    onError,
  } = options;

  // Use refs to avoid re-renders when options change
  const serializerRef = useRef(serializer);
  serializerRef.current = serializer;
  const deserializerRef = useRef(deserializer);
  deserializerRef.current = deserializer;

  // Initialize state from localStorage
  const [storedValue, setStoredValue] = useState<T>(() => {
    return getStorageValue(key, defaultValue, deserializerRef.current);
  });

  // Update localStorage when state changes
  const setValue = useCallback(
    (value: SetValue<T>) => {
      setStoredValue((prevValue) => {
        const newValue =
          value instanceof Function ? value(prevValue) : value;
        setStorageValue(key, newValue, serializerRef.current, onError);
        return newValue;
      });
    },
    [key, onError]
  );

  // Remove from localStorage
  const removeValue = useCallback(() => {
    removeStorageValue(key);
    setStoredValue(defaultValue);
  }, [key, defaultValue]);

  // Sync across tabs
  useEffect(() => {
    if (!syncTabs || !isBrowser()) return;

    const handleStorageChange = (event: StorageEvent) => {
      if (event.key !== key) return;

      if (event.newValue === null) {
        setStoredValue(defaultValue);
      } else {
        try {
          setStoredValue(deserializerRef.current(event.newValue));
        } catch (error) {
          console.warn(`Error deserializing localStorage key "${key}":`, error);
        }
      }
    };

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, [key, defaultValue, syncTabs]);

  // Update state if key changes
  useEffect(() => {
    setStoredValue(getStorageValue(key, defaultValue, deserializerRef.current));
  }, [key, defaultValue]);

  return [storedValue, setValue, removeValue];
}

// ============================================================================
// Specialized Hooks
// ============================================================================

/**
 * Hook for boolean localStorage values
 */
export function useLocalStorageBoolean(
  key: string,
  defaultValue: boolean = false
): [boolean, (value: SetValue<boolean>) => void, () => void] {
  return useLocalStorage<boolean>(key, defaultValue, {
    serializer: (v) => String(v),
    deserializer: (v) => v === "true",
  });
}

/**
 * Hook for number localStorage values
 */
export function useLocalStorageNumber(
  key: string,
  defaultValue: number = 0
): [number, (value: SetValue<number>) => void, () => void] {
  return useLocalStorage<number>(key, defaultValue, {
    serializer: (v) => String(v),
    deserializer: (v) => {
      const n = Number(v);
      return isNaN(n) ? defaultValue : n;
    },
  });
}

/**
 * Hook for string localStorage values (no JSON overhead)
 */
export function useLocalStorageString(
  key: string,
  defaultValue: string = ""
): [string, (value: SetValue<string>) => void, () => void] {
  return useLocalStorage<string>(key, defaultValue, {
    serializer: (v) => v,
    deserializer: (v) => v,
  });
}

/**
 * Hook for array localStorage values
 */
export function useLocalStorageArray<T>(
  key: string,
  defaultValue: T[] = []
): [T[], (value: SetValue<T[]>) => void, () => void, (item: T) => void, (index: number) => void] {
  const [value, setValue, removeValue] = useLocalStorage<T[]>(key, defaultValue);

  const addItem = useCallback(
    (item: T) => {
      setValue((prev) => [...prev, item]);
    },
    [setValue]
  );

  const removeItem = useCallback(
    (index: number) => {
      setValue((prev) => prev.filter((_, i) => i !== index));
    },
    [setValue]
  );

  return [value, setValue, removeValue, addItem, removeItem];
}

/**
 * Hook for Set-like localStorage
 */
export function useLocalStorageSet<T>(
  key: string,
  defaultValue: T[] = []
): [T[], (item: T) => void, (item: T) => void, () => void] {
  const [value, setValue, clearValue] = useLocalStorage<T[]>(key, defaultValue);

  const add = useCallback(
    (item: T) => {
      setValue((prev) => {
        if (prev.includes(item)) return prev;
        return [...prev, item];
      });
    },
    [setValue]
  );

  const remove = useCallback(
    (item: T) => {
      setValue((prev) => prev.filter((i) => i !== item));
    },
    [setValue]
  );

  return [value, add, remove, clearValue];
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get all localStorage keys with a prefix
 */
export function getStorageKeys(prefix: string = ""): string[] {
  if (!isBrowser()) return [];

  const keys: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith(prefix)) {
      keys.push(key);
    }
  }
  return keys;
}

/**
 * Clear localStorage with a prefix
 */
export function clearStorageWithPrefix(prefix: string): void {
  if (!isBrowser()) return;

  const keys = getStorageKeys(prefix);
  keys.forEach((key) => localStorage.removeItem(key));
}

/**
 * Get localStorage size usage
 */
export function getStorageSize(): { used: number; remaining: number } {
  if (!isBrowser()) return { used: 0, remaining: 0 };

  let used = 0;
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key) {
      used += (key.length + (localStorage.getItem(key)?.length || 0)) * 2; // UTF-16
    }
  }

  // Typical localStorage limit is 5MB
  const limit = 5 * 1024 * 1024;
  return { used, remaining: limit - used };
}

// ============================================================================
// Display Names
// ============================================================================

useLocalStorage.displayName = "useLocalStorage";
useLocalStorageBoolean.displayName = "useLocalStorageBoolean";
useLocalStorageNumber.displayName = "useLocalStorageNumber";
useLocalStorageString.displayName = "useLocalStorageString";
useLocalStorageArray.displayName = "useLocalStorageArray";
useLocalStorageSet.displayName = "useLocalStorageSet";

// ============================================================================
// Default Export
// ============================================================================

export default useLocalStorage;
