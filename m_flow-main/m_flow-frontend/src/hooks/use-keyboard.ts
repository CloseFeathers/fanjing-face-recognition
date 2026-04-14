"use client";

/**
 * useKeyboard Hook
 *
 * Handle keyboard shortcuts and navigation.
 *
 * Features:
 * - Global keyboard shortcuts
 * - Key combination support
 * - Escape key handling
 * - Focus trap navigation
 *
 * @example
 * // Single key
 * useKeyboard("Escape", () => closeModal());
 *
 * // Key combination
 * useKeyboard("ctrl+k", () => openSearch());
 *
 * // Multiple keys
 * useKeyboard(["ArrowUp", "ArrowDown"], handleArrowKey);
 */

import { useEffect, useCallback, useRef } from "react";

// ============================================================================
// Types
// ============================================================================

export type KeyboardKey =
  | "Escape"
  | "Enter"
  | "Space"
  | "Tab"
  | "ArrowUp"
  | "ArrowDown"
  | "ArrowLeft"
  | "ArrowRight"
  | "Home"
  | "End"
  | "PageUp"
  | "PageDown"
  | "Backspace"
  | "Delete"
  | string;

export interface KeyboardOptions {
  /** Target element (default: document) */
  target?: HTMLElement | null;
  /** Event type */
  event?: "keydown" | "keyup" | "keypress";
  /** Prevent default behavior */
  preventDefault?: boolean;
  /** Stop propagation */
  stopPropagation?: boolean;
  /** Only trigger when input/textarea is NOT focused */
  ignoreInputs?: boolean;
  /** Only trigger when modifier key is held */
  modifier?: "ctrl" | "alt" | "shift" | "meta" | null;
  /** Whether the hook is enabled */
  enabled?: boolean;
}

export type KeyboardHandler = (event: KeyboardEvent, key: string) => void;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Check if currently focused element is an input
 */
function isInputFocused(): boolean {
  const activeElement = document.activeElement;
  if (!activeElement) return false;

  const tagName = activeElement.tagName.toLowerCase();
  return (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    (activeElement as HTMLElement).isContentEditable
  );
}

/**
 * Parse key string to check for modifiers
 */
function parseKeyString(keyString: string): {
  key: string;
  ctrl: boolean;
  alt: boolean;
  shift: boolean;
  meta: boolean;
} {
  const parts = keyString.toLowerCase().split("+");
  const key = parts.pop() || "";

  return {
    key,
    ctrl: parts.includes("ctrl") || parts.includes("control"),
    alt: parts.includes("alt"),
    shift: parts.includes("shift"),
    meta: parts.includes("meta") || parts.includes("cmd") || parts.includes("command"),
  };
}

/**
 * Check if key event matches key string
 */
function matchesKeyString(event: KeyboardEvent, keyString: string): boolean {
  const parsed = parseKeyString(keyString);
  const eventKey = event.key.toLowerCase();

  // Check modifiers
  if (parsed.ctrl !== event.ctrlKey) return false;
  if (parsed.alt !== event.altKey) return false;
  if (parsed.shift !== event.shiftKey) return false;
  if (parsed.meta !== event.metaKey) return false;

  // Check key
  return eventKey === parsed.key || event.code.toLowerCase() === parsed.key;
}

// ============================================================================
// Main Hook
// ============================================================================

/**
 * Hook for handling keyboard events
 */
export function useKeyboard(
  keys: KeyboardKey | KeyboardKey[],
  handler: KeyboardHandler,
  options: KeyboardOptions = {}
): void {
  const {
    target,
    event = "keydown",
    preventDefault = false,
    stopPropagation = false,
    ignoreInputs = true,
    modifier = null,
    enabled = true,
  } = options;

  // Store handler in ref to avoid re-attaching listener
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  // Normalize keys to array
  const keysArray = Array.isArray(keys) ? keys : [keys];

  useEffect(() => {
    if (!enabled) return;

    const handleKeyEvent = (e: KeyboardEvent) => {
      // Skip if focused on input (unless explicitly allowed)
      if (ignoreInputs && isInputFocused()) return;

      // Check modifier requirement
      if (modifier) {
        if (modifier === "ctrl" && !e.ctrlKey) return;
        if (modifier === "alt" && !e.altKey) return;
        if (modifier === "shift" && !e.shiftKey) return;
        if (modifier === "meta" && !e.metaKey) return;
      }

      // Check if key matches
      for (const key of keysArray) {
        if (matchesKeyString(e, key)) {
          if (preventDefault) e.preventDefault();
          if (stopPropagation) e.stopPropagation();
          handlerRef.current(e, key);
          return;
        }
      }
    };

    const targetElement = target || document;
    targetElement.addEventListener(event, handleKeyEvent as EventListener);

    return () => {
      targetElement.removeEventListener(event, handleKeyEvent as EventListener);
    };
  }, [keysArray.join(","), event, preventDefault, stopPropagation, ignoreInputs, modifier, enabled, target]);
}

// ============================================================================
// Specialized Hooks
// ============================================================================

/**
 * Hook for escape key handling
 */
export function useEscapeKey(
  handler: () => void,
  options: Omit<KeyboardOptions, "keys"> = {}
): void {
  useKeyboard("Escape", handler, options);
}

/**
 * Hook for enter key handling
 */
export function useEnterKey(
  handler: () => void,
  options: Omit<KeyboardOptions, "keys"> = {}
): void {
  useKeyboard("Enter", handler, options);
}

/**
 * Hook for arrow key navigation
 */
export interface ArrowNavigationOptions extends KeyboardOptions {
  /** Current selected index */
  currentIndex: number;
  /** Total number of items */
  itemCount: number;
  /** Whether to wrap around */
  wrap?: boolean;
  /** Orientation */
  orientation?: "horizontal" | "vertical" | "both";
}

export function useArrowNavigation(
  onNavigate: (newIndex: number) => void,
  options: ArrowNavigationOptions
): void {
  const {
    currentIndex,
    itemCount,
    wrap = true,
    orientation = "vertical",
    ...keyboardOptions
  } = options;

  const optionsRef = useRef({ currentIndex, itemCount, wrap, orientation });
  optionsRef.current = { currentIndex, itemCount, wrap, orientation };

  const handleArrow = useCallback(
    (event: KeyboardEvent, key: string) => {
      const { currentIndex, itemCount, wrap, orientation } = optionsRef.current;
      let newIndex = currentIndex;

      const isVertical = orientation === "vertical" || orientation === "both";
      const isHorizontal = orientation === "horizontal" || orientation === "both";

      switch (key) {
        case "ArrowUp":
          if (isVertical) {
            newIndex = currentIndex - 1;
            if (newIndex < 0) newIndex = wrap ? itemCount - 1 : 0;
          }
          break;
        case "ArrowDown":
          if (isVertical) {
            newIndex = currentIndex + 1;
            if (newIndex >= itemCount) newIndex = wrap ? 0 : itemCount - 1;
          }
          break;
        case "ArrowLeft":
          if (isHorizontal) {
            newIndex = currentIndex - 1;
            if (newIndex < 0) newIndex = wrap ? itemCount - 1 : 0;
          }
          break;
        case "ArrowRight":
          if (isHorizontal) {
            newIndex = currentIndex + 1;
            if (newIndex >= itemCount) newIndex = wrap ? 0 : itemCount - 1;
          }
          break;
        case "Home":
          newIndex = 0;
          break;
        case "End":
          newIndex = itemCount - 1;
          break;
      }

      if (newIndex !== currentIndex) {
        event.preventDefault();
        onNavigate(newIndex);
      }
    },
    [onNavigate]
  );

  useKeyboard(
    ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End"],
    handleArrow,
    { preventDefault: true, ...keyboardOptions }
  );
}

/**
 * Hook for global keyboard shortcuts
 */
export interface ShortcutConfig {
  key: string;
  handler: () => void;
  description?: string;
}

export function useKeyboardShortcuts(
  shortcuts: ShortcutConfig[],
  options: Omit<KeyboardOptions, "keys"> = {}
): void {
  const shortcutsRef = useRef(shortcuts);
  shortcutsRef.current = shortcuts;

  const handleShortcut = useCallback((event: KeyboardEvent) => {
    for (const shortcut of shortcutsRef.current) {
      if (matchesKeyString(event, shortcut.key)) {
        event.preventDefault();
        shortcut.handler();
        return;
      }
    }
  }, []);

  useEffect(() => {
    const { enabled = true } = options;
    if (!enabled) return;

    document.addEventListener("keydown", handleShortcut);
    return () => document.removeEventListener("keydown", handleShortcut);
  }, [handleShortcut, options]);
}

// ============================================================================
// Display Name
// ============================================================================

useKeyboard.displayName = "useKeyboard";
useEscapeKey.displayName = "useEscapeKey";
useEnterKey.displayName = "useEnterKey";
useArrowNavigation.displayName = "useArrowNavigation";
useKeyboardShortcuts.displayName = "useKeyboardShortcuts";

// ============================================================================
// Default Export
// ============================================================================

export default useKeyboard;
