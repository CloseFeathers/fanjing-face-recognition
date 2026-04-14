/**
 * Accessibility Utilities
 *
 * Provides helpers for building accessible UI components.
 * Implements WCAG 2.1 AA compliance patterns.
 *
 * Features:
 * - ARIA attribute helpers
 * - Focus management
 * - Keyboard navigation
 * - Screen reader announcements
 * - Color contrast utilities
 *
 * @example
 * // Add accessible button props
 * <button {...getButtonProps({ label: "Submit" })} />
 *
 * // Announce to screen readers
 * announce("Form submitted successfully");
 */

// ============================================================================
// ARIA Attribute Helpers
// ============================================================================

/**
 * Generate accessible button props
 */
export interface ButtonA11yProps {
  /** Accessible label (for icon-only buttons) */
  label?: string;
  /** Description for complex actions */
  description?: string;
  /** Whether button is disabled */
  disabled?: boolean;
  /** Whether button is loading */
  loading?: boolean;
  /** Whether button is pressed (toggle buttons) */
  pressed?: boolean;
  /** Whether button is expanded (disclosure buttons) */
  expanded?: boolean;
  /** ID of controlled element (for expanded) */
  controls?: string;
  /** Whether this is a menu button */
  hasPopup?: boolean | "menu" | "listbox" | "dialog";
}

export function getButtonProps({
  label,
  description,
  disabled,
  loading,
  pressed,
  expanded,
  controls,
  hasPopup,
}: ButtonA11yProps): Record<string, unknown> {
  const props: Record<string, unknown> = {
    role: "button",
    tabIndex: disabled ? -1 : 0,
  };

  if (label) props["aria-label"] = label;
  if (description) props["aria-describedby"] = description;
  if (disabled) props["aria-disabled"] = true;
  if (loading) {
    props["aria-disabled"] = true;
    props["aria-busy"] = true;
  }
  if (pressed !== undefined) props["aria-pressed"] = pressed;
  if (expanded !== undefined) props["aria-expanded"] = expanded;
  if (controls) props["aria-controls"] = controls;
  if (hasPopup) props["aria-haspopup"] = hasPopup;

  return props;
}

/**
 * Generate accessible expandable/collapsible props
 */
export interface ExpandableA11yProps {
  /** Whether expanded */
  expanded: boolean;
  /** ID of the controlled content */
  contentId: string;
  /** Button label when collapsed */
  collapsedLabel?: string;
  /** Button label when expanded */
  expandedLabel?: string;
}

export function getExpandableProps({
  expanded,
  contentId,
  collapsedLabel = "Expand",
  expandedLabel = "Collapse",
}: ExpandableA11yProps): {
  trigger: Record<string, unknown>;
  content: Record<string, unknown>;
} {
  return {
    trigger: {
      role: "button",
      tabIndex: 0,
      "aria-expanded": expanded,
      "aria-controls": contentId,
      "aria-label": expanded ? expandedLabel : collapsedLabel,
    },
    content: {
      id: contentId,
      role: "region",
      "aria-hidden": !expanded,
    },
  };
}

/**
 * Generate accessible tab props
 */
export interface TabA11yProps {
  /** Tab ID */
  id: string;
  /** Whether tab is selected */
  selected: boolean;
  /** Tab panel ID */
  panelId: string;
}

export function getTabProps({
  id,
  selected,
  panelId,
}: TabA11yProps): {
  tab: Record<string, unknown>;
  panel: Record<string, unknown>;
} {
  return {
    tab: {
      id,
      role: "tab",
      tabIndex: selected ? 0 : -1,
      "aria-selected": selected,
      "aria-controls": panelId,
    },
    panel: {
      id: panelId,
      role: "tabpanel",
      tabIndex: 0,
      "aria-labelledby": id,
      hidden: !selected,
    },
  };
}

/**
 * Generate accessible dialog/modal props
 */
export interface DialogA11yProps {
  /** Dialog title ID */
  titleId: string;
  /** Dialog description ID (optional) */
  descriptionId?: string;
  /** Whether dialog is modal */
  modal?: boolean;
}

export function getDialogProps({
  titleId,
  descriptionId,
  modal = true,
}: DialogA11yProps): Record<string, unknown> {
  const props: Record<string, unknown> = {
    role: "dialog",
    "aria-modal": modal,
    "aria-labelledby": titleId,
  };

  if (descriptionId) props["aria-describedby"] = descriptionId;

  return props;
}

/**
 * Generate accessible alert props
 */
export type AlertType = "info" | "success" | "warning" | "error";

export function getAlertProps(
  type: AlertType = "info"
): Record<string, unknown> {
  return {
    role: type === "error" || type === "warning" ? "alert" : "status",
    "aria-live": type === "error" ? "assertive" : "polite",
    "aria-atomic": true,
  };
}

// ============================================================================
// Focus Management
// ============================================================================

/**
 * Focusable element selectors
 */
export const FOCUSABLE_SELECTOR =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

/**
 * Get all focusable elements within a container
 */
export function getFocusableElements(
  container: HTMLElement
): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
  ).filter((el) => !el.hasAttribute("disabled") && el.offsetParent !== null);
}

/**
 * Trap focus within a container (for modals)
 */
export function createFocusTrap(container: HTMLElement): {
  activate: () => void;
  deactivate: () => void;
} {
  let previouslyFocused: HTMLElement | null = null;

  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key !== "Tab") return;

    const focusable = getFocusableElements(container);
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return {
    activate: () => {
      previouslyFocused = document.activeElement as HTMLElement;
      container.addEventListener("keydown", handleKeyDown);
      const focusable = getFocusableElements(container);
      if (focusable.length > 0) {
        focusable[0].focus();
      }
    },
    deactivate: () => {
      container.removeEventListener("keydown", handleKeyDown);
      if (previouslyFocused) {
        previouslyFocused.focus();
      }
    },
  };
}

/**
 * Focus the first focusable element in a container
 */
export function focusFirst(container: HTMLElement): void {
  const focusable = getFocusableElements(container);
  if (focusable.length > 0) {
    focusable[0].focus();
  }
}

/**
 * Focus the last focusable element in a container
 */
export function focusLast(container: HTMLElement): void {
  const focusable = getFocusableElements(container);
  if (focusable.length > 0) {
    focusable[focusable.length - 1].focus();
  }
}

// ============================================================================
// Keyboard Navigation
// ============================================================================

/**
 * Handle arrow key navigation in lists
 */
export interface ArrowKeyNavigationOptions {
  /** Orientation of the list */
  orientation?: "horizontal" | "vertical" | "both";
  /** Whether to wrap around at ends */
  wrap?: boolean;
  /** Callback when selection changes */
  onNavigate?: (index: number) => void;
}

export function handleArrowKeyNavigation(
  event: React.KeyboardEvent,
  currentIndex: number,
  itemCount: number,
  options: ArrowKeyNavigationOptions = {}
): number | null {
  const {
    orientation = "vertical",
    wrap = true,
    onNavigate,
  } = options;

  let nextIndex: number | null = null;

  const isVertical = orientation === "vertical" || orientation === "both";
  const isHorizontal = orientation === "horizontal" || orientation === "both";

  switch (event.key) {
    case "ArrowUp":
      if (isVertical) {
        event.preventDefault();
        nextIndex = currentIndex - 1;
        if (nextIndex < 0) nextIndex = wrap ? itemCount - 1 : 0;
      }
      break;
    case "ArrowDown":
      if (isVertical) {
        event.preventDefault();
        nextIndex = currentIndex + 1;
        if (nextIndex >= itemCount) nextIndex = wrap ? 0 : itemCount - 1;
      }
      break;
    case "ArrowLeft":
      if (isHorizontal) {
        event.preventDefault();
        nextIndex = currentIndex - 1;
        if (nextIndex < 0) nextIndex = wrap ? itemCount - 1 : 0;
      }
      break;
    case "ArrowRight":
      if (isHorizontal) {
        event.preventDefault();
        nextIndex = currentIndex + 1;
        if (nextIndex >= itemCount) nextIndex = wrap ? 0 : itemCount - 1;
      }
      break;
    case "Home":
      event.preventDefault();
      nextIndex = 0;
      break;
    case "End":
      event.preventDefault();
      nextIndex = itemCount - 1;
      break;
  }

  if (nextIndex !== null && onNavigate) {
    onNavigate(nextIndex);
  }

  return nextIndex;
}

/**
 * Check if a key event should trigger a click action
 */
export function isActivationKey(event: React.KeyboardEvent): boolean {
  return event.key === "Enter" || event.key === " ";
}

/**
 * Handle activation key for clickable elements
 */
export function handleActivationKey(
  event: React.KeyboardEvent,
  onClick: () => void
): void {
  if (isActivationKey(event)) {
    event.preventDefault();
    onClick();
  }
}

// ============================================================================
// Screen Reader Announcements
// ============================================================================

let announceRegion: HTMLElement | null = null;

/**
 * Initialize the live region for announcements
 */
function ensureAnnounceRegion(): HTMLElement {
  if (announceRegion) return announceRegion;

  announceRegion = document.createElement("div");
  announceRegion.setAttribute("aria-live", "polite");
  announceRegion.setAttribute("aria-atomic", "true");
  announceRegion.setAttribute("role", "status");
  announceRegion.className = "sr-only";
  announceRegion.style.cssText = `
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  `;
  document.body.appendChild(announceRegion);

  return announceRegion;
}

/**
 * Announce a message to screen readers
 */
export function announce(
  message: string,
  priority: "polite" | "assertive" = "polite"
): void {
  if (typeof document === "undefined") return;

  const region = ensureAnnounceRegion();
  region.setAttribute("aria-live", priority);

  // Clear and set message to ensure announcement
  region.textContent = "";
  requestAnimationFrame(() => {
    region.textContent = message;
  });
}

/**
 * Announce a polite message (doesn't interrupt)
 */
export function announcePolite(message: string): void {
  announce(message, "polite");
}

/**
 * Announce an assertive message (interrupts)
 */
export function announceAssertive(message: string): void {
  announce(message, "assertive");
}

// ============================================================================
// Visibility Utilities
// ============================================================================

/**
 * Screen reader only CSS class
 */
export const SR_ONLY_CLASS = "sr-only";

/**
 * Screen reader only CSS styles
 */
export const SR_ONLY_STYLES: React.CSSProperties = {
  position: "absolute",
  width: "1px",
  height: "1px",
  padding: 0,
  margin: "-1px",
  overflow: "hidden",
  clip: "rect(0, 0, 0, 0)",
  whiteSpace: "nowrap",
  border: 0,
};

/**
 * Check if element is visible
 */
export function isElementVisible(element: HTMLElement): boolean {
  return (
    element.offsetParent !== null &&
    window.getComputedStyle(element).visibility !== "hidden"
  );
}

// ============================================================================
// ID Generation
// ============================================================================

let idCounter = 0;

/**
 * Generate a unique ID for accessibility attributes
 */
export function generateId(prefix: string = "mflow"): string {
  return `${prefix}-${++idCounter}`;
}

/**
 * Hook-style ID generator (use with useMemo)
 */
export function useId(prefix: string = "mflow"): string {
  // In a real implementation, this would use React.useId() or a hook
  return generateId(prefix);
}

// ============================================================================
// ARIA Label Helpers
// ============================================================================

/**
 * Generate a descriptive label for status
 */
export function getStatusLabel(
  status: "up" | "down" | "warn" | "unknown"
): string {
  const labels = {
    up: "Operational",
    down: "Not available",
    warn: "Warning",
    unknown: "Status unknown",
  };
  return labels[status];
}

/**
 * Generate a descriptive label for loading state
 */
export function getLoadingLabel(isLoading: boolean): string {
  return isLoading ? "Loading" : "Loaded";
}

/**
 * Generate a descriptive label for progress
 */
export function getProgressLabel(current: number, total: number): string {
  const percentage = Math.round((current / total) * 100);
  return `${percentage}% complete, ${current} of ${total}`;
}

// ============================================================================
// Export Types
// ============================================================================

export type FocusTrap = ReturnType<typeof createFocusTrap>;
