/**
 * Utilities Barrel Export
 *
 * All utility functions and constants for the M-Flow frontend.
 *
 * Modules:
 * - animations: Animation timing and variants
 * - accessibility: ARIA helpers and focus management
 * - theme: Design tokens and class presets
 * - health: Health check utilities
 */

// Animations
export {
  TIMING,
  EASING,
  CSS_ANIMATIONS,
  TRANSITIONS,
  TRANSITION_CLASSES,
  MOTION_VARIANTS,
  createFadeInStyle,
  createFadeInUpStyle,
  createStaggeredStyles,
  prefersReducedMotion,
  getAnimationDuration,
  createMotionSafeStyle,
  getDelayClass,
  getDelayStyle,
} from "./animations";

export type {
  TimingKey,
  EasingKey,
  CssAnimationKey,
  TransitionKey,
  MotionVariantKey,
} from "./animations";

// Accessibility
export {
  getButtonProps,
  getExpandableProps,
  getTabProps,
  getDialogProps,
  getAlertProps,
  FOCUSABLE_SELECTOR,
  getFocusableElements,
  createFocusTrap,
  focusFirst,
  focusLast,
  handleArrowKeyNavigation,
  isActivationKey,
  handleActivationKey,
  announce,
  announcePolite,
  announceAssertive,
  SR_ONLY_CLASS,
  SR_ONLY_STYLES,
  isElementVisible,
  generateId,
  useId,
  getStatusLabel,
  getLoadingLabel,
  getProgressLabel,
} from "./accessibility";

export type {
  ButtonA11yProps,
  ExpandableA11yProps,
  TabA11yProps,
  DialogA11yProps,
  AlertType,
  ArrowKeyNavigationOptions,
  FocusTrap,
} from "./accessibility";

// Theme
export {
  colors,
  spacing,
  typography,
  borderRadius,
  shadows,
  zIndex,
  breakpoints,
  componentSizes,
  classPresets,
  generateCSSVariables,
} from "./theme";

export type {
  ColorKey,
  SpacingKey,
  FontSizeKey,
  BorderRadiusKey,
  ShadowKey,
  ZIndexKey,
  BreakpointKey,
} from "./theme";

// Health utilities (re-export from existing)
export * from "./health";
