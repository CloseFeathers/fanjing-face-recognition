/**
 * Animation Utilities
 *
 * Provides consistent animation configurations for the M-Flow frontend.
 * Includes both CSS-based and framer-motion compatible variants.
 *
 * Design Philosophy:
 * - Subtle, professional animations
 * - Performance-conscious (prefer CSS transforms)
 * - Consistent timing across the app
 * - Reduced motion support
 *
 * @example
 * // CSS class approach
 * <div className={cn(animations.fadeIn, "delay-100")} />
 *
 * // Style object approach
 * <div style={createFadeInStyle(0.1)} />
 */

// ============================================================================
// Animation Timing Constants
// ============================================================================

export const TIMING = {
  /** Extra fast - micro-interactions */
  instant: 75,
  /** Fast - small state changes */
  fast: 150,
  /** Normal - standard transitions */
  normal: 200,
  /** Slow - larger movements */
  slow: 300,
  /** Very slow - page transitions */
  verySlow: 500,
} as const;

export const EASING = {
  /** Default easing */
  default: "cubic-bezier(0.4, 0, 0.2, 1)",
  /** Enter easing (accelerate in) */
  enter: "cubic-bezier(0, 0, 0.2, 1)",
  /** Exit easing (accelerate out) */
  exit: "cubic-bezier(0.4, 0, 1, 1)",
  /** Spring-like easing */
  spring: "cubic-bezier(0.34, 1.56, 0.64, 1)",
  /** Linear (no easing) */
  linear: "linear",
} as const;

// ============================================================================
// CSS Animation Classes
// ============================================================================

/**
 * Pre-defined CSS animation class names
 * Use with Tailwind's animate-* utilities or custom keyframes
 */
export const CSS_ANIMATIONS = {
  // Fade animations
  fadeIn: "animate-fadeIn",
  fadeOut: "animate-fadeOut",
  fadeInUp: "animate-fadeInUp",
  fadeInDown: "animate-fadeInDown",

  // Scale animations
  scaleIn: "animate-scaleIn",
  scaleOut: "animate-scaleOut",

  // Slide animations
  slideInLeft: "animate-slideInLeft",
  slideInRight: "animate-slideInRight",
  slideInUp: "animate-slideInUp",
  slideInDown: "animate-slideInDown",

  // Special animations
  pulse: "animate-pulse",
  spin: "animate-spin",
  bounce: "animate-bounce",
  ping: "animate-ping",

  // Skeleton shimmer
  shimmer: "animate-shimmer",
} as const;

// ============================================================================
// Tailwind Keyframes Configuration
// ============================================================================

/**
 * Add these to your tailwind.config.js keyframes section:
 *
 * keyframes: {
 *   fadeIn: {
 *     from: { opacity: '0' },
 *     to: { opacity: '1' },
 *   },
 *   fadeInUp: {
 *     from: { opacity: '0', transform: 'translateY(10px)' },
 *     to: { opacity: '1', transform: 'translateY(0)' },
 *   },
 *   fadeInDown: {
 *     from: { opacity: '0', transform: 'translateY(-10px)' },
 *     to: { opacity: '1', transform: 'translateY(0)' },
 *   },
 *   scaleIn: {
 *     from: { opacity: '0', transform: 'scale(0.95)' },
 *     to: { opacity: '1', transform: 'scale(1)' },
 *   },
 *   slideInLeft: {
 *     from: { opacity: '0', transform: 'translateX(-10px)' },
 *     to: { opacity: '1', transform: 'translateX(0)' },
 *   },
 *   slideInRight: {
 *     from: { opacity: '0', transform: 'translateX(10px)' },
 *     to: { opacity: '1', transform: 'translateX(0)' },
 *   },
 *   shimmer: {
 *     '100%': { transform: 'translateX(100%)' },
 *   },
 * },
 * animation: {
 *   fadeIn: 'fadeIn 200ms ease-out',
 *   fadeInUp: 'fadeInUp 300ms ease-out',
 *   fadeInDown: 'fadeInDown 300ms ease-out',
 *   scaleIn: 'scaleIn 200ms ease-out',
 *   slideInLeft: 'slideInLeft 300ms ease-out',
 *   slideInRight: 'slideInRight 300ms ease-out',
 *   shimmer: 'shimmer 2s infinite',
 * },
 */

// ============================================================================
// Inline Style Creators
// ============================================================================

/**
 * Create fade-in animation style
 */
export function createFadeInStyle(delay: number = 0): React.CSSProperties {
  return {
    opacity: 0,
    animation: `fadeIn ${TIMING.normal}ms ${EASING.enter} ${delay * 1000}ms forwards`,
  };
}

/**
 * Create fade-in-up animation style
 */
export function createFadeInUpStyle(delay: number = 0): React.CSSProperties {
  return {
    opacity: 0,
    transform: "translateY(10px)",
    animation: `fadeInUp ${TIMING.slow}ms ${EASING.enter} ${delay * 1000}ms forwards`,
  };
}

/**
 * Create staggered animation styles for lists
 */
export function createStaggeredStyles(
  index: number,
  baseDelay: number = 0,
  staggerDelay: number = 0.05
): React.CSSProperties {
  return createFadeInUpStyle(baseDelay + index * staggerDelay);
}

// ============================================================================
// Transition Presets
// ============================================================================

/**
 * Common transition strings for use in className
 */
export const TRANSITIONS = {
  /** All properties */
  all: `transition-all duration-${TIMING.normal} ${EASING.default}`,
  /** Colors only */
  colors: `transition-colors duration-${TIMING.fast}`,
  /** Opacity only */
  opacity: `transition-opacity duration-${TIMING.fast}`,
  /** Transform only */
  transform: `transition-transform duration-${TIMING.normal}`,
  /** Background only */
  background: `transition-[background] duration-${TIMING.fast}`,
  /** Border only */
  border: `transition-[border-color] duration-${TIMING.fast}`,
  /** Shadow only */
  shadow: `transition-shadow duration-${TIMING.normal}`,
} as const;

/**
 * Tailwind transition class combinations
 */
export const TRANSITION_CLASSES = {
  /** Standard interactive element transition */
  interactive: "transition-colors duration-200",
  /** Card/container hover transition */
  container: "transition-all duration-200",
  /** Button transition */
  button: "transition-all duration-150",
  /** Icon rotation transition */
  iconRotate: "transition-transform duration-200",
  /** Expand/collapse transition */
  expand: "transition-all duration-300",
  /** Fade transition */
  fade: "transition-opacity duration-200",
} as const;

// ============================================================================
// Framer Motion Variants (if using framer-motion)
// ============================================================================

/**
 * Framer Motion variants for common animations
 * Use with <motion.div variants={variants} initial="hidden" animate="visible" />
 */
export const MOTION_VARIANTS = {
  /** Fade in */
  fadeIn: {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.2 } },
    exit: { opacity: 0, transition: { duration: 0.15 } },
  },

  /** Fade in from bottom */
  fadeInUp: {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
    exit: { opacity: 0, y: 10, transition: { duration: 0.2 } },
  },

  /** Scale in */
  scaleIn: {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1, transition: { duration: 0.2 } },
    exit: { opacity: 0, scale: 0.95, transition: { duration: 0.15 } },
  },

  /** Slide in from left */
  slideInLeft: {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.3 } },
    exit: { opacity: 0, x: -20, transition: { duration: 0.2 } },
  },

  /** Slide in from right */
  slideInRight: {
    hidden: { opacity: 0, x: 20 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.3 } },
    exit: { opacity: 0, x: 20, transition: { duration: 0.2 } },
  },

  /** Stagger children */
  staggerContainer: {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
        delayChildren: 0.1,
      },
    },
  },

  /** Stagger item (use with staggerContainer) */
  staggerItem: {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 },
  },

  /** Collapse/expand */
  collapse: {
    hidden: { height: 0, opacity: 0 },
    visible: { height: "auto", opacity: 1, transition: { duration: 0.3 } },
    exit: { height: 0, opacity: 0, transition: { duration: 0.2 } },
  },
} as const;

// ============================================================================
// Reduced Motion Support
// ============================================================================

/**
 * Check if user prefers reduced motion
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Get animation duration respecting reduced motion preference
 */
export function getAnimationDuration(
  normalDuration: number,
  reducedDuration: number = 0
): number {
  return prefersReducedMotion() ? reducedDuration : normalDuration;
}

/**
 * Create motion-safe animation style
 */
export function createMotionSafeStyle(
  style: React.CSSProperties
): React.CSSProperties {
  if (prefersReducedMotion()) {
    return {
      ...style,
      animation: "none",
      transition: "none",
    };
  }
  return style;
}

// ============================================================================
// Animation Delay Helpers
// ============================================================================

/**
 * Generate delay classes for staggered animations
 */
export function getDelayClass(index: number, baseMs: number = 50): string {
  const delay = index * baseMs;
  // Tailwind has predefined delays: delay-75, delay-100, delay-150, delay-200, etc.
  const validDelays = [0, 75, 100, 150, 200, 300, 500, 700, 1000];
  const closestDelay = validDelays.reduce((prev, curr) =>
    Math.abs(curr - delay) < Math.abs(prev - delay) ? curr : prev
  );
  return closestDelay === 0 ? "" : `delay-${closestDelay}`;
}

/**
 * Generate inline delay style
 */
export function getDelayStyle(index: number, baseMs: number = 50): React.CSSProperties {
  return { animationDelay: `${index * baseMs}ms` };
}

// ============================================================================
// Export Types
// ============================================================================

export type TimingKey = keyof typeof TIMING;
export type EasingKey = keyof typeof EASING;
export type CssAnimationKey = keyof typeof CSS_ANIMATIONS;
export type TransitionKey = keyof typeof TRANSITIONS;
export type MotionVariantKey = keyof typeof MOTION_VARIANTS;
