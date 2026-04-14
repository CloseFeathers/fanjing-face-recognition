/**
 * Theme Constants and Design Tokens
 *
 * Centralized theme configuration for the M-Flow frontend.
 * Ensures consistency across all components.
 *
 * Design System:
 * - Pure black background (#000000)
 * - High contrast zinc-based grays
 * - Accent colors for status indicators
 * - Minimal, professional aesthetic
 *
 * @example
 * import { colors, spacing, typography } from "@/lib/utils/theme";
 */

// ============================================================================
// Color Palette
// ============================================================================

export const colors = {
  // Background colors
  background: {
    base: "#000000",
    elevated: "rgb(9, 9, 11)", // zinc-950
    card: "rgba(9, 9, 11, 0.3)",
    hover: "rgba(24, 24, 27, 0.5)",
    overlay: "rgba(0, 0, 0, 0.8)",
  },

  // Border colors
  border: {
    default: "rgb(39, 39, 42)", // zinc-800
    hover: "rgb(63, 63, 70)", // zinc-700
    focus: "rgb(82, 82, 91)", // zinc-600
    accent: "rgb(244, 244, 245)", // zinc-100
  },

  // Text colors
  text: {
    primary: "rgb(244, 244, 245)", // zinc-100
    secondary: "rgb(161, 161, 170)", // zinc-400
    tertiary: "rgb(113, 113, 122)", // zinc-500
    muted: "rgb(82, 82, 91)", // zinc-600
    inverse: "rgb(9, 9, 11)", // zinc-950
  },

  // Status colors
  status: {
    // Success/Up
    success: {
      base: "rgb(52, 211, 153)", // emerald-400
      bg: "rgba(16, 185, 129, 0.1)",
      bgHover: "rgba(16, 185, 129, 0.2)",
      border: "rgba(16, 185, 129, 0.3)",
    },
    // Warning
    warning: {
      base: "rgb(251, 191, 36)", // amber-400
      bg: "rgba(245, 158, 11, 0.1)",
      bgHover: "rgba(245, 158, 11, 0.2)",
      border: "rgba(245, 158, 11, 0.3)",
    },
    // Error/Down
    error: {
      base: "rgb(248, 113, 113)", // red-400
      bg: "rgba(239, 68, 68, 0.1)",
      bgHover: "rgba(239, 68, 68, 0.2)",
      border: "rgba(239, 68, 68, 0.3)",
    },
    // Info
    info: {
      base: "rgb(96, 165, 250)", // blue-400
      bg: "rgba(59, 130, 246, 0.1)",
      bgHover: "rgba(59, 130, 246, 0.2)",
      border: "rgba(59, 130, 246, 0.3)",
    },
  },

  // Interactive colors
  interactive: {
    primary: "rgb(244, 244, 245)", // zinc-100
    primaryHover: "rgb(255, 255, 255)",
    secondary: "rgb(39, 39, 42)", // zinc-800
    secondaryHover: "rgb(63, 63, 70)", // zinc-700
  },
} as const;

// ============================================================================
// Spacing Scale
// ============================================================================

export const spacing = {
  0: "0",
  px: "1px",
  0.5: "0.125rem", // 2px
  1: "0.25rem", // 4px
  1.5: "0.375rem", // 6px
  2: "0.5rem", // 8px
  2.5: "0.625rem", // 10px
  3: "0.75rem", // 12px
  3.5: "0.875rem", // 14px
  4: "1rem", // 16px
  5: "1.25rem", // 20px
  6: "1.5rem", // 24px
  7: "1.75rem", // 28px
  8: "2rem", // 32px
  9: "2.25rem", // 36px
  10: "2.5rem", // 40px
  11: "2.75rem", // 44px
  12: "3rem", // 48px
  14: "3.5rem", // 56px
  16: "4rem", // 64px
  20: "5rem", // 80px
  24: "6rem", // 96px
} as const;

// ============================================================================
// Typography Scale
// ============================================================================

export const typography = {
  // Font families
  fontFamily: {
    sans: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    mono: '"JetBrains Mono", "Fira Code", Consolas, monospace',
  },

  // Font sizes with line heights
  fontSize: {
    xs: ["0.625rem", { lineHeight: "1rem" }], // 10px
    sm: ["0.6875rem", { lineHeight: "1rem" }], // 11px
    base: ["0.75rem", { lineHeight: "1.125rem" }], // 12px
    md: ["0.8125rem", { lineHeight: "1.25rem" }], // 13px
    lg: ["0.875rem", { lineHeight: "1.25rem" }], // 14px
    xl: ["1rem", { lineHeight: "1.5rem" }], // 16px
    "2xl": ["1.125rem", { lineHeight: "1.75rem" }], // 18px
    "3xl": ["1.25rem", { lineHeight: "1.75rem" }], // 20px
    "4xl": ["1.5rem", { lineHeight: "2rem" }], // 24px
    "5xl": ["1.75rem", { lineHeight: "2rem" }], // 28px
  },

  // Font weights
  fontWeight: {
    normal: "400",
    medium: "500",
    semibold: "600",
    bold: "700",
  },

  // Letter spacing
  letterSpacing: {
    tighter: "-0.05em",
    tight: "-0.025em",
    normal: "0",
    wide: "0.025em",
    wider: "0.05em",
    widest: "0.1em",
  },
} as const;

// ============================================================================
// Border Radius
// ============================================================================

export const borderRadius = {
  none: "0",
  sm: "0.125rem", // 2px
  DEFAULT: "0.25rem", // 4px
  md: "0.375rem", // 6px
  lg: "0.5rem", // 8px
  xl: "0.75rem", // 12px
  "2xl": "1rem", // 16px
  "3xl": "1.5rem", // 24px
  full: "9999px",
} as const;

// ============================================================================
// Shadows
// ============================================================================

export const shadows = {
  none: "none",
  sm: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
  DEFAULT: "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
  md: "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
  lg: "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
  xl: "0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)",
  // Glow shadows for status indicators
  glowSuccess: "0 0 10px rgba(16, 185, 129, 0.3)",
  glowWarning: "0 0 10px rgba(245, 158, 11, 0.3)",
  glowError: "0 0 10px rgba(239, 68, 68, 0.3)",
} as const;

// ============================================================================
// Z-Index Scale
// ============================================================================

export const zIndex = {
  hide: -1,
  auto: "auto",
  base: 0,
  docked: 10,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  skipLink: 1600,
  toast: 1700,
  tooltip: 1800,
} as const;

// ============================================================================
// Breakpoints
// ============================================================================

export const breakpoints = {
  sm: "640px",
  md: "768px",
  lg: "1024px",
  xl: "1280px",
  "2xl": "1536px",
} as const;

// ============================================================================
// Component Size Presets
// ============================================================================

export const componentSizes = {
  // Button sizes
  button: {
    xs: { height: "1.5rem", padding: "0 0.5rem", fontSize: "0.625rem" },
    sm: { height: "2rem", padding: "0 0.75rem", fontSize: "0.6875rem" },
    md: { height: "2.5rem", padding: "0 1rem", fontSize: "0.75rem" },
    lg: { height: "3rem", padding: "0 1.5rem", fontSize: "0.875rem" },
    xl: { height: "3.5rem", padding: "0 2rem", fontSize: "1rem" },
  },

  // Input sizes
  input: {
    sm: { height: "2rem", padding: "0 0.5rem", fontSize: "0.6875rem" },
    md: { height: "2.5rem", padding: "0 0.75rem", fontSize: "0.75rem" },
    lg: { height: "3rem", padding: "0 1rem", fontSize: "0.875rem" },
  },

  // Icon sizes
  icon: {
    xs: "0.75rem", // 12px
    sm: "0.875rem", // 14px
    md: "1rem", // 16px
    lg: "1.25rem", // 20px
    xl: "1.5rem", // 24px
  },

  // Avatar sizes
  avatar: {
    xs: "1.5rem", // 24px
    sm: "2rem", // 32px
    md: "2.5rem", // 40px
    lg: "3rem", // 48px
    xl: "4rem", // 64px
  },
} as const;

// ============================================================================
// Tailwind Class Presets
// ============================================================================

/**
 * Common Tailwind class combinations for consistency
 */
export const classPresets = {
  // Card styles
  card: {
    base: "rounded-xl border border-zinc-800 bg-zinc-900/30",
    interactive: "rounded-xl border border-zinc-800 bg-zinc-900/30 hover:border-zinc-700 hover:bg-zinc-900/50 transition-colors cursor-pointer",
    selected: "rounded-xl border border-zinc-100 bg-zinc-900/50 ring-1 ring-zinc-100",
  },

  // Button styles
  button: {
    primary: "bg-zinc-100 text-zinc-900 hover:bg-white",
    secondary: "bg-zinc-800 text-zinc-100 hover:bg-zinc-700",
    ghost: "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800",
    danger: "bg-red-500/20 text-red-300 hover:bg-red-500/30",
    disabled: "bg-zinc-800 text-zinc-500 cursor-not-allowed",
  },

  // Input styles
  input: {
    base: "bg-zinc-900 border border-zinc-800 text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-600",
    error: "bg-zinc-900 border border-red-900/50 text-zinc-200 focus:outline-none focus:ring-1 focus:ring-red-500/50",
  },

  // Text styles
  text: {
    heading: "text-zinc-100 font-semibold tracking-tight",
    body: "text-zinc-400",
    muted: "text-zinc-500",
    link: "text-zinc-400 hover:text-zinc-200 underline underline-offset-2",
  },

  // Badge styles
  badge: {
    default: "px-2 py-0.5 rounded text-[10px] font-medium bg-zinc-800 text-zinc-400",
    success: "px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30",
    warning: "px-2 py-0.5 rounded text-[10px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30",
    error: "px-2 py-0.5 rounded text-[10px] font-medium bg-red-500/10 text-red-400 border border-red-500/30",
    info: "px-2 py-0.5 rounded text-[10px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/30",
  },

  // Section header
  sectionHeader: "pb-4 border-b border-zinc-800",

  // Focus ring
  focusRing: "focus:outline-none focus:ring-2 focus:ring-zinc-500 focus:ring-offset-2 focus:ring-offset-black",
} as const;

// ============================================================================
// CSS Variable Exports
// ============================================================================

/**
 * Generate CSS custom properties for theming
 */
export function generateCSSVariables(): string {
  return `
:root {
  /* Background */
  --color-bg-base: ${colors.background.base};
  --color-bg-elevated: ${colors.background.elevated};
  --color-bg-card: ${colors.background.card};
  --color-bg-hover: ${colors.background.hover};
  --color-bg-overlay: ${colors.background.overlay};

  /* Border */
  --color-border-default: ${colors.border.default};
  --color-border-hover: ${colors.border.hover};
  --color-border-focus: ${colors.border.focus};
  --color-border-accent: ${colors.border.accent};

  /* Text */
  --color-text-primary: ${colors.text.primary};
  --color-text-secondary: ${colors.text.secondary};
  --color-text-tertiary: ${colors.text.tertiary};
  --color-text-muted: ${colors.text.muted};

  /* Status */
  --color-success: ${colors.status.success.base};
  --color-warning: ${colors.status.warning.base};
  --color-error: ${colors.status.error.base};
  --color-info: ${colors.status.info.base};

  /* Typography */
  --font-sans: ${typography.fontFamily.sans};
  --font-mono: ${typography.fontFamily.mono};
}
  `.trim();
}

// ============================================================================
// Type Exports
// ============================================================================

export type ColorKey = keyof typeof colors;
export type SpacingKey = keyof typeof spacing;
export type FontSizeKey = keyof typeof typography.fontSize;
export type BorderRadiusKey = keyof typeof borderRadius;
export type ShadowKey = keyof typeof shadows;
export type ZIndexKey = keyof typeof zIndex;
export type BreakpointKey = keyof typeof breakpoints;
