/**
 * Typography Scale
 *
 * Unified typography system for M-Flow.
 * Use these constants to ensure consistent text sizing.
 *
 * Scale (5 sizes):
 * - xs:  11px - Metadata, timestamps, hints
 * - sm:  13px - Body text, descriptions
 * - md:  15px - Emphasized text, buttons
 * - lg:  18px - Section headers
 * - xl:  24px - Page titles
 */

export const typography = {
  // Size classes
  xs: "text-[11px]",
  sm: "text-[13px]",
  md: "text-[15px]",
  lg: "text-[18px]",
  xl: "text-[24px]",

  // Common combinations
  pageTitle: "text-[24px] font-semibold tracking-tight",
  sectionTitle: "text-[18px] font-semibold",
  cardTitle: "text-[15px] font-medium",
  body: "text-[13px]",
  hint: "text-[11px] text-zinc-500",
  label: "text-[12px] font-medium",
  button: "text-[13px] font-medium",
  badge: "text-[10px] font-medium uppercase tracking-wider",

  // Semantic colors
  primary: "text-[var(--text-primary)]",
  secondary: "text-[var(--text-secondary)]",
  muted: "text-[var(--text-muted)]",
  success: "text-emerald-400",
  warning: "text-amber-400",
  error: "text-red-400",
} as const;

/**
 * Heading component classes
 */
export const headings = {
  h1: "text-[24px] font-semibold text-zinc-100 tracking-tight",
  h2: "text-[18px] font-semibold text-zinc-100",
  h3: "text-[15px] font-semibold text-zinc-100",
  h4: "text-[13px] font-semibold text-zinc-200",
} as const;

/**
 * Text component classes
 */
export const text = {
  body: "text-[13px] text-zinc-300",
  muted: "text-[13px] text-zinc-500",
  small: "text-[11px] text-zinc-500",
  code: "font-mono text-[12px] bg-zinc-800 px-1 rounded",
} as const;

export default typography;
