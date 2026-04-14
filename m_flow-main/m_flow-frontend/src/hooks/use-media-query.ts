"use client";

/**
 * useMediaQuery Hook
 *
 * React hook for responsive design using CSS media queries.
 *
 * Features:
 * - SSR compatible
 * - Breakpoint presets
 * - Custom queries support
 * - Orientation detection
 * - Reduced motion detection
 *
 * @example
 * const isMobile = useMediaQuery("(max-width: 640px)");
 * const { isMobile, isTablet, isDesktop } = useBreakpoints();
 */

import { useState, useEffect, useCallback } from "react";

// ============================================================================
// Types
// ============================================================================

export interface Breakpoints {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isLargeDesktop: boolean;
}

export interface DeviceInfo {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isPortrait: boolean;
  isLandscape: boolean;
  isTouchDevice: boolean;
  prefersReducedMotion: boolean;
  prefersDarkMode: boolean;
}

// ============================================================================
// Breakpoint Values (matches Tailwind defaults)
// ============================================================================

export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  "2xl": 1536,
} as const;

// ============================================================================
// Main Hook
// ============================================================================

/**
 * Hook for matching CSS media queries
 */
export function useMediaQuery(query: string): boolean {
  // Server-side default to false
  const [matches, setMatches] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === "undefined") return;

    const mediaQuery = window.matchMedia(query);

    // Set initial value
    setMatches(mediaQuery.matches);

    // Handle changes
    const handleChange = (event: MediaQueryListEvent) => {
      setMatches(event.matches);
    };

    // Modern browsers
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", handleChange);
      return () => mediaQuery.removeEventListener("change", handleChange);
    }

    // Legacy browsers (Safari < 14)
    mediaQuery.addListener(handleChange);
    return () => mediaQuery.removeListener(handleChange);
  }, [query]);

  return matches;
}

// ============================================================================
// Breakpoint Hooks
// ============================================================================

/**
 * Hook for all breakpoint states
 */
export function useBreakpoints(): Breakpoints {
  const isMobile = useMediaQuery(`(max-width: ${BREAKPOINTS.sm - 1}px)`);
  const isTablet = useMediaQuery(
    `(min-width: ${BREAKPOINTS.sm}px) and (max-width: ${BREAKPOINTS.lg - 1}px)`
  );
  const isDesktop = useMediaQuery(
    `(min-width: ${BREAKPOINTS.lg}px) and (max-width: ${BREAKPOINTS.xl - 1}px)`
  );
  const isLargeDesktop = useMediaQuery(`(min-width: ${BREAKPOINTS.xl}px)`);

  return { isMobile, isTablet, isDesktop, isLargeDesktop };
}

/**
 * Hook for minimum width breakpoint
 */
export function useMinWidth(breakpoint: keyof typeof BREAKPOINTS): boolean {
  return useMediaQuery(`(min-width: ${BREAKPOINTS[breakpoint]}px)`);
}

/**
 * Hook for maximum width breakpoint
 */
export function useMaxWidth(breakpoint: keyof typeof BREAKPOINTS): boolean {
  return useMediaQuery(`(max-width: ${BREAKPOINTS[breakpoint] - 1}px)`);
}

/**
 * Hook for between breakpoints
 */
export function useBetweenWidth(
  minBreakpoint: keyof typeof BREAKPOINTS,
  maxBreakpoint: keyof typeof BREAKPOINTS
): boolean {
  return useMediaQuery(
    `(min-width: ${BREAKPOINTS[minBreakpoint]}px) and (max-width: ${BREAKPOINTS[maxBreakpoint] - 1}px)`
  );
}

// ============================================================================
// Convenience Hooks
// ============================================================================

/**
 * Hook for mobile detection
 */
export function useIsMobile(): boolean {
  return useMediaQuery(`(max-width: ${BREAKPOINTS.md - 1}px)`);
}

/**
 * Hook for tablet detection
 */
export function useIsTablet(): boolean {
  return useMediaQuery(
    `(min-width: ${BREAKPOINTS.md}px) and (max-width: ${BREAKPOINTS.lg - 1}px)`
  );
}

/**
 * Hook for desktop detection
 */
export function useIsDesktop(): boolean {
  return useMediaQuery(`(min-width: ${BREAKPOINTS.lg}px)`);
}

// ============================================================================
// Orientation Hooks
// ============================================================================

/**
 * Hook for portrait orientation
 */
export function useIsPortrait(): boolean {
  return useMediaQuery("(orientation: portrait)");
}

/**
 * Hook for landscape orientation
 */
export function useIsLandscape(): boolean {
  return useMediaQuery("(orientation: landscape)");
}

// ============================================================================
// Preference Hooks
// ============================================================================

/**
 * Hook for reduced motion preference
 */
export function usePrefersReducedMotion(): boolean {
  return useMediaQuery("(prefers-reduced-motion: reduce)");
}

/**
 * Hook for dark mode preference
 */
export function usePrefersDarkMode(): boolean {
  return useMediaQuery("(prefers-color-scheme: dark)");
}

/**
 * Hook for light mode preference
 */
export function usePrefersLightMode(): boolean {
  return useMediaQuery("(prefers-color-scheme: light)");
}

/**
 * Hook for high contrast preference
 */
export function usePrefersHighContrast(): boolean {
  return useMediaQuery("(prefers-contrast: more)");
}

// ============================================================================
// Device Detection
// ============================================================================

/**
 * Hook for touch device detection
 */
export function useIsTouchDevice(): boolean {
  const [isTouchDevice, setIsTouchDevice] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const checkTouch = () => {
      setIsTouchDevice(
        "ontouchstart" in window ||
          navigator.maxTouchPoints > 0 ||
          // @ts-ignore - For older browsers
          navigator.msMaxTouchPoints > 0
      );
    };

    checkTouch();
    window.addEventListener("touchstart", checkTouch, { once: true });

    return () => window.removeEventListener("touchstart", checkTouch);
  }, []);

  return isTouchDevice;
}

/**
 * Hook for comprehensive device info
 */
export function useDeviceInfo(): DeviceInfo {
  const isMobile = useIsMobile();
  const isTablet = useIsTablet();
  const isDesktop = useIsDesktop();
  const isPortrait = useIsPortrait();
  const isLandscape = useIsLandscape();
  const isTouchDevice = useIsTouchDevice();
  const prefersReducedMotion = usePrefersReducedMotion();
  const prefersDarkMode = usePrefersDarkMode();

  return {
    isMobile,
    isTablet,
    isDesktop,
    isPortrait,
    isLandscape,
    isTouchDevice,
    prefersReducedMotion,
    prefersDarkMode,
  };
}

// ============================================================================
// Window Size Hook
// ============================================================================

export interface WindowSize {
  width: number;
  height: number;
}

/**
 * Hook for window dimensions
 */
export function useWindowSize(): WindowSize {
  const [size, setSize] = useState<WindowSize>(() => {
    if (typeof window === "undefined") {
      return { width: 0, height: 0 };
    }
    return {
      width: window.innerWidth,
      height: window.innerHeight,
    };
  });

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    // Debounce resize events
    let timeoutId: ReturnType<typeof setTimeout>;
    const debouncedResize = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(handleResize, 100);
    };

    window.addEventListener("resize", debouncedResize);
    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener("resize", debouncedResize);
    };
  }, []);

  return size;
}

// ============================================================================
// Responsive Value Hook
// ============================================================================

export interface ResponsiveValue<T> {
  base: T;
  sm?: T;
  md?: T;
  lg?: T;
  xl?: T;
  "2xl"?: T;
}

/**
 * Hook for responsive values
 */
export function useResponsiveValue<T>(values: ResponsiveValue<T>): T {
  const { width } = useWindowSize();

  const getValue = useCallback((): T => {
    if (width >= BREAKPOINTS["2xl"] && values["2xl"] !== undefined) {
      return values["2xl"];
    }
    if (width >= BREAKPOINTS.xl && values.xl !== undefined) {
      return values.xl;
    }
    if (width >= BREAKPOINTS.lg && values.lg !== undefined) {
      return values.lg;
    }
    if (width >= BREAKPOINTS.md && values.md !== undefined) {
      return values.md;
    }
    if (width >= BREAKPOINTS.sm && values.sm !== undefined) {
      return values.sm;
    }
    return values.base;
  }, [width, values]);

  return getValue();
}

// ============================================================================
// Display Names
// ============================================================================

useMediaQuery.displayName = "useMediaQuery";
useBreakpoints.displayName = "useBreakpoints";
useMinWidth.displayName = "useMinWidth";
useMaxWidth.displayName = "useMaxWidth";
useBetweenWidth.displayName = "useBetweenWidth";
useIsMobile.displayName = "useIsMobile";
useIsTablet.displayName = "useIsTablet";
useIsDesktop.displayName = "useIsDesktop";
useIsPortrait.displayName = "useIsPortrait";
useIsLandscape.displayName = "useIsLandscape";
usePrefersReducedMotion.displayName = "usePrefersReducedMotion";
usePrefersDarkMode.displayName = "usePrefersDarkMode";
usePrefersLightMode.displayName = "usePrefersLightMode";
useIsTouchDevice.displayName = "useIsTouchDevice";
useDeviceInfo.displayName = "useDeviceInfo";
useWindowSize.displayName = "useWindowSize";
useResponsiveValue.displayName = "useResponsiveValue";

// ============================================================================
// Default Export
// ============================================================================

export default useMediaQuery;
