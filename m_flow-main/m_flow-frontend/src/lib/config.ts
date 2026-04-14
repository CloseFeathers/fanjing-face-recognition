/**
 * Frontend Configuration
 *
 * Centralized configuration for the M-Flow frontend application.
 * All environment variables and default values are managed here.
 *
 * Environment Variables:
 * - NEXT_PUBLIC_API_URL: Backend API URL (default: http://localhost:8000)
 * - NEXT_PUBLIC_WS_URL: WebSocket URL (default: derived from API_URL)
 * - NEXT_PUBLIC_AUTO_LOGIN: Enable auto-login with default credentials (default: true)
 * - NEXT_PUBLIC_DEFAULT_USER_EMAIL: Default user email
 * - NEXT_PUBLIC_DEFAULT_USER_PASSWORD: Default user password
 */

// ============================================================================
// API Configuration
// ============================================================================

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_URL || API_BASE_URL.replace(/^http/, "ws");

// ============================================================================
// Authentication Configuration
// ============================================================================

/**
 * Whether auto-login with default credentials is enabled.
 * Set NEXT_PUBLIC_AUTO_LOGIN=false in production to disable.
 */
const AUTO_LOGIN_ENABLED = process.env.NEXT_PUBLIC_AUTO_LOGIN !== "false";

/**
 * Default user credentials.
 * Should match backend base_config.py settings.
 */
const DEFAULT_USER_EMAIL =
  process.env.NEXT_PUBLIC_DEFAULT_USER_EMAIL || "default_user@example.com";
const DEFAULT_USER_PASSWORD =
  process.env.NEXT_PUBLIC_DEFAULT_USER_PASSWORD || "default_password";

// ============================================================================
// Storage Keys
// ============================================================================

/**
 * LocalStorage and SessionStorage keys used by the application.
 * Centralized to avoid key conflicts and enable easy management.
 */
export const STORAGE_KEYS = {
  AUTH_TOKEN: "mflow_token",
  LOGOUT_FLAG: "mflow_explicit_logout",
  WIZARD_PROGRESS: "mflow-setup-wizard-progress",
  COREF_SESSION_PREFIX: "coref_session_",
} as const;

// ============================================================================
// Exported Configuration
// ============================================================================

export const config = {
  // API URLs
  API_BASE_URL,
  WS_BASE_URL,

  // Authentication
  AUTO_LOGIN_ENABLED,
  DEFAULT_USER_EMAIL,
  DEFAULT_USER_PASSWORD,
} as const;

// Type exports for TypeScript usage
export type Config = typeof config;
export type StorageKeys = typeof STORAGE_KEYS;
