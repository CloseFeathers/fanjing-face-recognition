"use client";

import React, { useState, useEffect, useCallback } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConfirmProvider } from "@/components/ui/confirm-dialog";
import { apiClient } from "@/lib/api/client";
import { useAuthStore } from "@/lib/store";
import { LoginForm } from "@/components/auth/LoginForm";
import { config, STORAGE_KEYS } from "@/lib/config";

type AuthPhase = "initializing" | "ready";

function AuthInitializer({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, user, setUser, setLoading } = useAuthStore();
  const [phase, setPhase] = useState<AuthPhase>("initializing");
  const [loginError, setLoginError] = useState<string | null>(null);

  // Handle successful login - clear logout flag
  const handleLoginSuccess = useCallback(() => {
    setLoginError(null);
    if (typeof window !== "undefined") {
      sessionStorage.removeItem(STORAGE_KEYS.LOGOUT_FLAG);
    }
  }, []);

  useEffect(() => {
    async function initAuth() {
      // Check if user explicitly logged out - don't auto-login
      const explicitLogout = typeof window !== "undefined" 
        ? sessionStorage.getItem(STORAGE_KEYS.LOGOUT_FLAG) === "true"
        : false;

      // Check if already have token
      const token = typeof window !== "undefined" 
        ? localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN) 
        : null;

      if (token) {
        // Validate existing token
        try {
          const user = await apiClient.getCurrentUser();
          setUser(user);
          setPhase("ready");
          return;
        } catch {
          // Token invalid, clear it
          localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
          apiClient.setToken(null);
        }
      }

      // Skip auto-login if user explicitly logged out
      if (explicitLogout) {
        setLoading(false);
        setPhase("ready");
        return;
      }

      // Try auto-login with default credentials (if enabled)
      if (config.AUTO_LOGIN_ENABLED) {
        try {
          const user = await apiClient.login({
            username: config.DEFAULT_USER_EMAIL,
            password: config.DEFAULT_USER_PASSWORD,
          });
          setUser(user);
        } catch (err) {
          const is401 = err instanceof Error && err.message.includes("401");
          setLoginError(
            is401
              ? null
              : `Auto-login failed. Please ensure the backend is running at ${config.API_BASE_URL} and sign in manually.`
          );
          setLoading(false);
        }
      } else {
        // Auto-login disabled, show login form
        setLoading(false);
      }
      
      setPhase("ready");
    }

    initAuth();
  }, [setUser, setLoading]);

  // Show loading state during initialization
  if (phase === "initializing") {
    return (
      <div className="min-h-screen bg-[var(--bg-base)] flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
          <span className="text-sm text-zinc-500">Initializing...</span>
        </div>
      </div>
    );
  }

  // After initialization, check Zustand auth state
  // This ensures logout triggers login form display
  if (!isAuthenticated || !user) {
    return (
      <LoginForm 
        error={loginError}
        onSuccess={handleLoginSuccess} 
      />
    );
  }

  return <>{children}</>;
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ConfirmProvider>
        <AuthInitializer>{children}</AuthInitializer>
      </ConfirmProvider>
    </QueryClientProvider>
  );
}
