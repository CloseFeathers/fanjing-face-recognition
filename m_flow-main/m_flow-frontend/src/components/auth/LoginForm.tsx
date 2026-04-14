"use client";

import React, { useState } from "react";
import { apiClient } from "@/lib/api/client";
import { useAuthStore } from "@/lib/store";
import { config } from "@/lib/config";

interface LoginFormProps {
  onSuccess?: () => void;
  error?: string | null;
}

export function LoginForm({ onSuccess, error: initialError }: LoginFormProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(initialError || null);
  const { setUser } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const user = await apiClient.login({
        username: email,
        password,
      });
      setUser(user);
      onSuccess?.();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Login failed";
      const isNetwork = msg.includes("fetch") || msg.includes("network") || msg.includes("ECONNREFUSED") || msg === "Login failed";
      setError(isNetwork
        ? `Login failed. Please ensure the backend is running at ${config.API_BASE_URL}.`
        : msg
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDefaultLogin = async () => {
    setLoading(true);
    setError(null);

    try {
      const user = await apiClient.login({
        username: config.DEFAULT_USER_EMAIL,
        password: config.DEFAULT_USER_PASSWORD,
      });
      setUser(user);
      onSuccess?.();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "";
      const isNetwork = msg.includes("fetch") || msg.includes("network") || msg.includes("ECONNREFUSED");
      setError(isNetwork
        ? `Connection failed. Please ensure the backend is running at ${config.API_BASE_URL}.`
        : "Default account login failed. The password may have been changed."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--bg-base)] flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <div className="bg-[var(--bg-elevated)] border border-[var(--border-default)] rounded-lg p-6 shadow-xl">
          {/* Header */}
          <div className="text-center mb-6">
            <h1 
              className="text-xl font-normal text-[var(--text-primary)] tracking-[0.15em]"
              style={{ fontFamily: "'Cinzel', serif" }}
            >
              M-Flow
            </h1>
            <p className="text-sm text-[var(--text-secondary)] mt-1">
              Sign in to continue
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-md">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-[var(--text-secondary)] mb-1.5">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@example.com"
                className="w-full px-3 py-2 bg-[var(--bg-subtle)] border border-[var(--border-default)] rounded-md text-sm text-[var(--text-primary)] placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                disabled={loading}
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-[var(--text-secondary)] mb-1.5">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                className="w-full px-3 py-2 bg-[var(--bg-subtle)] border border-[var(--border-default)] rounded-md text-sm text-[var(--text-primary)] placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                disabled={loading}
              />
            </div>

            <button
              type="submit"
              disabled={loading || !email || !password}
              className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white text-sm font-medium rounded-md transition-colors"
            >
              {loading ? "Signing in..." : "Sign In"}
            </button>
          </form>

          {/* Quick Login - Only show if auto-login is enabled */}
          {config.AUTO_LOGIN_ENABLED && (
            <>
              {/* Divider */}
              <div className="relative my-5">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-[var(--border-subtle)]" />
                </div>
                <div className="relative flex justify-center text-xs">
                  <span className="px-2 bg-[var(--bg-elevated)] text-[var(--text-tertiary)]">
                    or
                  </span>
                </div>
              </div>

              <button
                type="button"
                onClick={handleDefaultLogin}
                disabled={loading}
                className="w-full py-2 px-4 bg-[var(--bg-subtle)] hover:bg-[var(--bg-hover)] border border-[var(--border-default)] text-[var(--text-secondary)] text-sm rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Use Default Account
              </button>

              <p className="mt-3 text-center text-xs text-[var(--text-tertiary)]">
                Default: {config.DEFAULT_USER_EMAIL}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
