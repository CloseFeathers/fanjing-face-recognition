"use client";

import React from "react";
import { AlertTriangle, ArrowLeft, Info } from "lucide-react";
import { useUIStore } from "@/lib/store";

interface UnavailablePageProps {
  title: string;
  description: string;
  reason?: string;
  alternatives?: Array<{
    label: string;
    viewId: string;
  }>;
}

export function UnavailablePage({ 
  title, 
  description, 
  reason,
  alternatives 
}: UnavailablePageProps) {
  const { setCurrentView } = useUIStore();

  return (
    <div className="max-w-2xl mx-auto py-16">
      <div className="text-center space-y-6">
        {/* Icon */}
        <div className="flex justify-center">
          <div className="w-16 h-16 rounded-full bg-[var(--warning)]/10 flex items-center justify-center">
            <AlertTriangle size={32} className="text-[var(--warning)]" />
          </div>
        </div>

        {/* Title */}
        <div>
          <h1 className="text-xl font-medium text-[var(--text-primary)]">{title}</h1>
          <p className="text-sm text-[var(--text-muted)] mt-2">{description}</p>
        </div>

        {/* Reason */}
        {reason && (
          <div className="p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg text-left">
            <div className="flex items-start gap-3">
              <Info size={16} className="text-[var(--text-muted)] mt-0.5 flex-shrink-0" />
              <p className="text-xs text-[var(--text-secondary)]">{reason}</p>
            </div>
          </div>
        )}

        {/* Alternatives */}
        {alternatives && alternatives.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider">
              Available Alternatives
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {alternatives.map((alt) => (
                <button
                  key={alt.viewId}
                  onClick={() => setCurrentView(alt.viewId as any)}
                  className="px-4 py-2 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--text-muted)] transition-colors"
                >
                  {alt.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Back Button */}
        <button
          onClick={() => setCurrentView("dashboard")}
          className="inline-flex items-center gap-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
        >
          <ArrowLeft size={14} />
          Return to Dashboard
        </button>
      </div>
    </div>
  );
}
