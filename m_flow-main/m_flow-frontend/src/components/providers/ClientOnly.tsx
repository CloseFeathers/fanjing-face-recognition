"use client";

import { useState, useEffect } from "react";
import { AppProvider } from "./AppProvider";
import { ToastProvider } from "@/components/ui/toast-provider";

export function ClientOnly({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-[var(--bg-base)] flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <AppProvider>
      {children}
      <ToastProvider />
    </AppProvider>
  );
}
