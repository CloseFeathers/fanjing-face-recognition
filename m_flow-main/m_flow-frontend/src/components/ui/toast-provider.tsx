"use client";

import { Toaster } from "sonner";

export function ToastProvider() {
  return (
    <Toaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: "#18181b",
          border: "1px solid #27272a",
          color: "#fafafa",
        },
        classNames: {
          toast: "group toast",
          title: "text-zinc-100 font-medium",
          description: "text-zinc-400 text-sm",
          actionButton: "bg-blue-600 text-white",
          cancelButton: "bg-zinc-800 text-zinc-400",
          success: "!border-emerald-500/30",
          error: "!border-red-500/30",
          warning: "!border-amber-500/30",
          info: "!border-blue-500/30",
        },
      }}
      closeButton
      richColors
      expand
    />
  );
}
