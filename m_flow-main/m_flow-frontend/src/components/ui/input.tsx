"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: string;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, error, ...props }, ref) => {
    return (
      <div className="w-full">
        <input
          type={type}
          className={cn(
            "flex h-9 w-full rounded-lg bg-[var(--bg-surface)] border px-3 py-2 text-[13px] text-[var(--text-primary)] transition-colors",
            "placeholder:text-[var(--text-muted)]",
            "focus:outline-none focus:ring-2",
            "disabled:cursor-not-allowed disabled:opacity-50",
            error
              ? "border-[var(--error)] focus:border-[var(--error)] focus:ring-[var(--error)]/20"
              : "border-[var(--border-default)] focus:border-[var(--accent)] focus:ring-[var(--accent-muted)]",
            className
          )}
          ref={ref}
          {...props}
        />
        {error && (
          <p className="mt-1 text-[12px] text-[var(--error)]">{error}</p>
        )}
      </div>
    );
  }
);
Input.displayName = "Input";

export { Input };
