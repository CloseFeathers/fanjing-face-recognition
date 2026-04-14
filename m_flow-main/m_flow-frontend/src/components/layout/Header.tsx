"use client";

import React, { useState, useRef, useEffect } from "react";
import { useUIStore, useAuthStore } from "@/lib/store";
import { useDatasets, useHealthCheck, useLogout } from "@/hooks/use-api";
import { cn } from "@/lib/utils";
import { 
  LogOut, 
  ChevronDown,
  Database,
  BookOpen,
} from "lucide-react";

// ============================================================================
// Dropdown menu component
// ============================================================================

interface DropdownOption {
  id: string;
  label: string;
}

interface DropdownProps {
  label: string;
  icon: React.ReactNode;
  options: DropdownOption[];
  value: string | null;
  onChange: (id: string) => void;
  placeholder?: string;
}

function Dropdown({ label, icon, options, value, onChange, placeholder = "Select..." }: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedOption = options.find(opt => opt.id === value);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors",
          "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]",
          "border border-transparent hover:border-[var(--border-subtle)]"
        )}
      >
        {icon}
        <span className="text-[12px] text-[var(--text-muted)]">{label}:</span>
        <span className="text-[13px] max-w-[120px] truncate">
          {selectedOption?.label || placeholder}
        </span>
        <ChevronDown size={14} className={cn("transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-48 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded-md shadow-lg z-50">
          {options.length === 0 ? (
            <div className="px-3 py-2 text-[12px] text-[var(--text-muted)]">
              No options available
            </div>
          ) : (
            options.map((option) => (
              <button
                key={option.id}
                onClick={() => {
                  onChange(option.id);
                  setIsOpen(false);
                }}
                className={cn(
                  "w-full px-3 py-2 text-left text-[13px] transition-colors",
                  value === option.id
                    ? "bg-[var(--bg-hover)] text-[var(--text-primary)]"
                    : "text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
                )}
              >
                {option.label}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Header main component
// ============================================================================

export function Header() {
  const { sidebarCollapsed, datasetContext, setDatasetContext, setCurrentView } = useUIStore();
  const { user, isAuthenticated } = useAuthStore();
  const { data: health } = useHealthCheck();
  const { data: datasetsData, isLoading: datasetsLoading } = useDatasets();
  const logout = useLogout();

  // Auto-select first dataset when no dataset is selected
  React.useEffect(() => {
    if (!datasetContext.datasetId && datasetsData && datasetsData.length > 0) {
      const firstDataset = datasetsData[0];
      setDatasetContext({
        datasetId: firstDataset.id,
        datasetName: firstDataset.name,
      });
    }
  }, [datasetsData, datasetContext.datasetId, setDatasetContext]);

  // Datasets from API - use name as id (important for downstream components)
  const datasets: DropdownOption[] = (datasetsData ?? []).map(ds => ({
    id: ds.name,
    label: ds.name,
  }));

  return (
    <header
      className={cn(
        "fixed top-0 right-0 h-14 bg-[var(--bg-base)]/95 backdrop-blur-sm border-b border-[var(--border-subtle)] z-40 transition-all duration-200",
        sidebarCollapsed ? "left-14" : "left-56"
      )}
    >
      <div className="flex items-center justify-between h-full px-4">
        {/* Left: Dataset selector */}
        <div className="flex items-center gap-1">
          <Dropdown
            label="Dataset"
            icon={<Database size={14} strokeWidth={1.5} />}
            options={datasets}
            value={datasetContext.datasetName}
            onChange={(name) => {
              const dataset = datasetsData?.find(d => d.name === name);
              setDatasetContext({ 
                datasetId: dataset?.id || null,
                datasetName: name 
              });
            }}
            placeholder={datasetsLoading ? "Loading..." : "Select dataset"}
          />
        </div>

        {/* Right: Status + Docs + User */}
        <div className="flex items-center gap-4">
          {/* Docs link */}
          <a
            href="https://docs.m-flow.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-2 py-1 text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            <BookOpen size={14} strokeWidth={1.5} />
            <span className="text-[12px]">Docs</span>
          </a>

          {/* Status */}
          <div className="flex items-center gap-2">
            <div
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                health?.status === "healthy" ? "bg-[var(--success)]" : "bg-[var(--error)]"
              )}
            />
            <span className="text-[11px] text-[var(--text-muted)]">
              {health?.status === "healthy" ? "Online" : "Offline"}
            </span>
          </div>

          {/* User Info */}
          {isAuthenticated && user ? (
            <div className="flex items-center gap-2 pl-2 border-l border-[var(--border-subtle)]">
              <div className="w-6 h-6 rounded-md bg-[var(--bg-elevated)] flex items-center justify-center">
                <span className="text-[11px] text-[var(--text-secondary)]">
                  {(user.username || user.email)?.[0]?.toUpperCase() || "U"}
                </span>
              </div>
              <span className="text-[12px] text-[var(--text-secondary)] max-w-[80px] truncate">
                {user.username || user.email?.split("@")[0]}
              </span>
              <button
                onClick={() => logout.mutateAsync()}
                className="p-1 text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
                title="Logout"
              >
                <LogOut size={14} strokeWidth={1.5} />
              </button>
            </div>
          ) : (
            <span className="text-[11px] text-[var(--text-muted)] pl-2 border-l border-[var(--border-subtle)]">
              Guest
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
