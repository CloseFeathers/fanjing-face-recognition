"use client";

import React, { useState, useRef, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronRight, Check, Database, User } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface DatasetItem {
  id: string;
  name: string;
  ownerId?: string;
}

export interface UserInfo {
  id: string;
  name: string;
}

interface DatasetMultiSelectProps {
  datasets: DatasetItem[];
  users?: UserInfo[];
  selected: string[]; // dataset names
  onChange: (selected: string[]) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

interface GroupedDatasets {
  userId: string;
  userName: string;
  datasets: DatasetItem[];
}

// ============================================================================
// Main Component
// ============================================================================

export function DatasetMultiSelect({
  datasets,
  users = [],
  selected,
  onChange,
  disabled = false,
  placeholder = "Select datasets...",
  className,
}: DatasetMultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedUsers, setExpandedUsers] = useState<Set<string>>(new Set());
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

  // Create user lookup map
  const userMap = useMemo(() => {
    const map = new Map<string, string>();
    users.forEach(u => map.set(u.id, u.name));
    return map;
  }, [users]);

  // Group datasets by owner
  const groupedDatasets = useMemo(() => {
    const groups = new Map<string, DatasetItem[]>();
    const unknownGroup: DatasetItem[] = [];

    datasets.forEach(ds => {
      if (ds.ownerId) {
        const existing = groups.get(ds.ownerId) || [];
        existing.push(ds);
        groups.set(ds.ownerId, existing);
      } else {
        unknownGroup.push(ds);
      }
    });

    const result: GroupedDatasets[] = [];
    
    groups.forEach((dsList, userId) => {
      result.push({
        userId,
        userName: userMap.get(userId) || userId.slice(0, 8),
        datasets: dsList,
      });
    });

    // Sort by user name
    result.sort((a, b) => a.userName.localeCompare(b.userName));

    // Add unknown group at end if exists
    if (unknownGroup.length > 0) {
      result.push({
        userId: "__unknown__",
        userName: "Unknown Owner",
        datasets: unknownGroup,
      });
    }

    return result;
  }, [datasets, userMap]);

  // Check states
  const isAllSelected = selected.length === datasets.length && datasets.length > 0;
  const isNoneSelected = selected.length === 0;

  const isUserAllSelected = (userId: string) => {
    const group = groupedDatasets.find(g => g.userId === userId);
    if (!group) return false;
    return group.datasets.every(ds => selected.includes(ds.name));
  };

  const isUserPartialSelected = (userId: string) => {
    const group = groupedDatasets.find(g => g.userId === userId);
    if (!group) return false;
    const selectedCount = group.datasets.filter(ds => selected.includes(ds.name)).length;
    return selectedCount > 0 && selectedCount < group.datasets.length;
  };

  // Toggle functions
  const toggleAll = () => {
    if (isAllSelected) {
      onChange([]);
    } else {
      onChange(datasets.map(ds => ds.name));
    }
  };

  const toggleUser = (userId: string) => {
    const group = groupedDatasets.find(g => g.userId === userId);
    if (!group) return;

    const userDatasetNames = group.datasets.map(ds => ds.name);
    const allSelected = isUserAllSelected(userId);

    if (allSelected) {
      // Deselect all of this user's datasets
      onChange(selected.filter(name => !userDatasetNames.includes(name)));
    } else {
      // Select all of this user's datasets
      const newSelected = new Set(selected);
      userDatasetNames.forEach(name => newSelected.add(name));
      onChange(Array.from(newSelected));
    }
  };

  const toggleDataset = (datasetName: string) => {
    if (selected.includes(datasetName)) {
      onChange(selected.filter(name => name !== datasetName));
    } else {
      onChange([...selected, datasetName]);
    }
  };

  const toggleUserExpanded = (userId: string) => {
    const newExpanded = new Set(expandedUsers);
    if (newExpanded.has(userId)) {
      newExpanded.delete(userId);
    } else {
      newExpanded.add(userId);
    }
    setExpandedUsers(newExpanded);
  };

  // Display text
  // Note: isNoneSelected (empty array) means "search all" - show placeholder which defaults to "All datasets"
  const displayText = useMemo(() => {
    if (isNoneSelected) return placeholder; // Empty = search all, show placeholder
    if (isAllSelected) return "All datasets";
    if (selected.length === 1) return selected[0];
    return `${selected.length} datasets`;
  }, [selected, isAllSelected, isNoneSelected, placeholder]);

  return (
    <div className={cn("relative", className)} ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={cn(
          "flex items-center justify-between gap-2 w-full px-2 py-1 rounded",
          "bg-[var(--bg-surface)] border border-[var(--border-subtle)]",
          "text-xs text-[var(--text-secondary)]",
          "hover:border-[var(--border-default)] transition-colors",
          "focus:outline-none focus:ring-1 focus:ring-[var(--border-default)]",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <Database size={12} className="shrink-0 text-[var(--text-muted)]" />
          <span className="truncate">{displayText}</span>
        </div>
        <ChevronDown 
          size={12} 
          className={cn(
            "shrink-0 text-[var(--text-muted)] transition-transform",
            isOpen && "rotate-180"
          )} 
        />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-64 max-h-72 overflow-auto py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded-md shadow-lg z-50">
          {/* All Datasets Option */}
          <button
            type="button"
            onClick={toggleAll}
            className={cn(
              "w-full flex items-center gap-2 px-3 py-2 text-left text-xs",
              "hover:bg-[var(--bg-hover)] transition-colors",
              isAllSelected && "bg-[var(--bg-hover)]"
            )}
          >
            <div className={cn(
              "w-3.5 h-3.5 rounded border flex items-center justify-center",
              isAllSelected 
                ? "bg-[var(--text-primary)] border-[var(--text-primary)]" 
                : "border-[var(--border-default)]"
            )}>
              {isAllSelected && <Check size={10} className="text-[var(--bg-base)]" />}
            </div>
            <span className="font-medium text-[var(--text-primary)]">All datasets</span>
            <span className="ml-auto text-[var(--text-muted)]">({datasets.length})</span>
          </button>

          <div className="h-px bg-[var(--border-subtle)] my-1" />

          {/* Grouped by User */}
          {groupedDatasets.map((group) => (
            <div key={group.userId}>
              {/* User Header */}
              <div className="flex items-center">
                <button
                  type="button"
                  onClick={() => toggleUserExpanded(group.userId)}
                  className="p-1.5 hover:bg-[var(--bg-hover)] transition-colors"
                >
                  {expandedUsers.has(group.userId) ? (
                    <ChevronDown size={12} className="text-[var(--text-muted)]" />
                  ) : (
                    <ChevronRight size={12} className="text-[var(--text-muted)]" />
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => toggleUser(group.userId)}
                  className={cn(
                    "flex-1 flex items-center gap-2 px-1 py-1.5 text-left text-xs",
                    "hover:bg-[var(--bg-hover)] transition-colors"
                  )}
                >
                  <div className={cn(
                    "w-3.5 h-3.5 rounded border flex items-center justify-center",
                    isUserAllSelected(group.userId)
                      ? "bg-[var(--text-primary)] border-[var(--text-primary)]"
                      : isUserPartialSelected(group.userId)
                        ? "bg-[var(--text-muted)] border-[var(--text-muted)]"
                        : "border-[var(--border-default)]"
                  )}>
                    {isUserAllSelected(group.userId) && <Check size={10} className="text-[var(--bg-base)]" />}
                    {isUserPartialSelected(group.userId) && (
                      <div className="w-1.5 h-1.5 bg-[var(--bg-base)] rounded-sm" />
                    )}
                  </div>
                  <User size={11} className="text-[var(--text-muted)]" />
                  <span className="text-[var(--text-secondary)]">{group.userName}</span>
                  <span className="ml-auto text-[var(--text-muted)]">({group.datasets.length})</span>
                </button>
              </div>

              {/* User's Datasets */}
              {expandedUsers.has(group.userId) && (
                <div className="pl-6">
                  {group.datasets.map((ds) => (
                    <button
                      key={ds.id}
                      type="button"
                      onClick={() => toggleDataset(ds.name)}
                      className={cn(
                        "w-full flex items-center gap-2 px-3 py-1.5 text-left text-xs",
                        "hover:bg-[var(--bg-hover)] transition-colors",
                        selected.includes(ds.name) && "bg-[var(--bg-hover)]"
                      )}
                    >
                      <div className={cn(
                        "w-3.5 h-3.5 rounded border flex items-center justify-center",
                        selected.includes(ds.name)
                          ? "bg-[var(--text-primary)] border-[var(--text-primary)]"
                          : "border-[var(--border-default)]"
                      )}>
                        {selected.includes(ds.name) && <Check size={10} className="text-[var(--bg-base)]" />}
                      </div>
                      <Database size={11} className="text-[var(--text-muted)]" />
                      <span className="text-[var(--text-secondary)] truncate">{ds.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          {datasets.length === 0 && (
            <div className="px-3 py-4 text-xs text-[var(--text-muted)] text-center">
              No datasets available
            </div>
          )}
        </div>
      )}
    </div>
  );
}
