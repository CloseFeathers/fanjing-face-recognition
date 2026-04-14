"use client";

/**
 * CommandPalette Component
 *
 * Global command palette accessible via Cmd+K / Ctrl+K.
 * Provides quick navigation and actions across the application.
 *
 * Features:
 * - Fuzzy search across pages and actions
 * - Keyboard navigation (arrows, enter, escape)
 * - Recent items memory
 * - Action categories
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";
import { useUIStore, View } from "@/lib/store/ui";
import {
  Search,
  LayoutDashboard,
  Upload,
  Network,
  Activity,
  Download,
  Rocket,
  Users,
  FileText,
  Sparkles,
  Code2,
  CornerDownLeft,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
  category: "navigation" | "action" | "recent";
  shortcut?: string;
  action: () => void;
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

// ============================================================================
// Command Registry
// ============================================================================

function useCommands(): CommandItem[] {
  const { setCurrentView } = useUIStore();

  return useMemo(
    () => [
      // Navigation
      {
        id: "nav-dashboard",
        label: "Dashboard",
        description: "View overview and statistics",
        icon: <LayoutDashboard size={16} />,
        category: "navigation" as const,
        shortcut: "⌘1",
        action: () => setCurrentView("dashboard"),
      },
      {
        id: "nav-setup",
        label: "Setup",
        description: "Configure LLM and services",
        icon: <Rocket size={16} />,
        category: "navigation" as const,
        shortcut: "⌘2",
        action: () => setCurrentView("setup"),
      },
      {
        id: "nav-ingest",
        label: "Ingest",
        description: "Upload documents and files",
        icon: <Upload size={16} />,
        category: "navigation" as const,
        shortcut: "⌘3",
        action: () => setCurrentView("memorize-ingest"),
      },
      {
        id: "nav-search",
        label: "Search",
        description: "Query your knowledge base",
        icon: <Search size={16} />,
        category: "navigation" as const,
        shortcut: "⌘4",
        action: () => setCurrentView("retrieve-episodic"),
      },
      {
        id: "nav-graph",
        label: "Graph",
        description: "Explore knowledge graph",
        icon: <Network size={16} />,
        category: "navigation" as const,
        shortcut: "⌘5",
        action: () => setCurrentView("memories"),
      },
      {
        id: "nav-audit",
        label: "Monitoring & Audit",
        description: "View system status and search history",
        icon: <Activity size={16} />,
        category: "navigation" as const,
        action: () => setCurrentView("audit"),
      },
      {
        id: "nav-export",
        label: "Export",
        description: "Export knowledge data",
        icon: <Download size={16} />,
        category: "navigation" as const,
        action: () => setCurrentView("export"),
      },
      {
        id: "nav-users",
        label: "Users",
        description: "User management",
        icon: <Users size={16} />,
        category: "navigation" as const,
        action: () => setCurrentView("users"),
      },
      // Actions
      {
        id: "action-add-memory",
        label: "Add Memory",
        description: "Add new text content",
        icon: <FileText size={16} />,
        category: "action" as const,
        shortcut: "⌘N",
        action: () => setCurrentView("memorize-add"),
      },
      {
        id: "action-episodic-search",
        label: "Episodic Search",
        description: "Search with natural language",
        icon: <Sparkles size={16} />,
        category: "action" as const,
        action: () => setCurrentView("retrieve-episodic"),
      },
      {
        id: "action-cypher",
        label: "Cypher Query",
        description: "Run graph database queries",
        icon: <Code2 size={16} />,
        category: "action" as const,
        action: () => setCurrentView("retrieve-cypher"),
      },
    ],
    [setCurrentView]
  );
}

// ============================================================================
// Search/Filter Logic
// ============================================================================

function filterCommands(commands: CommandItem[], query: string): CommandItem[] {
  if (!query.trim()) {
    return commands;
  }

  const lowerQuery = query.toLowerCase();
  return commands.filter(
    (cmd) =>
      cmd.label.toLowerCase().includes(lowerQuery) ||
      cmd.description?.toLowerCase().includes(lowerQuery)
  );
}

function groupByCategory(
  commands: CommandItem[]
): Record<string, CommandItem[]> {
  return commands.reduce((acc, cmd) => {
    if (!acc[cmd.category]) {
      acc[cmd.category] = [];
    }
    acc[cmd.category].push(cmd);
    return acc;
  }, {} as Record<string, CommandItem[]>);
}

const CATEGORY_LABELS: Record<string, string> = {
  navigation: "Navigation",
  action: "Actions",
  recent: "Recent",
};

// ============================================================================
// Command Item Component
// ============================================================================

interface CommandItemRowProps {
  item: CommandItem;
  isSelected: boolean;
  onSelect: () => void;
}

function CommandItemRow({ item, isSelected, onSelect }: CommandItemRowProps) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left",
        "transition-colors",
        isSelected
          ? "bg-zinc-800 text-zinc-100"
          : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
      )}
    >
      <span className={cn(isSelected ? "text-zinc-100" : "text-zinc-500")}>
        {item.icon}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-[13px] font-medium truncate">{item.label}</p>
        {item.description && (
          <p className="text-[11px] text-zinc-500 truncate">{item.description}</p>
        )}
      </div>
      {item.shortcut && (
        <kbd className="px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 bg-zinc-800 rounded border border-zinc-700">
          {item.shortcut}
        </kbd>
      )}
    </button>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function CommandPalette({ isOpen, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const commands = useCommands();

  const filteredCommands = useMemo(
    () => filterCommands(commands, query),
    [commands, query]
  );

  const groupedCommands = useMemo(
    () => groupByCategory(filteredCommands),
    [filteredCommands]
  );

  // Flat list for keyboard navigation
  const flatList = useMemo(
    () => Object.values(groupedCommands).flat(),
    [groupedCommands]
  );

  // Reset selection when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((i) => Math.min(i + 1, flatList.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (flatList[selectedIndex]) {
            flatList[selectedIndex].action();
            onClose();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    },
    [flatList, selectedIndex, onClose]
  );

  // Handle item selection
  const handleSelect = useCallback(
    (item: CommandItem) => {
      item.action();
      onClose();
    },
    [onClose]
  );

  // SSR safety check
  const [isMounted, setIsMounted] = React.useState(false);
  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isOpen || !isMounted) return null;

  // Calculate selected index offset for each category
  let currentIndex = 0;
  const getItemIndex = (categoryIndex: number, itemIndex: number): number => {
    let offset = 0;
    const categories = Object.keys(groupedCommands);
    for (let i = 0; i < categoryIndex; i++) {
      offset += groupedCommands[categories[i]].length;
    }
    return offset + itemIndex;
  };

  return createPortal(
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 z-50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="fixed top-[20%] left-1/2 -translate-x-1/2 w-full max-w-lg z-50">
        <div
          className={cn(
            "bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl",
            "overflow-hidden"
          )}
          onKeyDown={handleKeyDown}
        >
          {/* Search Input */}
          <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-800">
            <Search size={18} className="text-zinc-500 shrink-0" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search commands..."
              className={cn(
                "flex-1 bg-transparent text-[14px] text-zinc-100",
                "placeholder-zinc-500 outline-none"
              )}
            />
            <kbd className="px-1.5 py-0.5 text-[10px] text-zinc-500 bg-zinc-800 rounded border border-zinc-700">
              ESC
            </kbd>
          </div>

          {/* Results */}
          <div className="max-h-[360px] overflow-y-auto p-2">
            {flatList.length === 0 ? (
              <div className="py-8 text-center">
                <p className="text-[13px] text-zinc-500">No results found</p>
              </div>
            ) : (
              Object.entries(groupedCommands).map(([category, items], catIdx) => (
                <div key={category} className={cn(catIdx > 0 && "mt-3")}>
                  <p className="px-3 py-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                    {CATEGORY_LABELS[category] || category}
                  </p>
                  <div className="space-y-0.5">
                    {items.map((item, itemIdx) => (
                      <CommandItemRow
                        key={item.id}
                        item={item}
                        isSelected={getItemIndex(catIdx, itemIdx) === selectedIndex}
                        onSelect={() => handleSelect(item)}
                      />
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-4 py-2 border-t border-zinc-800 text-[10px] text-zinc-500">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <kbd className="px-1 py-0.5 bg-zinc-800 rounded border border-zinc-700">↑</kbd>
                <kbd className="px-1 py-0.5 bg-zinc-800 rounded border border-zinc-700">↓</kbd>
                to navigate
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1 py-0.5 bg-zinc-800 rounded border border-zinc-700">
                  <CornerDownLeft size={10} />
                </kbd>
                to select
              </span>
            </div>
          </div>
        </div>
      </div>
    </>,
    document.body
  );
}

// ============================================================================
// Hook for Global Keyboard Shortcut
// ============================================================================

export function useCommandPalette() {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+K or Ctrl+K
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    toggle: () => setIsOpen((prev) => !prev),
  };
}

CommandPalette.displayName = "CommandPalette";

export default CommandPalette;
