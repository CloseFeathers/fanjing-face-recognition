"use client";

/**
 * Sidebar Component
 *
 * Main navigation sidebar for M-Flow application.
 * Provides hierarchical navigation with collapsible sections.
 *
 * Features:
 * - Collapsible sidebar
 * - Nested navigation items
 * - Active state highlighting
 * - Keyboard accessible
 */

import React, { useState } from "react";
import { useUIStore, View } from "@/lib/store/ui";
import { cn } from "@/lib/utils";
import {
  Rocket,
  LayoutDashboard,
  Upload,
  Search,
  Activity,
  Network,
  Download,
  Users,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  FileText,
  FileCode,
  Sparkles,
  Code2,
  FileSearch,
  Layers,
  Edit3,
  FolderInput,
  BookOpen,
  Trash2,
} from "lucide-react";

// ============================================================================
// Navigation Item Types
// ============================================================================

interface NavItem {
  id: View;
  label: string;
  icon: React.ReactNode;
  children?: NavItem[];
  externalUrl?: string;
}

// ============================================================================
// Navigation Configuration
// ============================================================================

const mainNavItems: NavItem[] = [
  {
    id: "setup",
    label: "Setup",
    icon: <Rocket size={18} strokeWidth={1.5} />,
  },
  {
    id: "dashboard",
    label: "Dashboard",
    icon: <LayoutDashboard size={18} strokeWidth={1.5} />,
  },
  {
    id: "memorize",
    label: "Import Data",
    icon: <Upload size={18} strokeWidth={1.5} />,
    children: [
      { id: "memorize-ingest", label: "Quick Import", icon: <FolderInput size={16} strokeWidth={1.5} /> },
      { id: "memorize-add", label: "Add Content", icon: <Edit3 size={16} strokeWidth={1.5} /> },
      { id: "memorize-process", label: "Build Graph", icon: <Sparkles size={16} strokeWidth={1.5} /> },
      { id: "memorize-manual", label: "Structured Entry", icon: <FileText size={16} strokeWidth={1.5} /> },
      { id: "memorize-learn", label: "Extract Procedures", icon: <BookOpen size={16} strokeWidth={1.5} /> },
    ],
  },
  {
    id: "retrieve",
    label: "Search",
    icon: <Search size={18} strokeWidth={1.5} />,
    children: [
      { id: "retrieve-episodic", label: "Episodic Search", icon: <Sparkles size={16} strokeWidth={1.5} /> },
      { id: "retrieve-triplet", label: "Graph Triplets", icon: <Network size={16} strokeWidth={1.5} /> },
      { id: "retrieve-procedural", label: "Procedural Search", icon: <Layers size={16} strokeWidth={1.5} /> },
      { id: "retrieve-cypher", label: "Graph Query", icon: <Code2 size={16} strokeWidth={1.5} /> },
      { id: "retrieve-lexical", label: "Keyword Search", icon: <FileSearch size={16} strokeWidth={1.5} /> },
    ],
  },
  {
    id: "audit",
    label: "Monitoring & Audit",
    icon: <Activity size={18} strokeWidth={1.5} />,
  },
  {
    id: "memories",
    label: "Knowledge Graph",
    icon: <Network size={18} strokeWidth={1.5} />,
  },
  {
    id: "export",
    label: "Export",
    icon: <Download size={18} strokeWidth={1.5} />,
  },
];

const systemNavItems: NavItem[] = [
  {
    id: "prompts",
    label: "Prompts",
    icon: <FileCode size={18} strokeWidth={1.5} />,
  },
  {
    id: "users",
    label: "Users & Datasets",
    icon: <Users size={18} strokeWidth={1.5} />,
  },
  {
    id: "prune",
    label: "Maintenance",
    icon: <Trash2 size={18} strokeWidth={1.5} />,
  },
  {
    id: "docs",
    label: "Documentation",
    icon: <BookOpen size={18} strokeWidth={1.5} />,
    externalUrl: "https://docs.m-flow.ai",
  },
];

// ============================================================================
// Sub Navigation Item Component
// ============================================================================

interface NavSubItemProps {
  item: NavItem;
  isActive: boolean;
  onClick: () => void;
  collapsed: boolean;
}

function NavSubItem({ item, isActive, onClick, collapsed }: NavSubItemProps) {
  if (collapsed) return null;

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-2 pl-9 pr-3 py-1.5 rounded-md transition-all duration-150",
        "text-[12px]",
        isActive
          ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
          : "text-[var(--text-tertiary)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
      )}
    >
      {item.icon}
      <span>{item.label}</span>
    </button>
  );
}

// ============================================================================
// Navigation Item Component
// ============================================================================

interface NavItemComponentProps {
  item: NavItem;
  currentView: View;
  setCurrentView: (view: View) => void;
  collapsed: boolean;
  expandedItems: Set<string>;
  toggleExpanded: (id: string) => void;
}

function NavItemComponent({
  item,
  currentView,
  setCurrentView,
  collapsed,
  expandedItems,
  toggleExpanded,
}: NavItemComponentProps) {
  const hasChildren = item.children && item.children.length > 0;
  const isExpanded = expandedItems.has(item.id);
  const isActive =
    currentView === item.id ||
    (hasChildren && item.children?.some((child) => currentView === child.id));

  const handleClick = () => {
    if (item.externalUrl) {
      window.open(item.externalUrl, "_blank", "noopener,noreferrer");
      return;
    }
    if (hasChildren) {
      toggleExpanded(item.id);
    } else {
      setCurrentView(item.id);
    }
  };

  return (
    <div>
      <button
        onClick={handleClick}
        className={cn(
          "w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-150",
          isActive
            ? "bg-[var(--bg-elevated)] text-[var(--text-primary)]"
            : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]",
          collapsed && "justify-center px-0"
        )}
        title={collapsed ? item.label : undefined}
      >
        {item.icon}
        {!collapsed && (
          <>
            <span className="flex-1 text-left text-[13px]">{item.label}</span>
            {hasChildren && (
              <span className="text-[var(--text-muted)]">
                {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </span>
            )}
          </>
        )}
      </button>

      {/* Sub-menu */}
      {hasChildren && isExpanded && !collapsed && (
        <div className="mt-1 space-y-0.5">
          {item.children?.map((child) => (
            <NavSubItem
              key={child.id}
              item={child}
              isActive={currentView === child.id}
              onClick={() => setCurrentView(child.id)}
              collapsed={collapsed}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Sidebar Component
// ============================================================================

export function Sidebar() {
  const { currentView, setCurrentView, sidebarCollapsed, toggleSidebar } = useUIStore();

  // Expanded menu items state
  const [expandedItems, setExpandedItems] = useState<Set<string>>(
    new Set(["memorize", "retrieve"])
  );

  const toggleExpanded = (id: string) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 h-full z-50 flex flex-col transition-all duration-200",
        "bg-[var(--bg-base)] border-r border-[var(--border-subtle)]",
        sidebarCollapsed ? "w-14" : "w-56"
      )}
    >
      {/* Logo */}
      <div className="flex items-center h-14 px-6 border-b border-[var(--border-subtle)]">
        <span 
          className="text-[14px] font-normal text-[var(--text-primary)] tracking-[0.15em]"
          style={{ fontFamily: "'Cinzel', serif" }}
        >
          {sidebarCollapsed ? "M" : "M-flow"}
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
        {/* Main Navigation */}
        <div className="space-y-0.5">
          {mainNavItems.map((item) => (
            <NavItemComponent
              key={item.id}
              item={item}
              currentView={currentView}
              setCurrentView={setCurrentView}
              collapsed={sidebarCollapsed}
              expandedItems={expandedItems}
              toggleExpanded={toggleExpanded}
            />
          ))}
        </div>

        {/* Divider */}
        <div className="my-4 h-px bg-[var(--border-subtle)]" />

        {/* System Navigation */}
        <div className="space-y-0.5">
          {!sidebarCollapsed && (
            <p className="px-3 py-1 text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
              System
            </p>
          )}
          {systemNavItems.map((item) => (
            <NavItemComponent
              key={item.id}
              item={item}
              currentView={currentView}
              setCurrentView={setCurrentView}
              collapsed={sidebarCollapsed}
              expandedItems={expandedItems}
              toggleExpanded={toggleExpanded}
            />
          ))}
        </div>
      </nav>

      {/* Collapse Button */}
      <div className="p-2 border-t border-[var(--border-subtle)]">
        <button
          onClick={toggleSidebar}
          className={cn(
            "w-full flex items-center justify-center p-2 rounded-md",
            "text-[var(--text-muted)] hover:text-[var(--text-secondary)]",
            "hover:bg-[var(--bg-hover)] transition-colors"
          )}
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </aside>
  );
}

Sidebar.displayName = "Sidebar";

export default Sidebar;
