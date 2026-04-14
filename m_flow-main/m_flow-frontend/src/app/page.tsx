"use client";

import { useState, useMemo } from "react";
import { Sidebar, Header, MainContent } from "@/components/layout";
import { CommandPalette, useKeyboardShortcuts, CommandItem } from "@/components/ui/command-palette";
import { useUIStore } from "@/lib/store";
import {
  Search,
  Upload,
  Network,
  HelpCircle,
  Database,
  Code,
  Rocket,
  Activity,
  Download,
  Users,
  LayoutDashboard,
  Sparkles,
  Layers,
  FileSearch,
  Plus,
  Cog,
  FileEdit,
  Shield,
  ScrollText,
} from "lucide-react";

export default function Home() {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const { setCurrentView } = useUIStore();

  // Command palette commands list
  const commands: CommandItem[] = useMemo(
    () => [
      // Setup
      {
        id: "setup",
        icon: <Rocket className="h-4 w-4" />,
        label: "Quick Start Guide",
        shortcut: "⌘1",
        category: "Setup",
        onSelect: () => setCurrentView("setup"),
      },
      
      // Dashboard
      {
        id: "dashboard",
        icon: <LayoutDashboard className="h-4 w-4" />,
        label: "Dashboard",
        shortcut: "⌘D",
        category: "Navigation",
        onSelect: () => setCurrentView("dashboard"),
      },

      // Memorize
      {
        id: "memorize-ingest",
        icon: <Upload className="h-4 w-4" />,
        label: "Ingest Data",
        category: "Memorize",
        onSelect: () => setCurrentView("memorize-ingest"),
      },
      {
        id: "memorize-add",
        icon: <Plus className="h-4 w-4" />,
        label: "Add Memory",
        shortcut: "⌘U",
        category: "Memorize",
        onSelect: () => setCurrentView("memorize-add"),
      },
      {
        id: "memorize-process",
        icon: <Cog className="h-4 w-4" />,
        label: "Memorize Process",
        category: "Memorize",
        onSelect: () => setCurrentView("memorize-process"),
      },
      {
        id: "memorize-manual",
        icon: <FileEdit className="h-4 w-4" />,
        label: "Structured Import",
        category: "Memorize",
        onSelect: () => setCurrentView("memorize-manual"),
      },

      // Retrieve
      {
        id: "retrieve-episodic",
        icon: <Sparkles className="h-4 w-4" />,
        label: "Episodic Search",
        shortcut: "⌘S",
        category: "Retrieve",
        onSelect: () => setCurrentView("retrieve-episodic"),
      },
      {
        id: "retrieve-triplet",
        icon: <Network className="h-4 w-4" />,
        label: "Triplet Search",
        category: "Retrieve",
        onSelect: () => setCurrentView("retrieve-triplet"),
      },
      {
        id: "retrieve-procedural",
        icon: <Layers className="h-4 w-4" />,
        label: "Procedural Search",
        category: "Retrieve",
        onSelect: () => setCurrentView("retrieve-procedural"),
      },
      {
        id: "retrieve-cypher",
        icon: <Code className="h-4 w-4" />,
        label: "Cypher Query",
        category: "Retrieve",
        onSelect: () => setCurrentView("retrieve-cypher"),
      },
      {
        id: "retrieve-lexical",
        icon: <FileSearch className="h-4 w-4" />,
        label: "Lexical Search",
        category: "Retrieve",
        onSelect: () => setCurrentView("retrieve-lexical"),
      },

      // Monitoring & Audit
      {
        id: "monitoring",
        icon: <Activity className="h-4 w-4" />,
        label: "Monitoring & Audit",
        category: "Navigation",
        onSelect: () => setCurrentView("audit"),
      },

      // Memories
      {
        id: "memories",
        icon: <Network className="h-4 w-4" />,
        label: "Knowledge Graph",
        shortcut: "⌘G",
        category: "Navigation",
        onSelect: () => setCurrentView("memories"),
      },

      // Export
      {
        id: "export",
        icon: <Download className="h-4 w-4" />,
        label: "Memory Export",
        category: "Navigation",
        onSelect: () => setCurrentView("export"),
      },

      // System
      {
        id: "users",
        icon: <Users className="h-4 w-4" />,
        label: "User Management",
        category: "System",
        onSelect: () => setCurrentView("users"),
      },
      {
        id: "permissions",
        icon: <Shield className="h-4 w-4" />,
        label: "Permissions",
        category: "System",
        onSelect: () => setCurrentView("permissions"),
      },
      {
        id: "audit",
        icon: <ScrollText className="h-4 w-4" />,
        label: "Audit Log",
        category: "System",
        onSelect: () => setCurrentView("audit"),
      },
    ],
    [setCurrentView]
  );

  // Global keyboard shortcuts
  useKeyboardShortcuts({
    "mod+k": () => setCommandPaletteOpen(true),
  });

  return (
    <div className="min-h-screen bg-[var(--bg-base)]">
      <Sidebar />
      <Header />
      <MainContent />

      {/* Command Palette */}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        commands={commands}
      />
    </div>
  );
}
