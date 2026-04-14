"use client";

/**
 * GettingStartedPanel Component
 *
 * Main container for the Getting Started section of the Setup page.
 * Provides tutorials, code examples, environment configuration help,
 * and quick action links.
 *
 * Features:
 * - Tabbed navigation (Tutorials, Code Examples, Environment)
 * - Tutorial cards with categories
 * - Code examples with syntax highlighting
 * - Environment variable reference
 * - Quick action links
 *
 * @example
 * <GettingStartedPanel onTutorialSelect={(t) => console.log(t)} />
 */

import React, { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  BookOpen,
  Code,
  Settings,
  Zap,
  ChevronRight,
} from "lucide-react";
import { useDetailedHealth } from "@/hooks/use-api";

// Components
import { TutorialCard, TutorialGrid } from "./TutorialCard";
import { CodeExample, CodeExampleTabs } from "./CodeExample";
import { EnvFileHelper } from "./EnvFileHelper";
import { QuickActions } from "./QuickActions";

// Content
import {
  TUTORIALS,
  CODE_EXAMPLES,
  ENV_SECTIONS,
  QUICK_ACTIONS,
  TUTORIAL_CATEGORIES,
  getTutorialsByCategory,
  getCodeExamplesByLanguage,
} from "@/content/tutorials";

import type { TutorialOption, TutorialCategory } from "@/types/setup";

// ============================================================================
// Types
// ============================================================================

export interface GettingStartedPanelProps {
  /** Callback when a tutorial is selected */
  onTutorialSelect?: (tutorial: TutorialOption) => void;
  /** Initial active tab */
  initialTab?: TabId;
  /** Additional CSS classes */
  className?: string;
}

type TabId = "tutorials" | "code" | "env";

interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

// ============================================================================
// Tab Configuration
// ============================================================================

const TABS: Tab[] = [
  { id: "tutorials", label: "Tutorials", icon: <BookOpen size={14} /> },
  { id: "code", label: "Code Examples", icon: <Code size={14} /> },
  { id: "env", label: "Environment", icon: <Settings size={14} /> },
];

// ============================================================================
// Tab Navigation Component
// ============================================================================

interface TabNavigationProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  return (
    <div className="flex border-b border-zinc-800 overflow-x-auto">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={cn(
            "flex items-center gap-2 px-4 py-3 text-[13px] font-medium",
            "border-b-2 transition-colors whitespace-nowrap",
            activeTab === tab.id
              ? "border-zinc-100 text-zinc-100"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          )}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Tutorials Tab Content
// ============================================================================

interface TutorialsTabProps {
  onSelect?: (tutorial: TutorialOption) => void;
}

function TutorialsTab({ onSelect }: TutorialsTabProps) {
  const [selectedCategory, setSelectedCategory] = useState<TutorialCategory | "all">("all");

  const filteredTutorials = useMemo(() => {
    if (selectedCategory === "all") return TUTORIALS;
    return getTutorialsByCategory(selectedCategory);
  }, [selectedCategory]);

  return (
    <div className="space-y-6">
      {/* Category Filter */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2">
        <button
          onClick={() => setSelectedCategory("all")}
          className={cn(
            "px-3 py-1.5 rounded-lg text-[12px] font-medium whitespace-nowrap transition-colors",
            selectedCategory === "all"
              ? "bg-zinc-100 text-zinc-900"
              : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
          )}
        >
          All
        </button>
        {Object.entries(TUTORIAL_CATEGORIES).map(([key, config]) => (
          <button
            key={key}
            onClick={() => setSelectedCategory(key as TutorialCategory)}
            className={cn(
              "px-3 py-1.5 rounded-lg text-[12px] font-medium whitespace-nowrap transition-colors",
              selectedCategory === key
                ? "bg-zinc-100 text-zinc-900"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            )}
          >
            {config.title}
          </button>
        ))}
      </div>

      {/* Tutorial Grid */}
      <TutorialGrid
        tutorials={filteredTutorials}
        onSelect={onSelect}
        variant="default"
        columns={2}
      />

      {/* External Resources */}
      <div>
        <h4 className="text-[14px] font-medium text-zinc-200 mb-3">
          External Resources
        </h4>
        <QuickActions
          actions={QUICK_ACTIONS.filter((a) => a.external)}
          variant="list"
        />
      </div>
    </div>
  );
}

// ============================================================================
// Code Examples Tab Content
// ============================================================================

function CodeExamplesTab() {
  const [selectedLanguage, setSelectedLanguage] = useState<string>("python");

  const languages = [
    { id: "python", label: "Python" },
    { id: "bash", label: "cURL / Bash" },
    { id: "typescript", label: "TypeScript" },
  ];

  const filteredExamples = useMemo(() => {
    return getCodeExamplesByLanguage(selectedLanguage);
  }, [selectedLanguage]);

  return (
    <div className="space-y-6">
      {/* Language Filter */}
      <div className="flex items-center gap-2">
        {languages.map((lang) => (
          <button
            key={lang.id}
            onClick={() => setSelectedLanguage(lang.id)}
            className={cn(
              "px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors",
              selectedLanguage === lang.id
                ? "bg-zinc-100 text-zinc-900"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            )}
          >
            {lang.label}
          </button>
        ))}
      </div>

      {/* Code Examples */}
      <div className="space-y-4">
        {filteredExamples.map((example) => (
          <CodeExample
            key={example.id}
            example={example}
            showTitle={true}
            collapsible={false}
          />
        ))}
      </div>

      {/* All Examples in Tabs */}
      <div>
        <h4 className="text-[14px] font-medium text-zinc-200 mb-3">
          Compare Languages
        </h4>
        <CodeExampleTabs
          examples={[
            CODE_EXAMPLES.find((e) => e.id === "python-add")!,
            CODE_EXAMPLES.find((e) => e.id === "curl-add")!,
            CODE_EXAMPLES.find((e) => e.id === "typescript-fetch")!,
          ].filter(Boolean)}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Environment Tab Content
// ============================================================================

function EnvironmentTab() {
  const { data: healthData } = useDetailedHealth({ refetchInterval: false });

  return (
    <div className="space-y-6">
      {/* Introduction */}
      <div className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800">
        <h4 className="text-[14px] font-medium text-zinc-200 mb-2">
          Environment Configuration
        </h4>
        <p className="text-[12px] text-zinc-500">
          M-Flow uses environment variables for configuration. Create a{" "}
          <code className="text-zinc-300">.env</code> file in the project root
          with the following variables. Variables marked as "Required" must be
          set before starting the server.
        </p>
      </div>

      {/* Env File Helper */}
      <EnvFileHelper
        sections={ENV_SECTIONS}
        healthData={healthData}
        showFilter={true}
        showGenerate={true}
      />
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function GettingStartedPanel({
  onTutorialSelect,
  initialTab = "tutorials",
  className,
}: GettingStartedPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>(initialTab);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Tab Navigation */}
      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === "tutorials" && (
          <TutorialsTab onSelect={onTutorialSelect} />
        )}
        {activeTab === "code" && <CodeExamplesTab />}
        {activeTab === "env" && <EnvironmentTab />}
      </div>
    </div>
  );
}

// ============================================================================
// Display Name
// ============================================================================

GettingStartedPanel.displayName = "GettingStartedPanel";

// ============================================================================
// Default Export
// ============================================================================

export default GettingStartedPanel;
