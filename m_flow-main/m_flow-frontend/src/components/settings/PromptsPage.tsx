"use client";

import React, { useState, useEffect } from "react";
import { usePrompts, usePrompt, useUpdatePrompt, useResetPrompt, useResetAllPrompts } from "@/hooks/use-api";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import {
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Edit3,
  RotateCcw,
  Check,
  X,
  Info,
  FileText,
  Sparkles,
  MessageSquare,
  Layers,
  Zap,
  Search,
  Network,
  FlaskConical,
  Users,
} from "lucide-react";
import type { Prompt, PromptCategory } from "@/types";

// ============================================================================
// Category Icons & Labels
// ============================================================================

const CATEGORY_CONFIG: Record<string, { icon: React.ElementType; label: string; description: string }> = {
  answering: {
    icon: MessageSquare,
    label: "Question Answering",
    description: "Question answering and context generation",
  },
  episodic: {
    icon: Layers,
    label: "Episodic Memory",
    description: "Episode/Facet extraction and routing",
  },
  entity: {
    icon: Users,
    label: "Entity Processing",
    description: "Entity extraction and description generation",
  },
  graph: {
    icon: Network,
    label: "Graph Generation",
    description: "Knowledge graph construction",
  },
  summarization: {
    icon: FileText,
    label: "Summarization",
    description: "Content summarization tasks",
  },
  processing: {
    icon: Zap,
    label: "Text Processing",
    description: "Sentence grouping and classification",
  },
  evaluation: {
    icon: FlaskConical,
    label: "Evaluation",
    description: "Benchmarks and testing prompts",
  },
  other: {
    icon: Sparkles,
    label: "Other",
    description: "Miscellaneous prompts",
  },
};

// ============================================================================
// Prompt Editor Component
// ============================================================================

interface PromptEditorProps {
  filename: string;
  onClose: () => void;
}

function PromptEditor({ filename, onClose }: PromptEditorProps) {
  const { data: prompt, isLoading, error } = usePrompt(filename);
  const updatePrompt = useUpdatePrompt();
  const resetPrompt = useResetPrompt();
  
  const [content, setContent] = useState("");
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (prompt) {
      setContent(prompt.content);
      setHasChanges(false);
    }
  }, [prompt]);

  const handleContentChange = (newContent: string) => {
    setContent(newContent);
    setHasChanges(newContent !== prompt?.content);
  };

  const handleSave = async () => {
    try {
      await updatePrompt.mutateAsync({ filename, content });
      setHasChanges(false);
      toast.success("Prompt saved successfully");
    } catch {
      toast.error("Failed to save prompt");
    }
  };

  const handleReset = async () => {
    if (!confirm("Reset this prompt to its default value? Your changes will be lost.")) {
      return;
    }
    try {
      const result = await resetPrompt.mutateAsync(filename);
      setContent(result.content);
      setHasChanges(false);
      toast.success("Prompt reset to default");
    } catch {
      toast.error("Failed to reset prompt");
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 size={20} className="text-[var(--text-muted)] animate-spin" />
      </div>
    );
  }

  if (error || !prompt) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <AlertCircle size={24} className="text-[var(--error)] mb-2" />
        <p className="text-sm text-[var(--text-muted)]">Failed to load prompt</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-[var(--text-primary)]">{prompt.name}</h3>
          {prompt.description && (
            <p className="text-xs text-[var(--text-muted)] mt-0.5">{prompt.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {prompt.is_modified && (
            <span className="px-2 py-0.5 text-[10px] bg-amber-500/20 text-amber-400 rounded">
              Modified
            </span>
          )}
          <button
            onClick={onClose}
            className="p-1 text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Editor */}
      <div className="relative">
        <textarea
          value={content}
          onChange={(e) => handleContentChange(e.target.value)}
          className={cn(
            "w-full h-80 p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg",
            "text-sm font-mono text-[var(--text-primary)] placeholder:text-[var(--text-muted)]",
            "focus:outline-none focus:border-[var(--border-default)] resize-none"
          )}
          placeholder="Enter prompt template..."
        />
        <div className="absolute bottom-3 right-3 text-[10px] text-[var(--text-muted)]">
          {content.length} chars
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <button
          onClick={handleReset}
          disabled={resetPrompt.isPending}
          className={cn(
            "flex items-center gap-2 px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors",
            resetPrompt.isPending && "opacity-50 cursor-not-allowed"
          )}
        >
          <RotateCcw size={12} />
          Reset to Default
        </button>
        <div className="flex items-center gap-2">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || updatePrompt.isPending}
            className={cn(
              "flex items-center gap-2 px-4 py-1.5 text-xs bg-[var(--text-primary)] text-[var(--bg-base)] rounded transition-opacity",
              (!hasChanges || updatePrompt.isPending) && "opacity-50 cursor-not-allowed"
            )}
          >
            {updatePrompt.isPending ? (
              <Loader2 size={12} className="animate-spin" />
            ) : (
              <Check size={12} />
            )}
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Category Section Component
// ============================================================================

interface CategorySectionProps {
  category: PromptCategory;
  onEditPrompt: (filename: string) => void;
}

function CategorySection({ category, onEditPrompt }: CategorySectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const config = CATEGORY_CONFIG[category.category] || CATEGORY_CONFIG.other;
  const Icon = config.icon;

  const modifiedCount = category.prompts.filter((p) => p.is_modified).length;

  return (
    <div className="border border-[var(--border-subtle)] rounded-lg overflow-hidden">
      {/* Category Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 bg-[var(--bg-surface)] hover:bg-[var(--bg-elevated)] transition-colors"
      >
        <div className="flex items-center gap-3">
          <Icon size={16} className="text-[var(--text-muted)]" />
          <div className="text-left">
            <h3 className="text-sm font-medium text-[var(--text-primary)]">{config.label}</h3>
            <p className="text-xs text-[var(--text-muted)]">{config.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[var(--text-muted)]">
            {category.prompts.length} prompts
          </span>
          {modifiedCount > 0 && (
            <span className="px-2 py-0.5 text-[10px] bg-amber-500/20 text-amber-400 rounded">
              {modifiedCount} modified
            </span>
          )}
          {isExpanded ? (
            <ChevronDown size={16} className="text-[var(--text-muted)]" />
          ) : (
            <ChevronRight size={16} className="text-[var(--text-muted)]" />
          )}
        </div>
      </button>

      {/* Prompt List */}
      {isExpanded && (
        <div className="divide-y divide-[var(--border-subtle)]">
          {category.prompts.map((prompt) => (
            <div
              key={prompt.filename}
              className="flex items-center justify-between p-3 hover:bg-[var(--bg-surface)] transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                <FileText size={14} className="text-[var(--text-muted)] flex-shrink-0" />
                <div className="min-w-0">
                  <p className="text-sm text-[var(--text-primary)] truncate">{prompt.name}</p>
                  {prompt.description && (
                    <p className="text-xs text-[var(--text-muted)] truncate">{prompt.description}</p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {prompt.is_modified && (
                  <span className="w-2 h-2 bg-amber-400 rounded-full" title="Modified" />
                )}
                <button
                  onClick={() => onEditPrompt(prompt.filename)}
                  className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] rounded transition-colors"
                  title="Edit prompt"
                >
                  <Edit3 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

export function PromptsPage() {
  const { data: categories, isLoading, error, refetch } = usePrompts();
  const resetAllPrompts = useResetAllPrompts();
  
  const [editingPrompt, setEditingPrompt] = useState<string | null>(null);

  const handleResetAll = async () => {
    if (!confirm("Reset ALL prompts to their default values? This cannot be undone.")) {
      return;
    }
    try {
      const result = await resetAllPrompts.mutateAsync();
      toast.success(result.message);
      refetch();
    } catch {
      toast.error("Failed to reset prompts");
    }
  };

  const totalPrompts = categories?.reduce((sum, cat) => sum + cat.prompts.length, 0) || 0;
  const modifiedCount = categories?.reduce(
    (sum, cat) => sum + cat.prompts.filter((p) => p.is_modified).length,
    0
  ) || 0;

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <Loader2 size={20} className="text-[var(--text-muted)] animate-spin mb-3" />
        <p className="text-sm text-[var(--text-muted)]">Loading prompts...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <AlertCircle size={24} className="text-[var(--error)] mb-3" />
        <p className="text-sm text-[var(--text-primary)] mb-1">Failed to load prompts</p>
        <p className="text-xs text-[var(--text-muted)]">Check if the backend is running</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-lg font-medium text-[var(--text-primary)]">Prompt Templates</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">Customize LLM prompts for knowledge extraction and processing.</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-sm text-[var(--text-primary)]">{totalPrompts} prompts</p>
            {modifiedCount > 0 && (
              <p className="text-xs text-amber-400">{modifiedCount} modified</p>
            )}
          </div>
          <button
            onClick={handleResetAll}
            disabled={resetAllPrompts.isPending || modifiedCount === 0}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 text-xs border border-[var(--border-subtle)] rounded hover:bg-[var(--bg-surface)] transition-colors",
              (resetAllPrompts.isPending || modifiedCount === 0) && "opacity-50 cursor-not-allowed"
            )}
          >
            <RotateCcw size={12} />
            Reset All
          </button>
        </div>
      </div>

      {/* Info Banner */}
      <div className="flex items-start gap-3 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <Info size={16} className="text-blue-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-blue-300">
          <p className="mb-1">
            <strong>Prompt templates</strong> control how the LLM processes your data.
          </p>
          <p>
            Changes take effect immediately for new processing jobs. Modified prompts are marked with an amber indicator.
            Use "Reset to Default" to restore individual prompts.
          </p>
        </div>
      </div>

      {/* Editor Panel or Category List */}
      {editingPrompt ? (
        <div className="p-6 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
          <PromptEditor
            filename={editingPrompt}
            onClose={() => setEditingPrompt(null)}
          />
        </div>
      ) : (
        <div className="space-y-3">
          {categories?.map((category) => (
            <CategorySection
              key={category.category}
              category={category}
              onEditPrompt={setEditingPrompt}
            />
          ))}
        </div>
      )}
    </div>
  );
}
