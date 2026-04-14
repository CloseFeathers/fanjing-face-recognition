"use client";

import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Check,
  Edit3,
  AlertTriangle,
  MessageSquare,
} from "lucide-react";

interface PromptEditorProps {
  title: string;
  description?: string;
  value: string;
  defaultValue: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  collapsible?: boolean;
  defaultExpanded?: boolean;
}

export function PromptEditor({
  title,
  description,
  value,
  defaultValue,
  onChange,
  placeholder,
  className,
  collapsible = true,
  defaultExpanded = false,
}: PromptEditorProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [isEditing, setIsEditing] = useState(false);
  const [localValue, setLocalValue] = useState(value);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  const isModified = value.trim() !== "" && value.trim() !== defaultValue.trim();

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleSave = () => {
    onChange(localValue);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setLocalValue(value);
    setIsEditing(false);
  };

  const handleReset = () => {
    if (showResetConfirm) {
      onChange("");
      setLocalValue("");
      setIsEditing(false);
      setShowResetConfirm(false);
    } else {
      setShowResetConfirm(true);
      setTimeout(() => setShowResetConfirm(false), 3000);
    }
  };

  const displayValue = value || defaultValue;

  return (
    <div
      className={cn(
        "bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg overflow-hidden",
        className
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center justify-between px-4 py-3",
          collapsible && "cursor-pointer hover:bg-[var(--bg-elevated)]"
        )}
        onClick={() => collapsible && setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <MessageSquare size={14} className="text-[var(--text-muted)]" />
          <span className="text-sm font-medium text-[var(--text-primary)]">
            {title}
          </span>
          {isModified && (
            <span className="px-1.5 py-0.5 text-[10px] bg-amber-500/10 text-amber-400 rounded">
              modified
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!isEditing && !isExpanded && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsExpanded(true);
                setIsEditing(true);
              }}
              className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] rounded transition-colors"
              title="Edit prompt"
            >
              <Edit3 size={12} />
            </button>
          )}
          {collapsible && (
            <span className="text-[var(--text-muted)]">
              {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </span>
          )}
        </div>
      </div>

      {/* Description */}
      {description && isExpanded && (
        <div className="px-4 pb-2">
          <p className="text-xs text-[var(--text-muted)]">{description}</p>
        </div>
      )}

      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-3">
          {isEditing ? (
            <>
              <textarea
                value={localValue || defaultValue}
                onChange={(e) => setLocalValue(e.target.value)}
                placeholder={placeholder || "Enter custom prompt..."}
                rows={10}
                className="w-full px-3 py-2 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-xs text-[var(--text-primary)] font-mono placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] transition-colors resize-y min-h-[120px]"
              />
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {showResetConfirm ? (
                    <button
                      onClick={handleReset}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-amber-400 bg-amber-500/10 rounded hover:bg-amber-500/20 transition-colors"
                    >
                      <AlertTriangle size={12} />
                      Confirm Reset
                    </button>
                  ) : (
                    <button
                      onClick={handleReset}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] rounded transition-colors"
                      title="Reset to default"
                    >
                      <RotateCcw size={12} />
                      Reset
                    </button>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCancel}
                    className="px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] rounded transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSave}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-[var(--text-primary)] text-[var(--bg-base)] rounded hover:opacity-90 transition-opacity"
                  >
                    <Check size={12} />
                    Save
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="relative">
                <pre className="w-full px-3 py-2 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-xs text-[var(--text-secondary)] font-mono whitespace-pre-wrap overflow-x-auto max-h-[200px] overflow-y-auto">
                  {displayValue.length > 500
                    ? `${displayValue.slice(0, 500)}...`
                    : displayValue}
                </pre>
              </div>
              <div className="flex items-center justify-end gap-2">
                <button
                  onClick={() => setIsEditing(true)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] rounded transition-colors"
                >
                  <Edit3 size={12} />
                  Edit
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Inline prompt editor for simpler use cases
 */
interface InlinePromptEditorProps {
  value: string;
  defaultValue: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
}

export function InlinePromptEditor({
  value,
  defaultValue,
  onChange,
  placeholder,
  rows = 3,
}: InlinePromptEditorProps) {
  const isModified = value.trim() !== "" && value.trim() !== defaultValue.trim();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isModified && (
            <span className="px-1.5 py-0.5 text-[10px] bg-amber-500/10 text-amber-400 rounded">
              modified
            </span>
          )}
        </div>
        {isModified && (
          <button
            onClick={() => onChange("")}
            className="flex items-center gap-1 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
          >
            <RotateCcw size={10} />
            Reset
          </button>
        )}
      </div>
      <textarea
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || defaultValue.slice(0, 100) + "..."}
        rows={rows}
        className="w-full px-2 py-1.5 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] transition-colors resize-none"
      />
    </div>
  );
}
