"use client";

import React, { useState, useRef, KeyboardEvent } from "react";
import { cn } from "@/lib/utils";
import { X, Plus } from "lucide-react";

interface TagInputProps {
  tags: string[];
  onTagsChange: (tags: string[]) => void;
  placeholder?: string;
  maxTags?: number;
  disabled?: boolean;
  className?: string;
}

export function TagInput({
  tags,
  onTagsChange,
  placeholder = "Add tag...",
  maxTags = 10,
  disabled = false,
  className,
}: TagInputProps) {
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const addTag = (tag: string) => {
    const trimmed = tag.trim();
    if (!trimmed) return;
    if (tags.includes(trimmed)) return;
    if (tags.length >= maxTags) return;

    onTagsChange([...tags, trimmed]);
    setInputValue("");
  };

  const removeTag = (tagToRemove: string) => {
    onTagsChange(tags.filter((tag) => tag !== tagToRemove));
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addTag(inputValue);
    } else if (e.key === "Backspace" && !inputValue && tags.length > 0) {
      removeTag(tags[tags.length - 1]);
    }
  };

  const handleContainerClick = () => {
    if (!disabled) {
      inputRef.current?.focus();
    }
  };

  return (
    <div
      onClick={handleContainerClick}
      className={cn(
        "flex flex-wrap gap-1.5 p-2 min-h-[38px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded cursor-text transition-colors",
        "focus-within:border-[var(--border-default)]",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {tags.map((tag) => (
        <span
          key={tag}
          className="flex items-center gap-1 px-2 py-0.5 text-xs bg-[var(--bg-surface)] text-[var(--text-secondary)] border border-[var(--border-subtle)] rounded"
        >
          {tag}
          {!disabled && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                removeTag(tag);
              }}
              className="text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
            >
              <X size={10} />
            </button>
          )}
        </span>
      ))}
      {tags.length < maxTags && !disabled && (
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => addTag(inputValue)}
          placeholder={tags.length === 0 ? placeholder : ""}
          disabled={disabled}
          className="flex-1 min-w-[80px] bg-transparent text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none"
        />
      )}
    </div>
  );
}

/**
 * Compact version for inline use
 */
interface CompactTagInputProps {
  tags: string[];
  onAdd: (tag: string) => void;
  onRemove: (tag: string) => void;
  placeholder?: string;
  maxTags?: number;
  disabled?: boolean;
}

export function CompactTagInput({
  tags,
  onAdd,
  onRemove,
  placeholder = "Add...",
  maxTags = 5,
  disabled = false,
}: CompactTagInputProps) {
  const [inputValue, setInputValue] = useState("");
  const [showInput, setShowInput] = useState(false);

  const handleAdd = () => {
    const trimmed = inputValue.trim();
    if (trimmed && !tags.includes(trimmed) && tags.length < maxTags) {
      onAdd(trimmed);
      setInputValue("");
      setShowInput(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAdd();
    } else if (e.key === "Escape") {
      setInputValue("");
      setShowInput(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {tags.map((tag) => (
        <span
          key={tag}
          className="flex items-center gap-1 px-1.5 py-0.5 text-[10px] bg-[var(--bg-surface)] text-[var(--text-secondary)] border border-[var(--border-subtle)] rounded"
        >
          {tag}
          {!disabled && (
            <button
              onClick={() => onRemove(tag)}
              className="text-[var(--text-muted)] hover:text-[var(--error)] transition-colors"
            >
              <X size={8} />
            </button>
          )}
        </span>
      ))}
      {tags.length < maxTags && !disabled && (
        <>
          {showInput ? (
            <input
              autoFocus
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={handleAdd}
              placeholder={placeholder}
              className="w-20 px-1.5 py-0.5 text-[10px] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)]"
            />
          ) : (
            <button
              onClick={() => setShowInput(true)}
              className="flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] border border-dashed border-[var(--border-subtle)] rounded transition-colors"
            >
              <Plus size={10} />
              Add
            </button>
          )}
        </>
      )}
    </div>
  );
}
