"use client";

/**
 * FAQSection Component
 *
 * Displays frequently asked questions in an accordion format.
 * Supports search and category filtering.
 *
 * Features:
 * - Accordion expand/collapse
 * - Search functionality
 * - Category grouping
 * - Keyboard navigation
 *
 * @example
 * <FAQSection faqs={faqData} searchQuery="database" />
 */

import React, { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, HelpCircle, Search } from "lucide-react";
import type { FAQ } from "@/content/troubleshooting";

// ============================================================================
// Types
// ============================================================================

export interface FAQSectionProps {
  /** FAQ data */
  faqs: FAQ[];
  /** Search query for filtering */
  searchQuery?: string;
  /** Whether to show search input */
  showSearch?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export interface FAQItemProps {
  faq: FAQ;
  isExpanded: boolean;
  onToggle: () => void;
  searchQuery?: string;
}

// ============================================================================
// FAQ Item Component
// ============================================================================

export function FAQItem({
  faq,
  isExpanded,
  onToggle,
  searchQuery,
}: FAQItemProps) {
  // Highlight matching text
  const highlightText = (text: string) => {
    if (!searchQuery) return text;

    const regex = new RegExp(`(${searchQuery})`, "gi");
    const parts = text.split(regex);

    return parts.map((part, i) =>
      regex.test(part) ? (
        <mark key={i} className="bg-amber-500/30 text-amber-200 rounded px-0.5">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <div
      className={cn(
        "border-b border-zinc-800 last:border-b-0",
        isExpanded && "bg-zinc-900/30"
      )}
    >
      {/* Question */}
      <button
        onClick={onToggle}
        className={cn(
          "w-full flex items-start gap-3 p-4 text-left",
          "hover:bg-zinc-900/30 transition-colors"
        )}
        aria-expanded={isExpanded}
      >
        <HelpCircle
          size={16}
          className={cn(
            "shrink-0 mt-0.5 transition-colors",
            isExpanded ? "text-zinc-200" : "text-zinc-500"
          )}
        />
        <div className="flex-1 min-w-0">
          <h4
            className={cn(
              "text-[13px] font-medium transition-colors",
              isExpanded ? "text-zinc-100" : "text-zinc-300"
            )}
          >
            {highlightText(faq.question)}
          </h4>
        </div>
        <ChevronDown
          size={16}
          className={cn(
            "shrink-0 text-zinc-500 transition-transform",
            isExpanded && "rotate-180"
          )}
        />
      </button>

      {/* Answer */}
      {isExpanded && (
        <div className="px-4 pb-4 pl-11">
          <p className="text-[12px] text-zinc-400 leading-relaxed">
            {highlightText(faq.answer)}
          </p>
          <span className="inline-block mt-2 px-2 py-0.5 rounded text-[9px] bg-zinc-800 text-zinc-500">
            {faq.category}
          </span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function FAQSection({
  faqs,
  searchQuery: externalSearchQuery,
  showSearch = true,
  className,
}: FAQSectionProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [internalSearch, setInternalSearch] = useState("");

  const searchQuery = externalSearchQuery ?? internalSearch;

  // Filter FAQs by search query
  const filteredFaqs = useMemo(() => {
    if (!searchQuery) return faqs;

    const lowerQuery = searchQuery.toLowerCase();
    return faqs.filter(
      (faq) =>
        faq.question.toLowerCase().includes(lowerQuery) ||
        faq.answer.toLowerCase().includes(lowerQuery) ||
        faq.category.toLowerCase().includes(lowerQuery)
    );
  }, [faqs, searchQuery]);

  // Group by category
  const groupedFaqs = useMemo(() => {
    const groups: Record<string, FAQ[]> = {};
    filteredFaqs.forEach((faq) => {
      if (!groups[faq.category]) {
        groups[faq.category] = [];
      }
      groups[faq.category].push(faq);
    });
    return groups;
  }, [filteredFaqs]);

  const handleToggle = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Search */}
      {showSearch && !externalSearchQuery && (
        <div className="relative">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500"
          />
          <input
            type="text"
            value={internalSearch}
            onChange={(e) => setInternalSearch(e.target.value)}
            placeholder="Search FAQs..."
            className={cn(
              "w-full pl-9 pr-3 py-2 rounded-lg text-[12px]",
              "bg-zinc-900 border border-zinc-800",
              "text-zinc-200 placeholder-zinc-500",
              "focus:outline-none focus:ring-1 focus:ring-zinc-600"
            )}
          />
        </div>
      )}

      {/* FAQ List */}
      {Object.keys(groupedFaqs).length > 0 ? (
        <div className="space-y-4">
          {Object.entries(groupedFaqs).map(([category, categoryFaqs]) => (
            <div
              key={category}
              className="border border-zinc-800 rounded-xl overflow-hidden"
            >
              {/* Category Header */}
              <div className="px-4 py-2 bg-zinc-900/50 border-b border-zinc-800">
                <h5 className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {category}
                </h5>
              </div>

              {/* FAQ Items */}
              <div>
                {categoryFaqs.map((faq) => (
                  <FAQItem
                    key={faq.id}
                    faq={faq}
                    isExpanded={expandedId === faq.id}
                    onToggle={() => handleToggle(faq.id)}
                    searchQuery={searchQuery}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="py-8 text-center">
          <HelpCircle size={24} className="text-zinc-600 mx-auto mb-2" />
          <p className="text-[13px] text-zinc-500">
            No FAQs found matching "{searchQuery}"
          </p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Display Names
// ============================================================================

FAQItem.displayName = "FAQItem";
FAQSection.displayName = "FAQSection";

// ============================================================================
// Default Export
// ============================================================================

export default FAQSection;
