"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { HelpCircle } from "lucide-react";

// ============================================================================
// Glossary Terms
// ============================================================================

export const GLOSSARY: Record<string, { term: string; definition: string; example?: string }> = {
  knowledgeGraph: {
    term: "Knowledge Graph",
    definition: "A structured representation of information where entities (people, places, concepts) are connected by relationships. Enables intelligent search and reasoning.",
    example: '"John works at Acme" creates: John → WORKS_AT → Acme',
  },
  entity: {
    term: "Entity",
    definition: "A distinct object or concept in your data, such as a person, organization, place, or technical term.",
    example: "Person: 'John Smith', Organization: 'Acme Corp'",
  },
  relationship: {
    term: "Relationship",
    definition: "A connection between two entities that describes how they relate to each other.",
    example: "WORKS_AT, LOCATED_IN, CREATED_BY",
  },
  chunk: {
    term: "Chunk",
    definition: "A segment of your document. Documents are split into chunks for better processing and retrieval.",
    example: "A 2000-word document might be split into 4 chunks of ~500 words each",
  },
  embedding: {
    term: "Embedding",
    definition: "A numerical representation of text that captures its meaning. Similar texts have similar embeddings, enabling semantic search.",
  },
  semanticSearch: {
    term: "Semantic Search",
    definition: "Finding information based on meaning rather than exact keyword matches. Understands synonyms and context.",
    example: '"car repair" matches "automobile maintenance"',
  },
  triplet: {
    term: "Triplet",
    definition: "A fundamental unit of knowledge: Subject → Predicate → Object. Describes a single fact.",
    example: "Einstein → DEVELOPED → Relativity",
  },
  dataset: {
    term: "Dataset",
    definition: "A collection of related data. You can organize your content into multiple datasets for better management.",
  },
  pipeline: {
    term: "Pipeline",
    definition: "The automated process that analyzes your data, extracts entities and relationships, and builds your knowledge graph.",
  },
  cypher: {
    term: "Cypher",
    definition: "A query language for graph databases. Allows you to write precise queries to traverse and search your knowledge graph.",
    example: "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p, c",
  },
  incremental: {
    term: "Incremental Loading",
    definition: "Only processes new or changed content, skipping files that have already been processed. Saves time on repeated imports.",
  },
  chunker: {
    term: "Chunker",
    definition: "The algorithm that splits your documents into smaller pieces. Different chunkers work better for different types of content.",
  },
  episode: {
    term: "Episode",
    definition: "A unit of information in your knowledge graph, typically representing an event, fact, or piece of knowledge.",
  },
  facet: {
    term: "Facet",
    definition: "A specific aspect or attribute of an episode, providing additional detail and context.",
  },
};

// ============================================================================
// Types
// ============================================================================

export type GlossaryKey = keyof typeof GLOSSARY;

export interface GlossaryTooltipProps {
  termKey: GlossaryKey;
  className?: string;
  iconSize?: number;
  showIcon?: boolean;
  children?: React.ReactNode;
}

// ============================================================================
// Component
// ============================================================================

export function GlossaryTooltip({
  termKey,
  className,
  iconSize = 12,
  showIcon = true,
  children,
}: GlossaryTooltipProps) {
  const glossaryItem = GLOSSARY[termKey];

  if (!glossaryItem) {
    return <>{children}</>;
  }

  return (
    <span className={cn("inline-flex items-center gap-1 group relative", className)}>
      {children || (
        <span className="border-b border-dotted border-[var(--text-muted)] cursor-help">
          {glossaryItem.term}
        </span>
      )}
      {showIcon && (
        <HelpCircle
          size={iconSize}
          className="text-[var(--text-muted)] cursor-help opacity-60 group-hover:opacity-100 transition-opacity"
        />
      )}

      {/* Tooltip - use span elements to avoid hydration errors when nested in <p> */}
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-150 z-50 pointer-events-none block">
        <span className="block text-xs font-medium text-[var(--text-primary)] mb-1">
          {glossaryItem.term}
        </span>
        <span className="block text-[11px] text-[var(--text-secondary)] leading-relaxed">
          {glossaryItem.definition}
        </span>
        {glossaryItem.example && (
          <span className="block text-[10px] text-[var(--text-muted)] mt-2 pt-2 border-t border-[var(--border-subtle)]">
            <span className="font-medium">Example:</span> {glossaryItem.example}
          </span>
        )}

        {/* Arrow */}
        <span className="absolute top-full left-1/2 -translate-x-1/2 -mt-px block">
          <span className="border-8 border-transparent border-t-[var(--bg-elevated)] block" />
        </span>
      </span>
    </span>
  );
}

// ============================================================================
// Inline Term with Tooltip
// ============================================================================

export interface TermProps {
  termKey: GlossaryKey;
  children?: React.ReactNode;
}

export function Term({ termKey, children }: TermProps) {
  const glossaryItem = GLOSSARY[termKey];

  if (!glossaryItem) {
    return <>{children}</>;
  }

  return (
    <GlossaryTooltip termKey={termKey} showIcon={false}>
      {children || glossaryItem.term}
    </GlossaryTooltip>
  );
}

// ============================================================================
// Helper: Get term definition (for accessibility)
// ============================================================================

export function getTermDefinition(termKey: GlossaryKey): string | undefined {
  return GLOSSARY[termKey]?.definition;
}
