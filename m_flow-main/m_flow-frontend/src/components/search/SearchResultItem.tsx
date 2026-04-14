"use client";

import React, { useMemo } from "react";
import { SearchResultItem as SearchResultItemType, RecallMode } from "@/types";
import { truncate } from "@/lib/utils";
import { ArrowRight, Clock, Database } from "lucide-react";

interface SearchResultItemProps {
  item: SearchResultItemType;
  recallMode: RecallMode;
  index: number;
}

/**
 * Clean internal markers from content.
 * Removes: __node_content_start__, __node_content_end__, Node: [xxx], etc.
 */
function cleanContent(content: string): string {
  if (!content) return "";
  
  return content
    // Remove __node_content_start__ and __node_content_end__
    .replace(/\\n__node_content_start__\\n?/g, "\n")
    .replace(/\\n__node_content_end__\\n?/g, "\n")
    .replace(/__node_content_start__/g, "")
    .replace(/__node_content_end__/g, "")
    // Remove Node: [xxx] headers
    .replace(/\\nNode:\s*\[[^\]]+\]\\n?/g, "\n")
    .replace(/Node:\s*\[[^\]]+\]/g, "")
    // Remove ID: xxx lines
    .replace(/\\nID:\s*[^\n]+/g, "")
    .replace(/^ID:\s*[^\n]+\n?/gm, "")
    // Remove Connections: ... blocks
    .replace(/\\nConnections:\\n\[.*?\]/gs, "")
    .replace(/Connections:\s*\n?\[.*?\]/gs, "")
    // Clean up escaped newlines
    .replace(/\\n/g, "\n")
    // Clean up multiple newlines
    .replace(/\n{3,}/g, "\n\n")
    // Trim whitespace
    .trim();
}

/** Parsed episode from summary mode */
interface ParsedEpisode {
  time?: string;
  name?: string;
  sections: Array<{ title: string; content: string }>;
  rawContent?: string;
}

/**
 * Parse episodic summary content into structured episodes.
 * Backend returns format: "[time] [name]\n【section】content..." separated by "\n\n---\n\n"
 */
function parseEpisodicSummary(content: string): ParsedEpisode[] {
  const cleaned = cleanContent(content);
  if (!cleaned) return [];
  
  // Split by episode separator
  const episodeBlocks = cleaned.split(/\n\n---\n\n|\n---\n/);
  const episodes: ParsedEpisode[] = [];
  
  for (const block of episodeBlocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    
    const episode: ParsedEpisode = { sections: [] };
    
    // Try to extract header: [time] [name] or just [name]
    // Pattern: [xxx] [yyy] at the start, followed by newline
    const headerMatch = trimmed.match(/^(\[[^\]]+\])(?:\s*(\[[^\]]+\]))?\s*\n/);
    let bodyStart = 0;
    
    if (headerMatch) {
      bodyStart = headerMatch[0].length;
      const bracket1 = headerMatch[1].slice(1, -1); // Remove [ ]
      const bracket2 = headerMatch[2]?.slice(1, -1);
      
      if (bracket2) {
        // Two brackets: first is time, second is name
        episode.time = bracket1;
        episode.name = bracket2;
      } else {
        // One bracket: could be time or name
        // If contains time-like patterns (am/pm, :, dates), treat as time
        if (/\d{1,2}:\d{2}|am|pm|月|日|\d{4}/i.test(bracket1)) {
          episode.time = bracket1;
        } else {
          episode.name = bracket1;
        }
      }
    }
    
    const body = trimmed.slice(bodyStart).trim();
    
    // Parse 【section】content format
    const sectionRegex = /【([^】]+)】([^【]*)/g;
    let match;
    
    while ((match = sectionRegex.exec(body)) !== null) {
      episode.sections.push({
        title: match[1].trim(),
        content: match[2].trim(),
      });
    }
    
    // If no sections found, use raw content
    if (episode.sections.length === 0 && body) {
      episode.rawContent = body;
    }
    
    episodes.push(episode);
  }
  
  return episodes;
}

interface ParsedContent {
  summary: string;
  sections: Array<{ title: string; content: string }>;
}

/**
 * Parse episodic content into structured sections (legacy format).
 */
function parseEpisodicContent(content: string): ParsedContent {
  const cleaned = cleanContent(content);
  const sections: Array<{ title: string; content: string }> = [];
  
  // Try to parse 【title】content format
  const sectionRegex = /【([^】]+)】([^【]*)/g;
  let match;
  let foundSections = false;
  
  while ((match = sectionRegex.exec(cleaned)) !== null) {
    foundSections = true;
    sections.push({
      title: match[1].trim(),
      content: match[2].trim(),
    });
  }
  
  if (foundSections && sections.length > 0) {
    const firstSectionIndex = cleaned.indexOf("【");
    const summaryPart = firstSectionIndex > 0 ? cleaned.slice(0, firstSectionIndex).trim() : "";
    return {
      summary: summaryPart || sections.map(s => s.title).join(" | "),
      sections,
    };
  }
  
  return {
    summary: cleaned,
    sections: [],
  };
}

export function SearchResultItem({ item, recallMode, index }: SearchResultItemProps) {
  // Memoize content processing at component level
  const cleanedContent = useMemo(() => cleanContent(item.content || ""), [item.content]);
  const parsedEpisodes = useMemo(() => parseEpisodicSummary(item.content || ""), [item.content]);
  const parsedEpisodicContent = useMemo(() => parseEpisodicContent(item.content || ""), [item.content]);

  const renderTripletResult = () => (
    <div className="space-y-3">
      {item.subject && item.predicate && item.object && (
        <div className="flex items-center gap-2 flex-wrap text-[13px]">
          <span className="px-2 py-1 bg-[var(--bg-elevated)] text-[var(--text-primary)] rounded">
            {truncate(item.subject, 30)}
          </span>
          <ArrowRight size={12} className="text-[var(--text-muted)]" />
          <span className="px-2 py-1 bg-[var(--bg-elevated)] text-[var(--text-secondary)] rounded">
            {item.predicate}
          </span>
          <ArrowRight size={12} className="text-[var(--text-muted)]" />
          <span className="px-2 py-1 bg-[var(--bg-elevated)] text-[var(--text-primary)] rounded">
            {truncate(item.object, 30)}
          </span>
        </div>
      )}
      <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
        {cleanedContent}
      </p>
    </div>
  );

  /** Render a single episode block */
  const renderEpisodeBlock = (episode: ParsedEpisode, episodeIndex: number) => (
    <div key={episodeIndex} className="space-y-1.5">
      {/* Episode header: time and name */}
      {(episode.time || episode.name) && (
        <div className="flex items-center gap-2 text-[11px]">
          {episode.time && (
            <span className="flex items-center gap-1 text-[var(--text-muted)]">
              <Clock size={10} />
              {episode.time}
            </span>
          )}
          {episode.name && (
            <span className="font-medium text-[var(--text-secondary)]">
              {episode.name}
            </span>
          )}
        </div>
      )}
      
      {/* Episode sections */}
      {episode.sections.length > 0 ? (
        <div className="space-y-2 pl-2 border-l-2 border-[var(--border-subtle)]">
          {episode.sections.map((section, i) => (
            <div key={i}>
              <h4 className="text-[11px] font-medium text-[var(--text-secondary)] mb-0.5">
                {section.title}
              </h4>
              <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">
                {section.content}
              </p>
            </div>
          ))}
        </div>
      ) : episode.rawContent ? (
        <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed pl-2 border-l-2 border-[var(--border-subtle)]">
          {episode.rawContent}
        </p>
      ) : null}
    </div>
  );

  const renderEpisodicResult = () => {
    // If we have parsed episodes (from summary mode), render them
    if (parsedEpisodes.length > 0) {
      return (
        <div className="space-y-4">
          {parsedEpisodes.map((episode, i) => renderEpisodeBlock(episode, i))}
        </div>
      );
    }
    
    // Fallback: try legacy parsing
    if (parsedEpisodicContent.sections.length > 0) {
      return (
        <div className="space-y-2">
          {/* Metadata row */}
          <div className="flex items-center gap-3 text-[11px] text-[var(--text-muted)]">
            {item.timestamp && (
              <span className="flex items-center gap-1">
                <Clock size={10} />
                {new Date(item.timestamp).toLocaleString()}
              </span>
            )}
          </div>
          <div className="space-y-2 pl-2 border-l-2 border-[var(--border-subtle)]">
            {parsedEpisodicContent.sections.map((section, i) => (
              <div key={i}>
                <h4 className="text-[11px] font-medium text-[var(--text-secondary)] mb-0.5">
                  {section.title}
                </h4>
                <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">
                  {section.content}
                </p>
              </div>
            ))}
          </div>
        </div>
      );
    }
    
    // Final fallback: show cleaned content
    return (
      <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
        {cleanedContent}
      </p>
    );
  };

  const renderProceduralResult = () => (
    <div className="space-y-2">
      {item.node_type && (
        <span className="inline-block px-2 py-0.5 bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded text-[11px]">
          {item.node_type}
        </span>
      )}
      <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
        {cleanedContent}
      </p>
    </div>
  );

  const renderCypherResult = () => (
    <div className="space-y-2">
      <pre className="p-3 bg-[var(--bg-elevated)] rounded text-[11px] text-[var(--text-secondary)] overflow-x-auto">
        {JSON.stringify(item.metadata || item, null, 2)}
      </pre>
      {item.content && (
        <p className="text-[13px] text-[var(--text-secondary)] whitespace-pre-wrap">
          {cleanedContent}
        </p>
      )}
    </div>
  );

  const renderLexicalResult = () => {
    // ContentFragment JSON → extract readable text
    let text = cleanedContent;
    let chunkSize: number | undefined;
    try {
      const parsed = JSON.parse(item.content);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        text = parsed.text || parsed.content || parsed.summary || cleanedContent;
        chunkSize = parsed.chunk_size;
      }
    } catch {
      // not JSON
    }
    return (
      <div className="space-y-2">
        {chunkSize !== undefined && (
          <span className="inline-block px-2 py-0.5 bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded text-[10px]">
            {chunkSize} tokens
          </span>
        )}
        <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
          {text}
        </p>
      </div>
    );
  };

  const renderDefaultResult = () => (
    <div className="space-y-2">
      {item.source_dataset && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded text-[11px]">
          <Database size={10} />
          {item.source_dataset}
        </span>
      )}
      <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap">
        {cleanedContent}
      </p>
    </div>
  );

  const renderContent = () => {
    switch (recallMode) {
      case "TRIPLET_COMPLETION":
        return renderTripletResult();
      case "EPISODIC":
        return renderEpisodicResult();
      case "PROCEDURAL":
        return renderProceduralResult();
      case "CYPHER":
        return renderCypherResult();
      case "CHUNKS_LEXICAL":
        return renderLexicalResult();
      default:
        return renderDefaultResult();
    }
  };

  return (
    <div className="p-4 border border-[var(--border-subtle)] rounded-lg hover:border-[var(--border-default)] transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] text-[var(--text-muted)]">#{index + 1}</span>
        <div className="flex items-center gap-2">
          {/* Score bar */}
          <div className="w-12 h-1 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
            <div
              className="h-full bg-[var(--text-primary)] rounded-full"
              style={{ width: `${Math.min(item.score * 100, 100)}%` }}
            />
          </div>
          <span className="text-[11px] text-[var(--text-muted)] w-12 text-right">
            Score: {(item.score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {renderContent()}

      {/* Dataset badge */}
      {item.source_dataset && recallMode !== "CHUNKS_LEXICAL" && (
        <div className="mt-3 pt-2 border-t border-[var(--border-subtle)]">
          <span className="inline-flex items-center gap-1 text-[10px] text-[var(--text-muted)]">
            <Database size={10} />
            {item.source_dataset}
          </span>
        </div>
      )}
    </div>
  );
}
