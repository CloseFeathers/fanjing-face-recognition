"use client";

import React from "react";
import { ChevronRight, Home } from "lucide-react";
import { NavigationState } from "@/types";

interface BreadcrumbNavProps {
  navigation: NavigationState;
  onNavigate: (layer: NavigationState["layer"], id?: string) => void;
}

/**
 * Breadcrumb navigation for hierarchical graph exploration.
 * Displays the current navigation path and allows jumping to any level.
 */
export function BreadcrumbNav({ navigation, onNavigate }: BreadcrumbNavProps) {
  const items: Array<{
    label: string;
    layer: NavigationState["layer"];
    id?: string;
    active: boolean;
  }> = [];

  // Always show Overview as root
  items.push({
    label: "Overview",
    layer: "overview",
    active: navigation.layer === "overview",
  });

  // Episode level
  if (navigation.episodeId) {
    items.push({
      label: navigation.episodeName || `Episode`,
      layer: "episode",
      id: navigation.episodeId,
      active: navigation.layer === "episode",
    });
  }

  // Facet level
  if (navigation.facetId) {
    items.push({
      label: navigation.facetName || `Facet`,
      layer: "facet",
      id: navigation.facetId,
      active: navigation.layer === "facet",
    });
  }

  // Entity level
  if (navigation.entityId) {
    items.push({
      label: navigation.entityName || `Entity`,
      layer: "entity",
      id: navigation.entityId,
      active: navigation.layer === "entity",
    });
  }

  return (
    <nav className="flex items-center gap-1 text-[13px]">
      {items.map((item, index) => (
        <React.Fragment key={`${item.layer}-${item.id || "root"}`}>
          {index > 0 && (
            <ChevronRight
              size={14}
              className="text-[var(--text-muted)] flex-shrink-0"
            />
          )}
          <button
            onClick={() => onNavigate(item.layer, item.id)}
            disabled={item.active}
            className={`
              flex items-center gap-1.5 px-2 py-1 rounded-md transition-colors
              ${
                item.active
                  ? "text-[var(--text-primary)] bg-[var(--bg-elevated)] cursor-default"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]"
              }
            `}
          >
            {index === 0 && <Home size={12} />}
            <span className="truncate max-w-[150px]">{item.label}</span>
          </button>
        </React.Fragment>
      ))}
    </nav>
  );
}
