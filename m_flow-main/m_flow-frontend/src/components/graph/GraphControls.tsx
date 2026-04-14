"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RefreshCw,
  Download,
  Share2,
  Circle,
  GitBranch,
  Network,
} from "lucide-react";

export type LayoutType = "force" | "hierarchical" | "circular";

interface GraphControlsProps {
  layout: LayoutType;
  onLayoutChange: (layout: LayoutType) => void;
  linkDistance: number;
  onLinkDistanceChange: (distance: number) => void;
  chargeStrength: number;
  onChargeStrengthChange: (strength: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitView: () => void;
  onRefresh: () => void;
  onExport: (format: "png" | "svg" | "json") => void;
  nodeCount: number;
  edgeCount: number;
}

const layouts = [
  { value: "force", label: "Force", icon: <Share2 className="h-4 w-4" /> },
  { value: "hierarchical", label: "Hierarchy", icon: <GitBranch className="h-4 w-4" /> },
  { value: "circular", label: "Circular", icon: <Circle className="h-4 w-4" /> },
];

export function GraphControls({
  layout,
  onLayoutChange,
  linkDistance,
  onLinkDistanceChange,
  chargeStrength,
  onChargeStrengthChange,
  onZoomIn,
  onZoomOut,
  onFitView,
  onRefresh,
  onExport,
  nodeCount,
  edgeCount,
}: GraphControlsProps) {
  return (
    <div className="absolute top-4 left-4 flex flex-col gap-3 z-10">
      {/* Layout Selection */}
      <div className="p-3 rounded-lg bg-zinc-900/90 border border-zinc-800 space-y-3">
        <Label className="text-xs text-zinc-500">Layout Algorithm</Label>
        <Select value={layout} onValueChange={(v) => onLayoutChange(v as LayoutType)}>
          <SelectTrigger className="w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {layouts.map((l) => (
              <SelectItem key={l.value} value={l.value}>
                <span className="flex items-center gap-2">
                  {l.icon}
                  {l.label}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Force layout parameters (force layout only) */}
        {layout === "force" && (
          <>
            <div className="space-y-1">
              <div className="flex justify-between">
                <Label className="text-xs text-zinc-500">Link Distance</Label>
                <span className="text-xs text-zinc-400">{linkDistance}</span>
              </div>
              <Slider
                value={[linkDistance]}
                onValueChange={([v]) => onLinkDistanceChange(v)}
                min={50}
                max={300}
                step={10}
              />
            </div>
            <div className="space-y-1">
              <div className="flex justify-between">
                <Label className="text-xs text-zinc-500">Charge Strength</Label>
                <span className="text-xs text-zinc-400">{Math.abs(chargeStrength)}</span>
              </div>
              <Slider
                value={[Math.abs(chargeStrength)]}
                onValueChange={([v]) => onChargeStrengthChange(-v)}
                min={50}
                max={500}
                step={10}
              />
            </div>
          </>
        )}
      </div>

      {/* View Controls */}
      <div className="flex gap-1 p-1 rounded-lg bg-zinc-900/90 border border-zinc-800">
        <Button variant="ghost" size="icon" onClick={onZoomIn} title="Zoom In">
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={onZoomOut} title="Zoom Out">
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={onFitView} title="Fit View">
          <Maximize2 className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={onRefresh} title="Refresh Data">
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      {/* Export */}
      <div className="p-2 rounded-lg bg-zinc-900/90 border border-zinc-800">
        <Label className="text-xs text-zinc-500 mb-2 block">Export</Label>
        <div className="flex gap-1">
          <Button variant="outline" size="sm" onClick={() => onExport("png")}>
            PNG
          </Button>
          <Button variant="outline" size="sm" onClick={() => onExport("svg")}>
            SVG
          </Button>
          <Button variant="outline" size="sm" onClick={() => onExport("json")}>
            JSON
          </Button>
        </div>
      </div>

      {/* Statistics */}
      <div className="p-2 rounded-lg bg-zinc-900/90 border border-zinc-800 text-xs text-zinc-500">
        <div className="flex items-center gap-2">
          <Network className="h-3 w-3" />
          <span>{nodeCount} nodes · {edgeCount} edges</span>
        </div>
      </div>
    </div>
  );
}
