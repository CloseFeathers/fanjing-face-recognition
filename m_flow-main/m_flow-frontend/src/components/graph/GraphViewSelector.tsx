"use client";

import React from "react";
import { HierarchicalGraph } from "./HierarchicalGraph";

interface GraphViewSelectorProps {
  datasetId?: string;
}

/**
 * Knowledge Graph view component.
 */
export function GraphViewSelector({ datasetId }: GraphViewSelectorProps) {
  return (
    <div className="max-w-6xl mx-auto">
      <HierarchicalGraph datasetId={datasetId} />
    </div>
  );
}
