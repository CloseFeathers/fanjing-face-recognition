"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { GraphNode } from "@/types";
import {
  Eye,
  EyeOff,
  Expand,
  Pin,
  PinOff,
  Trash2,
  Copy,
  Search,
  ExternalLink,
} from "lucide-react";

interface NodeContextMenuProps {
  node: GraphNode;
  position: { x: number; y: number };
  isPinned: boolean;
  isExpanded: boolean;
  onClose: () => void;
  onPin: () => void;
  onUnpin: () => void;
  onExpand: () => void;
  onCollapse: () => void;
  onHide: () => void;
  onDelete: () => void;
  onCopyId: () => void;
  onSearch: () => void;
  onViewDetails: () => void;
}

export function NodeContextMenu({
  node,
  position,
  isPinned,
  isExpanded,
  onClose,
  onPin,
  onUnpin,
  onExpand,
  onCollapse,
  onHide,
  onDelete,
  onCopyId,
  onSearch,
  onViewDetails,
}: NodeContextMenuProps) {
  const menuItems = [
    {
      label: "View Details",
      icon: <ExternalLink className="h-4 w-4" />,
      onClick: onViewDetails,
    },
    {
      label: "Search This",
      icon: <Search className="h-4 w-4" />,
      onClick: onSearch,
    },
    { divider: true },
    isPinned
      ? {
          label: "Unpin",
          icon: <PinOff className="h-4 w-4" />,
          onClick: onUnpin,
        }
      : {
          label: "Pin Position",
          icon: <Pin className="h-4 w-4" />,
          onClick: onPin,
        },
    isExpanded
      ? {
          label: "Collapse Neighbors",
          icon: <EyeOff className="h-4 w-4" />,
          onClick: onCollapse,
        }
      : {
          label: "Expand Neighbors",
          icon: <Expand className="h-4 w-4" />,
          onClick: onExpand,
        },
    { divider: true },
    {
      label: "Copy ID",
      icon: <Copy className="h-4 w-4" />,
      onClick: onCopyId,
    },
    {
      label: "Hide Node",
      icon: <EyeOff className="h-4 w-4" />,
      onClick: onHide,
    },
    {
      label: "Delete Node",
      icon: <Trash2 className="h-4 w-4" />,
      onClick: onDelete,
      danger: true,
    },
  ];

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 z-40" onClick={onClose} />

      {/* Menu */}
      <div
        className="fixed z-50 min-w-48 py-1 rounded-lg bg-zinc-900 border border-zinc-700 shadow-xl"
        style={{
          left: position.x,
          top: position.y,
          transform: "translate(-50%, 0)",
        }}
      >
        {/* Node Info Header */}
        <div className="px-3 py-2 border-b border-zinc-800">
          <p className="text-sm font-medium text-zinc-200 truncate max-w-48">
            {node.name}
          </p>
          <p className="text-xs text-zinc-500">{node.type}</p>
        </div>

        {/* Menu Items */}
        <div className="py-1">
          {menuItems.map((item, index) =>
            "divider" in item ? (
              <div key={index} className="h-px bg-zinc-800 my-1" />
            ) : (
              <button
                key={index}
                onClick={() => {
                  item.onClick();
                  onClose();
                }}
                className={cn(
                  "w-full flex items-center gap-2 px-3 py-1.5 text-sm transition-colors",
                  "danger" in item && item.danger
                    ? "text-red-400 hover:bg-red-500/10"
                    : "text-zinc-300 hover:bg-zinc-800"
                )}
              >
                {item.icon}
                {item.label}
              </button>
            )
          )}
        </div>
      </div>
    </>
  );
}
