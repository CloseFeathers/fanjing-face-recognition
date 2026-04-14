"use client";

import React, { useState, useMemo } from "react";
import { cn, formatTimeAgo } from "@/lib/utils";
import { useDatasetsWithCounts, useUsers, useDataItems } from "@/hooks/use-api";
import { CreateDatasetModal } from "./CreateDatasetModal";
import { Loader2, ChevronDown, ChevronRight, CheckCircle2, Clock, AlertCircle } from "lucide-react";

// ============================================================================
// Constants
// ============================================================================

const ROLE_STYLES = {
  admin: "bg-red-500/10 text-red-400",
  user: "bg-blue-500/10 text-blue-400",
  viewer: "bg-gray-500/10 text-gray-400",
};

// ============================================================================
// Main Page
// ============================================================================

function getMemorizeStatus(ps: Record<string, Record<string, string>> | null | undefined): "completed" | "pending" | "not_processed" {
  if (!ps) return "not_processed";
  const mem = ps["memorize_pipeline"];
  if (!mem) return "not_processed";
  const vals = Object.values(mem);
  if (vals.some((v) => typeof v === "string" && v.includes("COMPLETED"))) return "completed";
  if (vals.length > 0) return "pending";
  return "not_processed";
}

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function DataItemRow({ item, datasetId }: { item: import("@/types").DataItemInfo; datasetId: string }) {
  const [showPreview, setShowPreview] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const status = getMemorizeStatus(item.pipelineStatus);

  const handleTogglePreview = async () => {
    if (showPreview) {
      setShowPreview(false);
      return;
    }
    if (preview !== null) {
      setShowPreview(true);
      return;
    }
    setLoadingPreview(true);
    try {
      const { apiClient } = await import("@/lib/api/client");
      const blob = await apiClient.getDataRaw(datasetId, item.id);
      const text = await blob.text();
      setPreview(text.slice(0, 2000));
    } catch {
      setPreview("[Failed to load content]");
    } finally {
      setLoadingPreview(false);
      setShowPreview(true);
    }
  };

  return (
    <>
      <tr
        className="border-b border-[var(--border-subtle)] last:border-0 hover:bg-[var(--bg-hover)] cursor-pointer"
        onClick={handleTogglePreview}
      >
        <td className="py-1.5 px-4 text-[11px] text-[var(--text-secondary)]">
          <div className="flex items-center gap-1 max-w-[200px]">
            {showPreview ? <ChevronDown size={10} className="text-[var(--text-muted)] flex-shrink-0" /> : <ChevronRight size={10} className="text-[var(--text-muted)] flex-shrink-0" />}
            <span className="truncate">{item.name}</span>
          </div>
        </td>
        <td className="py-1.5 px-3 text-[11px] text-[var(--text-muted)]">
          {formatBytes(item.dataSize)}
        </td>
        <td className="py-1.5 px-3 text-[11px] text-[var(--text-muted)]">
          {item.createdAt ? formatTimeAgo(item.createdAt) : "-"}
        </td>
        <td className="py-1.5 px-3">
          {status === "completed" && (
            <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400">
              <CheckCircle2 size={10} /> Built
            </span>
          )}
          {status === "pending" && (
            <span className="inline-flex items-center gap-1 text-[10px] text-amber-400">
              <Clock size={10} /> Pending
            </span>
          )}
          {status === "not_processed" && (
            <span className="inline-flex items-center gap-1 text-[10px] text-[var(--text-muted)]">
              <AlertCircle size={10} /> Not built
            </span>
          )}
        </td>
      </tr>
      {showPreview && (
        <tr>
          <td colSpan={4} className="p-0">
            <div className="px-6 py-2 bg-[var(--bg-base)] border-b border-[var(--border-subtle)]">
              {loadingPreview ? (
                <div className="flex items-center gap-2 text-[10px] text-[var(--text-muted)]">
                  <Loader2 size={10} className="animate-spin" /> Loading...
                </div>
              ) : (
                <pre className="text-[10px] text-[var(--text-muted)] font-mono whitespace-pre-wrap max-h-48 overflow-auto leading-relaxed">
                  {preview}
                  {preview && preview.length >= 2000 && (
                    <span className="text-[var(--text-muted)] opacity-50">{"\n\n"}... (truncated at 2000 chars)</span>
                  )}
                </pre>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function DatasetRow({ dataset, userMap }: { dataset: { id: string; name: string; ownerId?: string; dataCount?: number; createdAt?: string }; userMap: Map<string, string> }) {
  const [expanded, setExpanded] = useState(false);
  const { data: items, isLoading } = useDataItems(expanded ? dataset.id : null);

  return (
    <>
      <tr
        className="border-b border-[var(--border-subtle)] last:border-0 hover:bg-[var(--bg-hover)] cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="py-2.5 px-3 text-xs text-[var(--text-primary)]">
          <div className="flex items-center gap-1.5">
            {expanded ? <ChevronDown size={12} className="text-[var(--text-muted)]" /> : <ChevronRight size={12} className="text-[var(--text-muted)]" />}
            {dataset.name}
          </div>
        </td>
        <td className="py-2.5 px-3 text-xs text-[var(--text-secondary)]">
          {dataset.ownerId ? (userMap.get(dataset.ownerId) || dataset.ownerId.slice(0, 8)) : "-"}
        </td>
        <td className="py-2.5 px-3 text-xs text-[var(--text-secondary)]">
          {dataset.dataCount ?? 0}
        </td>
        <td className="py-2.5 px-3 text-xs text-[var(--text-secondary)]">
          {dataset.createdAt ? formatTimeAgo(dataset.createdAt) : "-"}
        </td>
        <td className="py-2.5 px-3" />
      </tr>
      {expanded && (
        <tr>
          <td colSpan={5} className="p-0">
            <div className="bg-[var(--bg-elevated)] border-b border-[var(--border-subtle)]">
              {isLoading ? (
                <div className="flex items-center gap-2 p-3 text-xs text-[var(--text-muted)]">
                  <Loader2 size={12} className="animate-spin" /> Loading items...
                </div>
              ) : !items || items.length === 0 ? (
                <div className="p-3 text-xs text-[var(--text-muted)]">No data items</div>
              ) : (
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-[var(--border-subtle)]">
                      <th className="text-left py-1.5 px-4 text-[9px] text-[var(--text-muted)] font-medium uppercase tracking-wider">Name</th>
                      <th className="text-left py-1.5 px-3 text-[9px] text-[var(--text-muted)] font-medium uppercase tracking-wider">Size</th>
                      <th className="text-left py-1.5 px-3 text-[9px] text-[var(--text-muted)] font-medium uppercase tracking-wider">Added</th>
                      <th className="text-left py-1.5 px-3 text-[9px] text-[var(--text-muted)] font-medium uppercase tracking-wider">Build Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item) => (
                      <DataItemRow key={item.id} item={item} datasetId={dataset.id} />
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export function UserManagementPage() {
  const [activeTab, setActiveTab] = useState<"users" | "datasets">("users");
  const [searchQuery, setSearchQuery] = useState("");
  const [showCreateDatasetModal, setShowCreateDatasetModal] = useState(false);

  // Data hooks
  const { data: datasets, isLoading: datasetsLoading } = useDatasetsWithCounts();
  const { data: usersData, isLoading: usersLoading } = useUsers();

  // Create user lookup map for displaying owner names
  const userMap = useMemo(() => {
    const map = new Map<string, string>();
    (usersData ?? []).forEach(u => {
      map.set(u.id, u.email.split("@")[0]);
    });
    return map;
  }, [usersData]);

  // Filtered data
  const filteredUsers = (usersData ?? []).filter((u) =>
    u.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredDatasets = (datasets ?? []).filter((d) =>
    d.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalDataItems = (datasets ?? []).reduce((sum, d) => sum + (d.dataCount || 0), 0);

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Users & Datasets</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Manage users, roles, and dataset permissions.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
          <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Users</p>
          <p className="text-xl font-medium text-white mt-1">
            {usersLoading ? (
              <Loader2 size={16} className="animate-spin inline" />
            ) : (
              usersData?.length ?? 0
            )}
          </p>
        </div>
        <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
          <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Admins</p>
          <p className="text-xl font-medium text-white mt-1">
            {usersLoading ? (
              <Loader2 size={16} className="animate-spin inline" />
            ) : (
              (usersData ?? []).filter((u) => u.is_superuser).length
            )}
          </p>
        </div>
        <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
          <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Datasets</p>
          <p className="text-xl font-medium text-white mt-1">
            {datasetsLoading ? (
              <Loader2 size={16} className="animate-spin inline" />
            ) : (
              datasets?.length ?? 0
            )}
          </p>
        </div>
        <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
          <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Data Items</p>
          <p className="text-xl font-medium text-white mt-1">
            {datasetsLoading ? (
              <Loader2 size={16} className="animate-spin inline" />
            ) : (
              totalDataItems
            )}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
        {/* Tab Header */}
        <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-3">
          <div className="flex">
            <button
              onClick={() => setActiveTab("users")}
              className={cn(
                "px-3 py-2.5 text-xs font-medium border-b-2 -mb-px transition-colors",
                activeTab === "users"
                  ? "border-[var(--text-primary)] text-[var(--text-primary)]"
                  : "border-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              )}
            >
              Users ({usersData?.length ?? 0})
            </button>
            <button
              onClick={() => setActiveTab("datasets")}
              className={cn(
                "px-3 py-2.5 text-xs font-medium border-b-2 -mb-px transition-colors",
                activeTab === "datasets"
                  ? "border-[var(--text-primary)] text-[var(--text-primary)]"
                  : "border-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              )}
            >
              Datasets ({datasets?.length ?? 0})
            </button>
          </div>

          <div className="flex items-center gap-2 py-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search..."
              className="w-40 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded px-2 py-1 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none"
            />
            {/* Create Dataset button - only on datasets tab */}
            {activeTab === "datasets" && (
              <button
                onClick={() => setShowCreateDatasetModal(true)}
                className="px-3 py-1 bg-[var(--text-primary)] text-[var(--bg-base)] text-xs font-medium rounded hover:opacity-90"
              >
                + Create Dataset
              </button>
            )}
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === "users" ? (
          <table className="w-full">
            <thead>
              <tr className="border-b border-[var(--border-subtle)]">
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  User
                </th>
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  Role
                </th>
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  Status
                </th>
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  Verified
                </th>
              </tr>
            </thead>
            <tbody>
              {usersLoading ? (
                <tr>
                  <td colSpan={4} className="py-8 text-center">
                    <Loader2 size={20} className="animate-spin mx-auto text-[var(--text-muted)]" />
                    <p className="text-xs text-[var(--text-muted)] mt-2">Loading users...</p>
                  </td>
                </tr>
              ) : filteredUsers.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-8 text-center">
                    <p className="text-xs text-[var(--text-muted)]">
                      {searchQuery ? "No users match your search" : "No users yet"}
                    </p>
                  </td>
                </tr>
              ) : (
                filteredUsers.map((user) => (
                  <tr
                    key={user.id}
                    className="border-b border-[var(--border-subtle)] last:border-0 hover:bg-[var(--bg-hover)]"
                  >
                    <td className="py-2.5 px-3">
                      <div>
                        <p className="text-xs text-[var(--text-primary)]">
                          {user.email.split("@")[0]}
                        </p>
                        <p className="text-[10px] text-[var(--text-muted)]">{user.email}</p>
                      </div>
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={cn(
                          "px-1.5 py-0.5 text-[10px] rounded",
                          user.is_superuser ? ROLE_STYLES.admin : ROLE_STYLES.user
                        )}
                      >
                        {user.is_superuser ? "admin" : "user"}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-xs text-[var(--text-secondary)]">
                      {user.is_active ? "Active" : "Inactive"}
                    </td>
                    <td className="py-2.5 px-3 text-xs text-[var(--text-secondary)]">
                      {user.is_verified ? "Yes" : "No"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-[var(--border-subtle)]">
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  Dataset
                </th>
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  Owner
                </th>
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  Data Items
                </th>
                <th className="text-left py-2.5 px-3 text-[10px] text-[var(--text-muted)] font-medium uppercase tracking-wider">
                  Created
                </th>
                <th className="w-12"></th>
              </tr>
            </thead>
            <tbody>
              {datasetsLoading ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center">
                    <Loader2 size={20} className="animate-spin mx-auto text-[var(--text-muted)]" />
                    <p className="text-xs text-[var(--text-muted)] mt-2">Loading datasets...</p>
                  </td>
                </tr>
              ) : filteredDatasets.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center">
                    <p className="text-xs text-[var(--text-muted)]">
                      {searchQuery ? "No datasets match your search" : "No datasets yet"}
                    </p>
                  </td>
                </tr>
              ) : (
                filteredDatasets.map((dataset) => (
                  <DatasetRow key={dataset.id} dataset={dataset} userMap={userMap} />
                ))
              )}
            </tbody>
          </table>
        )}
      </div>

      {/* Role Permissions */}
      <div className="mt-6 p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
        <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-3">Role Permissions</p>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <span className={cn("px-1.5 py-0.5 text-[10px] rounded", ROLE_STYLES.admin)}>admin</span>
            <p className="text-xs text-[var(--text-secondary)] mt-2">
              Full access to all features, user management, system configuration.
            </p>
          </div>
          <div>
            <span className={cn("px-1.5 py-0.5 text-[10px] rounded", ROLE_STYLES.user)}>user</span>
            <p className="text-xs text-[var(--text-secondary)] mt-2">
              Ingest and retrieve data, manage own datasets, view statistics.
            </p>
          </div>
          <div>
            <span className={cn("px-1.5 py-0.5 text-[10px] rounded", ROLE_STYLES.viewer)}>viewer</span>
            <p className="text-xs text-[var(--text-secondary)] mt-2">
              Read-only access, search and view data, no modifications.
            </p>
          </div>
        </div>
      </div>

      {/* Modals */}
      <CreateDatasetModal
        isOpen={showCreateDatasetModal}
        onClose={() => setShowCreateDatasetModal(false)}
      />
    </div>
  );
}
