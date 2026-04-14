"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { apiClient } from "@/lib/api/client";
import { Users, Shield, Building2, Plus, UserPlus, Key } from "lucide-react";

type Tab = "roles" | "users" | "tenants";

export function PermissionsPage() {
  const [activeTab, setActiveTab] = useState<Tab>("roles");

  return (
    <div className="max-w-3xl mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Access Control</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">Manage roles, users, and multi-tenancy.</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg w-fit">
        {[
          { id: "roles" as Tab, label: "Roles", icon: <Shield size={14} strokeWidth={1.5} /> },
          { id: "users" as Tab, label: "Users", icon: <Users size={14} strokeWidth={1.5} /> },
          { id: "tenants" as Tab, label: "Tenants", icon: <Building2 size={14} strokeWidth={1.5} /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 text-[13px] rounded-md transition-colors ${
              activeTab === tab.id
                ? "bg-[var(--text-primary)] text-[var(--bg-base)]"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeTab === "roles" && <RolesPanel />}
      {activeTab === "users" && <UsersPanel />}
      {activeTab === "tenants" && <TenantsPanel />}
    </div>
  );
}

// =============================================================================
// Roles Panel
// =============================================================================

function RolesPanel() {
  const [roleName, setRoleName] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  const handleCreate = async () => {
    if (!roleName.trim()) return;
    setIsCreating(true);
    try {
      const res = await apiClient.createRole(roleName);
      toast.success(`Role created: ${res.role_id?.slice(0, 8)}...`);
      setRoleName("");
    } catch {
      toast.error("Failed to create role");
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="p-5 border border-[var(--border-subtle)] rounded-lg space-y-4">
        <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Create Role</h3>
        <div className="space-y-2">
          <Label className="text-[12px] text-[var(--text-muted)]">Role name</Label>
          <input
            value={roleName}
            onChange={(e) => setRoleName(e.target.value)}
            placeholder="e.g., Editor, Viewer, Admin"
            className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)]"
          />
        </div>
        <Button onClick={handleCreate} loading={isCreating} disabled={!roleName.trim()}>
          <Plus size={14} className="mr-2" />
          Create Role
        </Button>
      </div>

      <div className="p-5 border border-[var(--border-subtle)] rounded-lg">
        <h3 className="text-[14px] font-medium text-[var(--text-primary)] mb-3">Existing Roles</h3>
        <p className="text-[13px] text-[var(--text-muted)]">
          Roles are managed at the backend level. Use the REST API for advanced operations.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// Users Panel
// =============================================================================

function UsersPanel() {
  const [userId, setUserId] = useState("");
  const [roleId, setRoleId] = useState("");
  const [isAssigning, setIsAssigning] = useState(false);

  const handleAssignRole = async () => {
    if (!userId.trim() || !roleId.trim()) return;
    setIsAssigning(true);
    try {
      await apiClient.assignRoleToUser(userId, roleId);
      toast.success("Role assigned");
      setUserId("");
      setRoleId("");
    } catch {
      toast.error("Failed to assign role");
    } finally {
      setIsAssigning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="p-5 border border-[var(--border-subtle)] rounded-lg space-y-4">
        <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Assign Role to User</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-[12px] text-[var(--text-muted)]">User ID (UUID)</Label>
            <input
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
              className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] font-mono"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-[12px] text-[var(--text-muted)]">Role ID (UUID)</Label>
            <input
              value={roleId}
              onChange={(e) => setRoleId(e.target.value)}
              placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
              className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] font-mono"
            />
          </div>
        </div>
        <Button onClick={handleAssignRole} loading={isAssigning} disabled={!userId.trim() || !roleId.trim()}>
          <UserPlus size={14} className="mr-2" />
          Assign Role
        </Button>
      </div>

      <div className="p-5 border border-[var(--border-subtle)] rounded-lg space-y-4">
        <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Dataset Permissions</h3>
        <p className="text-[13px] text-[var(--text-muted)]">
          Grant users or roles access to specific datasets (read, write, admin).
        </p>
        <DatasetPermissionForm />
      </div>
    </div>
  );
}

function DatasetPermissionForm() {
  const [principalId, setPrincipalId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [permission, setPermission] = useState("read");
  const [isGranting, setIsGranting] = useState(false);

  const handleGrant = async () => {
    if (!principalId.trim() || !datasetId.trim()) return;
    setIsGranting(true);
    try {
      await apiClient.grantDatasetPermission(principalId, [datasetId], permission);
      toast.success("Permission granted");
      setPrincipalId("");
      setDatasetId("");
    } catch {
      toast.error("Failed to grant permission");
    } finally {
      setIsGranting(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label className="text-[12px] text-[var(--text-muted)]">User/Role ID</Label>
          <input
            value={principalId}
            onChange={(e) => setPrincipalId(e.target.value)}
            placeholder="UUID"
            className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] font-mono"
          />
        </div>
        <div className="space-y-2">
          <Label className="text-[12px] text-[var(--text-muted)]">Dataset ID</Label>
          <input
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            placeholder="UUID"
            className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] font-mono"
          />
        </div>
        <div className="space-y-2">
          <Label className="text-[12px] text-[var(--text-muted)]">Permission</Label>
          <select
            value={permission}
            onChange={(e) => setPermission(e.target.value)}
            className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-default)]"
          >
            <option value="read">Read</option>
            <option value="write">Write</option>
            <option value="admin">Admin</option>
          </select>
        </div>
      </div>
      <Button onClick={handleGrant} loading={isGranting} disabled={!principalId.trim() || !datasetId.trim()}>
        <Key size={14} className="mr-2" />
        Grant Permission
      </Button>
    </div>
  );
}

// =============================================================================
// Tenants Panel
// =============================================================================

function TenantsPanel() {
  const [tenantName, setTenantName] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [assignUserId, setAssignUserId] = useState("");
  const [assignTenantId, setAssignTenantId] = useState("");
  const [isAssigning, setIsAssigning] = useState(false);

  const handleCreate = async () => {
    if (!tenantName.trim()) return;
    setIsCreating(true);
    try {
      const res = await apiClient.createTenant(tenantName);
      toast.success(`Tenant created: ${res.tenant_id?.slice(0, 8)}...`);
      setTenantName("");
    } catch {
      toast.error("Failed to create tenant");
    } finally {
      setIsCreating(false);
    }
  };

  const handleAssign = async () => {
    if (!assignUserId.trim() || !assignTenantId.trim()) return;
    setIsAssigning(true);
    try {
      await apiClient.assignTenantToUser(assignUserId, assignTenantId);
      toast.success("User added to tenant");
      setAssignUserId("");
      setAssignTenantId("");
    } catch {
      toast.error("Failed to assign tenant");
    } finally {
      setIsAssigning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="p-5 border border-[var(--border-subtle)] rounded-lg space-y-4">
        <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Create Tenant</h3>
        <p className="text-[12px] text-[var(--text-muted)]">
          Tenants provide resource isolation between organizations.
        </p>
        <div className="space-y-2">
          <Label className="text-[12px] text-[var(--text-muted)]">Tenant name</Label>
          <input
            value={tenantName}
            onChange={(e) => setTenantName(e.target.value)}
            placeholder="e.g., Acme Corp"
            className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)]"
          />
        </div>
        <Button onClick={handleCreate} loading={isCreating} disabled={!tenantName.trim()}>
          <Plus size={14} className="mr-2" />
          Create Tenant
        </Button>
      </div>

      <div className="p-5 border border-[var(--border-subtle)] rounded-lg space-y-4">
        <h3 className="text-[14px] font-medium text-[var(--text-primary)]">Add User to Tenant</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-[12px] text-[var(--text-muted)]">User ID (UUID)</Label>
            <input
              value={assignUserId}
              onChange={(e) => setAssignUserId(e.target.value)}
              placeholder="UUID"
              className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] font-mono"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-[12px] text-[var(--text-muted)]">Tenant ID (UUID)</Label>
            <input
              value={assignTenantId}
              onChange={(e) => setAssignTenantId(e.target.value)}
              placeholder="UUID"
              className="w-full h-10 px-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-md text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--border-default)] font-mono"
            />
          </div>
        </div>
        <Button onClick={handleAssign} loading={isAssigning} disabled={!assignUserId.trim() || !assignTenantId.trim()}>
          <UserPlus size={14} className="mr-2" />
          Add to Tenant
        </Button>
      </div>
    </div>
  );
}
