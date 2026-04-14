"use client";

import React from "react";
import { useHealthCheck, useSettings } from "@/hooks/use-api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Check, X, RefreshCw, Server, Brain, Network, Layers, Database, Clock, Loader2 } from "lucide-react";

interface ServiceStatus {
  name: string;
  icon: React.ReactNode;
  status: "healthy" | "unhealthy" | "unknown";
  details?: string;
}

export function HealthPage() {
  const { data: health, isLoading, error, refetch, dataUpdatedAt } = useHealthCheck();
  const { data: settings } = useSettings();

  const services: ServiceStatus[] = [
    { name: "API Service", icon: <Server size={16} strokeWidth={1.5} />, status: health?.status === "healthy" ? "healthy" : error ? "unhealthy" : "unknown", details: health?.version || "v0.5.1" },
    { name: "LLM", icon: <Brain size={16} strokeWidth={1.5} />, status: settings?.llm ? "healthy" : "unknown", details: settings?.llm?.llm_model || "Not configured" },
    { name: "Vector DB", icon: <Layers size={16} strokeWidth={1.5} />, status: settings?.vector_db ? "healthy" : "unknown", details: settings?.vector_db?.provider || "lancedb" },
    { name: "Graph DB", icon: <Network size={16} strokeWidth={1.5} />, status: "healthy", details: "kuzu" },
    { name: "Embedding", icon: <Database size={16} strokeWidth={1.5} />, status: settings?.embedding ? "healthy" : "unknown", details: settings?.embedding?.embedding_model || "Not configured" },
  ];

  const healthyCount = services.filter((s) => s.status === "healthy").length;
  const overallStatus = healthyCount === services.length ? "healthy" : healthyCount > 0 ? "degraded" : "unhealthy";

  return (
    <div className="max-w-2xl mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-lg font-medium text-[var(--text-primary)]">System Health</h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">Monitor component status.</p>
        </div>
        <Button variant="outline" onClick={() => refetch()} disabled={isLoading}>
          {isLoading ? <Loader2 size={14} className="animate-spin mr-2" /> : <RefreshCw size={14} className="mr-2" />}
          Refresh
        </Button>
      </div>

      {/* Overall Status */}
      <div className={cn(
        "flex items-center justify-between p-5 border rounded-lg",
        overallStatus === "healthy" ? "border-[var(--success)]/30 bg-[var(--success)]/5" : overallStatus === "degraded" ? "border-[var(--warning)]/30 bg-[var(--warning)]/5" : "border-[var(--error)]/30 bg-[var(--error)]/5"
      )}>
        <div className="flex items-center gap-4">
          <div className={cn(
            "w-3 h-3 rounded-full",
            overallStatus === "healthy" ? "bg-[var(--success)]" : overallStatus === "degraded" ? "bg-[var(--warning)]" : "bg-[var(--error)]"
          )} />
          <div>
            <p className={cn(
              "text-[16px] font-medium",
              overallStatus === "healthy" ? "text-[var(--success)]" : overallStatus === "degraded" ? "text-[var(--warning)]" : "text-[var(--error)]"
            )}>
              {overallStatus === "healthy" ? "All systems operational" : overallStatus === "degraded" ? "Partial outage" : "System down"}
            </p>
            <p className="text-[12px] text-[var(--text-muted)]">{healthyCount}/{services.length} services healthy</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-[11px] text-[var(--text-muted)] flex items-center gap-1">
            <Clock size={10} />
            {dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : "-"}
          </p>
        </div>
      </div>

      {/* Services */}
      <div className="space-y-2">
        <p className="text-[12px] text-[var(--text-muted)]">Services</p>
        {services.map((service, index) => (
          <div
            key={index}
            className={cn(
              "flex items-center justify-between p-4 border rounded-lg transition-colors",
              service.status === "healthy" ? "border-[var(--border-subtle)]" : service.status === "unhealthy" ? "border-[var(--error)]/30 bg-[var(--error)]/5" : "border-[var(--border-subtle)]"
            )}
          >
            <div className="flex items-center gap-3">
              <span className="text-[var(--text-muted)]">{service.icon}</span>
              <div>
                <p className="text-[13px] font-medium text-[var(--text-primary)]">{service.name}</p>
                <p className="text-[11px] text-[var(--text-muted)]">{service.details}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {service.status === "healthy" ? (
                <Check size={14} className="text-[var(--success)]" />
              ) : service.status === "unhealthy" ? (
                <X size={14} className="text-[var(--error)]" />
              ) : (
                <span className="w-2 h-2 rounded-full bg-[var(--text-muted)]" />
              )}
              <span className={cn(
                "text-[12px]",
                service.status === "healthy" ? "text-[var(--success)]" : service.status === "unhealthy" ? "text-[var(--error)]" : "text-[var(--text-muted)]"
              )}>
                {service.status === "healthy" ? "OK" : service.status === "unhealthy" ? "Error" : "Unknown"}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* System Info */}
      <div className="space-y-2">
        <p className="text-[12px] text-[var(--text-muted)]">System info</p>
        <div className="grid grid-cols-4 gap-3">
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
            <p className="text-[10px] text-[var(--text-muted)] mb-1">Version</p>
            <p className="text-[13px] text-[var(--text-primary)] font-medium">{health?.version || "-"}</p>
          </div>
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
            <p className="text-[10px] text-[var(--text-muted)] mb-1">LLM</p>
            <p className="text-[13px] text-[var(--text-primary)] font-medium truncate">{settings?.llm?.llm_model || "-"}</p>
          </div>
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
            <p className="text-[10px] text-[var(--text-muted)] mb-1">Vector</p>
            <p className="text-[13px] text-[var(--text-primary)] font-medium">{settings?.vector_db?.provider || "lancedb"}</p>
          </div>
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-lg">
            <p className="text-[10px] text-[var(--text-muted)] mb-1">Graph</p>
            <p className="text-[13px] text-[var(--text-primary)] font-medium">kuzu</p>
          </div>
        </div>
      </div>
    </div>
  );
}
