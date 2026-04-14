"use client";

import React from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useCorefConfig, useCorefStats, corefQueryKeys } from "@/hooks/use-coreference";
import { apiClient } from "@/lib/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Link2, RefreshCw, Activity, Loader2, AlertCircle, RotateCcw } from "lucide-react";
import type { CorefLanguage } from "@/types";

const languageOptions: { value: CorefLanguage; label: string; description: string }[] = [
  { value: "auto", label: "Auto Detect", description: "Detect language from query content" },
  { value: "zh", label: "Chinese", description: "Use Chinese coreference resolver" },
  { value: "en", label: "English", description: "Use English coreference resolver" },
];

export function CoreferenceSettings() {
  const {
    config,
    isLoading,
    error,
    updateConfig,
    isUpdating,
    refetch,
  } = useCorefConfig();

  const {
    stats,
    isLoading: statsLoading,
    refetch: refetchStats,
  } = useCorefStats({ autoRefresh: true, refreshInterval: 30000 });

  const {
    stats: sessionStats,
    isLoading: sessionsLoading,
    refetch: refetchSessions,
  } = useCorefStats({ includeSessions: true, autoRefresh: true, refreshInterval: 30000 });

  const queryClient = useQueryClient();
  const resetMutation = useMutation({
    mutationFn: (sessionId: string) => apiClient.resetCorefSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: corefQueryKeys.stats });
    },
  });

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-zinc-400" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12 gap-3">
          <AlertCircle className="h-8 w-8 text-amber-500" />
          <p className="text-sm text-zinc-400">Failed to load coreference configuration</p>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!config) return null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600">
              <Link2 className="h-5 w-5 text-white" />
            </div>
            <div>
              <CardTitle className="flex items-center gap-2">
                Coreference Resolution
                {isUpdating && <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />}
              </CardTitle>
              <CardDescription>
                Resolve pronouns and anaphoric references in queries
              </CardDescription>
            </div>
          </div>
          <Switch
            checked={config.enabled}
            onCheckedChange={(enabled) => updateConfig({ enabled })}
            disabled={isUpdating}
            title={config.enabled ? "Disable coreference preprocessing" : "Enable coreference preprocessing"}
          />
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Status Badge */}
        {stats && (
          <div className="flex items-center gap-4 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
            <Activity className="h-4 w-4 text-zinc-400" />
            <div className="flex items-center gap-3 text-sm">
              <span className="px-2 py-0.5 rounded border border-zinc-700 bg-zinc-800/50 font-mono text-xs">
                {stats.active_sessions} / {stats.max_sessions}
              </span>
              <span className="text-zinc-500">active sessions</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto"
              onClick={() => refetchStats()}
              disabled={statsLoading}
            >
              <RefreshCw className={`h-3 w-3 ${statsLoading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        )}

        {/* Language Mode */}
        <div className="space-y-2">
          <Label>Language Mode</Label>
          <Select
            value={config.language}
            onValueChange={(v) => updateConfig({ language: v as CorefLanguage })}
            disabled={isUpdating || !config.enabled}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languageOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  <div>
                    <div className="font-medium">{opt.label}</div>
                    <div className="text-xs text-zinc-500">{opt.description}</div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Entity History Limit */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Entity History Limit</Label>
            <span className="text-sm font-mono text-zinc-400">{config.max_history}</span>
          </div>
          <Slider
            value={[config.max_history]}
            onValueChange={([v]) => updateConfig({ max_history: v })}
            min={5}
            max={50}
            step={5}
            disabled={isUpdating || !config.enabled}
          />
          <p className="text-xs text-zinc-500">
            Higher values improve accuracy but increase memory usage (recommended: 10-20)
          </p>
        </div>

        {/* Session TTL */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Session TTL</Label>
            <span className="text-sm font-mono text-zinc-400">
              {formatDuration(config.session_ttl)}
            </span>
          </div>
          <Slider
            value={[config.session_ttl]}
            onValueChange={([v]) => updateConfig({ session_ttl: v })}
            min={60}
            max={86400}
            step={60}
            disabled={isUpdating || !config.enabled}
          />
          <p className="text-xs text-zinc-500">
            Inactive sessions are automatically cleaned up after this duration
          </p>
        </div>

        {/* Max Sessions (read-only info) */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-900/30">
          <div>
            <Label className="text-sm font-medium text-zinc-400">Max Sessions</Label>
            <p className="text-xs text-zinc-500 mt-1">
              Maximum concurrent sessions (server configured)
            </p>
          </div>
          <span className="text-sm font-mono text-zinc-400">
            {stats?.max_sessions?.toLocaleString() ?? "—"}
          </span>
        </div>

        {/* Paragraph Reset */}
        <div className="flex items-center justify-between p-3 rounded-lg border border-zinc-800">
          <div>
            <Label className="text-sm font-medium">Paragraph Reset</Label>
            <p className="text-xs text-zinc-500 mt-1">
              Reset partial context on new conversation turns
            </p>
          </div>
          <Switch
            checked={config.paragraph_reset}
            onCheckedChange={(paragraph_reset) => updateConfig({ paragraph_reset })}
            disabled={isUpdating || !config.enabled}
          />
        </div>

        {/* Sessions */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Sessions</Label>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refetchSessions()}
              disabled={sessionsLoading}
            >
              <RefreshCw className={`h-3 w-3 ${sessionsLoading ? "animate-spin" : ""}`} />
            </Button>
          </div>

          {sessionsLoading && !sessionStats?.sessions ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
            </div>
          ) : sessionStats?.sessions && sessionStats.sessions.length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-auto">
              {sessionStats.sessions.map((session) => (
                <div
                  key={session.session_id}
                  className="flex items-center justify-between p-3 rounded-lg bg-zinc-900/30 border border-zinc-800"
                >
                  <div className="flex-1 min-w-0 mr-3">
                    <code className="text-[11px] text-zinc-400 font-mono block truncate">
                      {session.session_id}
                    </code>
                    <div className="flex items-center gap-3 mt-1 text-[11px] text-zinc-500">
                      <span>{session.turn_count} turns</span>
                      <span>{formatRelativeTime(session.last_active)}</span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => resetMutation.mutate(session.session_id)}
                    disabled={resetMutation.isPending}
                    title="Reset session"
                  >
                    <RotateCcw className={`h-3 w-3 ${resetMutation.isPending ? "animate-spin" : ""}`} />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-500 py-3 text-center">No active sessions</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
}

function formatRelativeTime(isoString: string): string {
  const diffMs = Date.now() - new Date(isoString).getTime();
  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
