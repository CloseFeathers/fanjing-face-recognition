"use client";

import React from "react";
import { useIngestionConfigStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Database, RotateCcw, Info, Clock } from "lucide-react";

export function IngestionSettings() {
  const config = useIngestionConfigStore();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600">
            <Database className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle>Ingestion Configuration</CardTitle>
            <CardDescription>Configure knowledge extraction and ingestion parameters</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Divider */}
        <div className="relative py-4">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-zinc-800" />
          </div>
          <div className="relative flex justify-center">
            <span className="bg-zinc-900 px-3 text-xs text-zinc-500 flex items-center gap-1.5">
              <Clock size={12} />
              Coming Soon
            </span>
          </div>
        </div>

        {/* Coming Soon Features Notice */}
        <div className="p-4 bg-zinc-800/30 border border-zinc-700/50 rounded-lg">
          <div className="flex items-start gap-3">
            <Info size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-zinc-300">The following options are under development</p>
              <p className="text-xs text-zinc-500 mt-1">
                These features require backend API support and will be available in future versions. For now, use the custom prompt above to guide knowledge extraction.
              </p>
            </div>
          </div>
        </div>

        {/* Disabled Extraction Toggles */}
        <div className="space-y-4 opacity-50 pointer-events-none">
          <Label className="text-zinc-400">Knowledge Extraction Options</Label>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center justify-between p-3 bg-zinc-800/30 rounded-lg border border-zinc-700/30">
              <div>
                <div className="flex items-center gap-2">
                  <Label className="text-sm text-zinc-400">Extract Concepts</Label>
                  <span className="text-[9px] px-1 py-0.5 bg-zinc-700 text-zinc-400 rounded">Soon</span>
                </div>
                <p className="text-xs text-zinc-600">Identify entities from text</p>
              </div>
              <Switch checked={false} disabled />
            </div>

            <div className="flex items-center justify-between p-3 bg-zinc-800/30 rounded-lg border border-zinc-700/30">
              <div>
                <div className="flex items-center gap-2">
                  <Label className="text-sm text-zinc-400">Extract Relations</Label>
                  <span className="text-[9px] px-1 py-0.5 bg-zinc-700 text-zinc-400 rounded">Soon</span>
                </div>
                <p className="text-xs text-zinc-600">Identify relationships between entities</p>
              </div>
              <Switch checked={false} disabled />
            </div>

            <div className="flex items-center justify-between p-3 bg-zinc-800/30 rounded-lg border border-zinc-700/30">
              <div>
                <div className="flex items-center gap-2">
                  <Label className="text-sm text-zinc-400">Generate Summary</Label>
                  <span className="text-[9px] px-1 py-0.5 bg-zinc-700 text-zinc-400 rounded">Soon</span>
                </div>
                <p className="text-xs text-zinc-600">Auto-generate document summaries</p>
              </div>
              <Switch checked={false} disabled />
            </div>
          </div>
        </div>

        {/* Disabled Entity Types */}
        <div className="space-y-3 opacity-50 pointer-events-none">
          <div className="flex items-center gap-2">
            <Label className="text-zinc-400">Custom Entity Types</Label>
            <span className="text-[9px] px-1 py-0.5 bg-zinc-700 text-zinc-400 rounded">Soon</span>
          </div>
          <div className="p-3 bg-zinc-800/30 rounded-lg border border-zinc-700/30 border-dashed">
            <p className="text-xs text-zinc-600 text-center">
              Custom concept types (e.g., Person, Location, Organization) coming soon
            </p>
          </div>
        </div>

        {/* Disabled Relation Types */}
        <div className="space-y-3 opacity-50 pointer-events-none">
          <div className="flex items-center gap-2">
            <Label className="text-zinc-400">Custom Relation Types</Label>
            <span className="text-[9px] px-1 py-0.5 bg-zinc-700 text-zinc-400 rounded">Soon</span>
          </div>
          <div className="p-3 bg-zinc-800/30 rounded-lg border border-zinc-700/30 border-dashed">
            <p className="text-xs text-zinc-600 text-center">
              Custom relation types (e.g., belongs_to, contains, located_in) coming soon
            </p>
          </div>
        </div>

        {/* Reset Button */}
        <div className="flex justify-end pt-4 border-t border-zinc-800">
          <Button variant="ghost" onClick={() => config.reset()}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
