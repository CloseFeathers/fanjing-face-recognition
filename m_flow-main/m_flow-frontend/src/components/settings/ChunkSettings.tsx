"use client";

import React from "react";
import { useChunkConfigStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Scissors, RotateCcw } from "lucide-react";

const strategies = [
  { value: "paragraph", label: "By Paragraph", description: "Split text by paragraphs" },
  { value: "sentence", label: "By Sentence", description: "Split text by sentences" },
  { value: "word", label: "By Word Count", description: "Split by fixed word count" },
  { value: "fixed", label: "Fixed Size", description: "Split by fixed character count" },
];

export function ChunkSettings() {
  const config = useChunkConfigStore();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600">
            <Scissors className="h-5 w-5 text-white" />
          </div>
          <div>
            <CardTitle>Chunking Configuration</CardTitle>
            <CardDescription>Configure text chunking strategy and parameters</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Chunking Strategy */}
        <div className="space-y-2">
          <Label>Chunking Strategy</Label>
          <Select
            value={config.chunk_strategy}
            onValueChange={(v) =>
              config.setConfig({ chunk_strategy: v as "paragraph" | "sentence" | "word" | "fixed" })
            }
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {strategies.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  <div>
                    <div className="font-medium">{s.label}</div>
                    <div className="text-xs text-zinc-500">{s.description}</div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Chunk Size */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Chunk Size (Tokens)</Label>
            <span className="text-sm text-zinc-400">{config.chunk_size}</span>
          </div>
          <Slider
            value={[config.chunk_size]}
            onValueChange={([v]) => config.setConfig({ chunk_size: v })}
            min={128}
            max={4096}
            step={64}
          />
          <p className="text-xs text-zinc-500">
            Larger chunks preserve more context, smaller chunks provide more precise retrieval
          </p>
        </div>

        {/* Overlap Size */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Overlap Size (Tokens)</Label>
            <span className="text-sm text-zinc-400">{config.chunk_overlap}</span>
          </div>
          <Slider
            value={[config.chunk_overlap]}
            onValueChange={([v]) => config.setConfig({ chunk_overlap: v })}
            min={0}
            max={512}
            step={16}
          />
          <p className="text-xs text-zinc-500">
            Overlap between chunks helps maintain context continuity
          </p>
        </div>

        {/* Encoding Model */}
        <div className="space-y-2">
          <Label>Tokenizer Encoding Model</Label>
          <Input
            value={config.chunker_encoding_model || ""}
            onChange={(e) => config.setConfig({ chunker_encoding_model: e.target.value })}
            placeholder="cl100k_base"
          />
          <p className="text-xs text-zinc-500">
            Encoding model for token count calculation, defaults to cl100k_base (GPT-4)
          </p>
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
