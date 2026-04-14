"use client";

import React, { useState, useCallback, useRef } from "react";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/lib/store";
import { useAddTextMemory, useUploadFile } from "@/hooks/use-api";
import { FileTypeIcon, getActionableErrorMessage } from "@/components/common";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { Upload, File, X, Check, AlertCircle, Loader2, ChevronDown, ChevronUp, Settings2, Tag, Info } from "lucide-react";
import { CompactTagInput } from "@/components/common";

// ============================================================================
// Types
// ============================================================================

interface FileWithStatus {
  file: File;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

// ============================================================================
// Imports & Constants
// ============================================================================

import { 
  ACCEPTED_FILE_TYPES, 
  validateFiles 
} from "@/lib/file-validation";

const API_EXAMPLE = `import m_flow

result = await m_flow.add(
    data="Your content or file path",
    dataset_name="my_dataset",
    incremental_loading=True,
)`;

const ADD_PARAMS = [
  { name: "data", type: "str | List[str] | BinaryIO", required: true, description: "Content, file path, URL, or binary stream" },
  { name: "dataset_name", type: "str", default: '"main_dataset"', description: "Target dataset name" },
  { name: "preferred_loaders", type: 'List[str | dict]', default: "None", description: 'File loaders, e.g. ["pdf", "docx"]' },
  { name: "incremental_loading", type: "bool", default: "True", description: "Skip already processed files" },
  { name: "items_per_batch", type: "int", default: "20", description: "Items per processing batch" },
];

const FORMATS: { name: string; note?: string }[] = [
  { name: "TXT" },
  { name: "MD" },
  { name: "JSON" },
  { name: "CSV" },
  { name: "PDF", note: "requires pypdf" },
  { name: "HTML" },
  { name: "Code", note: ".py .js .ts .yaml" },
  { name: "DOCX", note: "requires unstructured" },
];

// ============================================================================
// Components
// ============================================================================

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="relative group">
      <pre className="bg-[#0a0a0a] border border-[var(--border-subtle)] rounded p-3 overflow-x-auto">
        <code className="text-xs text-[var(--text-secondary)] font-mono whitespace-pre">{code}</code>
      </pre>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 px-2 py-1 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

function FileItem({ 
  file, 
  onRemove 
}: { 
  file: FileWithStatus; 
  onRemove: () => void;
}) {
  return (
    <div className={cn(
      "flex items-center gap-3 p-2 bg-[var(--bg-surface)] border rounded transition-colors",
      file.status === "success" 
        ? "border-[var(--success)]/30 bg-[var(--success)]/5" 
        : file.status === "error"
          ? "border-[var(--error)]/30 bg-[var(--error)]/5"
          : "border-[var(--border-subtle)]"
    )}>
      <div className="relative flex-shrink-0">
        <FileTypeIcon filename={file.file.name} size={14} />
        {file.status === "uploading" && (
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-[var(--text-primary)] truncate">{file.file.name}</p>
        <p className="text-[10px] text-[var(--text-muted)]">
          {(file.file.size / 1024).toFixed(1)} KB
        </p>
      </div>
      {file.status === "pending" && (
        <button onClick={onRemove} className="text-[var(--text-muted)] hover:text-[var(--error)] transition-colors">
          <X size={14} />
        </button>
      )}
      {file.status === "uploading" && (
        <Loader2 size={14} className="text-blue-400 animate-spin" />
      )}
      {file.status === "success" && (
        <Check size={14} className="text-[var(--success)]" />
      )}
      {file.status === "error" && (
        <AlertCircle size={14} className="text-[var(--error)]" />
      )}
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export function AddPage() {
  const { datasetContext } = useUIStore();
  const inputRef = useRef<HTMLInputElement>(null);

  // Input state
  const [inputType, setInputType] = useState<"text" | "file">("text");
  const [textContent, setTextContent] = useState("");
  const [files, setFiles] = useState<FileWithStatus[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  // Options
  const [incrementalLoading, setIncrementalLoading] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [nodeSet, setNodeSet] = useState<string[]>([]);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);

  const addNode = (name: string) => {
    if (!nodeSet.includes(name)) {
      setNodeSet([...nodeSet, name]);
    }
  };

  const removeNode = (name: string) => {
    setNodeSet(nodeSet.filter((n) => n !== name));
  };

  // API hooks
  const addTextMemory = useAddTextMemory();
  const uploadFile = useUploadFile();

  // Dataset name from context or default
  const datasetName = datasetContext.datasetName || "main_dataset";

  // File handling with validation
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const { valid, rejected } = validateFiles(selectedFiles);
    
    rejected.forEach(({ file, reason }) => {
      toast.error(`${file.name}: ${reason}`);
    });
    
    if (valid.length > 0) {
      setFiles((prev) => [...prev, ...valid.map((file) => ({ file, status: "pending" as const }))]);
    }
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    const { valid, rejected } = validateFiles(droppedFiles);
    
    rejected.forEach(({ file, reason }) => {
      toast.error(`${file.name}: ${reason}`);
    });
    
    if (valid.length > 0) {
      setFiles((prev) => [...prev, ...valid.map((file) => ({ file, status: "pending" as const }))]);
    }
  }, []);

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // Submit handler - calls add() API only (no memorize)
  const handleSubmit = async () => {
    // Validation
    if (inputType === "text" && !textContent.trim()) {
      toast.error("Please enter some text content");
      return;
    }
    if (inputType === "file" && files.filter(f => f.status === "pending").length === 0) {
      toast.error("Please select at least one file");
      return;
    }

    setIsProcessing(true);

    try {
      if (inputType === "text") {
        await addTextMemory.mutateAsync({ 
          text: textContent, 
          datasetName,
          nodeSet: nodeSet.length > 0 ? nodeSet : undefined,
          incrementalLoading,
        });
        toast.success("Text added to dataset");
        setTextContent("");
      } else if (inputType === "file") {
        const pendingFiles = files.filter(f => f.status === "pending");
        let successCount = 0;
        let errorCount = 0;

        for (let i = 0; i < pendingFiles.length; i++) {
          const fileItem = pendingFiles[i];
          const fileIndex = files.findIndex(f => f === fileItem);
          
          setFiles((prev) => prev.map((f, idx) => 
            idx === fileIndex ? { ...f, status: "uploading" as const } : f
          ));

          try {
            await uploadFile.mutateAsync({ 
              file: fileItem.file, 
              datasetName,
              nodeSet: nodeSet.length > 0 ? nodeSet : undefined,
              incrementalLoading,
            });
            setFiles((prev) => prev.map((f, idx) => 
              idx === fileIndex ? { ...f, status: "success" as const } : f
            ));
            successCount++;
          } catch (error) {
            setFiles((prev) => prev.map((f, idx) => 
              idx === fileIndex ? { 
                ...f, 
                status: "error" as const, 
                error: error instanceof Error ? error.message : "Upload failed" 
              } : f
            ));
            errorCount++;
          }
        }

        if (successCount > 0) {
          toast.success(`${successCount} file(s) added to dataset`);
        }
        if (errorCount > 0) {
          toast.error(`${errorCount} file(s) failed to upload`);
        }
      }

    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Add operation failed");
      toast.error(actionableMessage);
      console.error("Add error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Check if can submit
  const canSubmit = !isProcessing && (
    (inputType === "text" && textContent.trim()) ||
    (inputType === "file" && files.filter(f => f.status === "pending").length > 0)
  );

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Add Content</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Store content without processing. Use this to add data now and build your knowledge graph later via "Build Graph".
        </p>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Input Form */}
        <div className="col-span-7 space-y-5">
          {/* Source Type */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Source Type
            </label>
            <div className="flex gap-1 p-1 bg-[var(--bg-surface)] rounded border border-[var(--border-subtle)]">
              {["text", "file"].map((type) => (
                <button
                  key={type}
                  onClick={() => setInputType(type as "text" | "file")}
                  disabled={isProcessing}
                  className={cn(
                    "flex-1 py-1.5 text-xs rounded transition-colors",
                    inputType === type
                      ? "bg-[var(--bg-base)] text-[var(--text-primary)]"
                      : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]",
                    isProcessing && "opacity-50 cursor-not-allowed"
                  )}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
              {/* URL support requires backend REST API changes */}
              <div className="relative group flex-1">
                <button
                  disabled
                  className="w-full py-1.5 text-xs rounded text-[var(--text-muted)] opacity-50 cursor-not-allowed"
                >
                  URL
                </button>
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                  Coming soon
                </div>
              </div>
            </div>
          </div>

          {/* Input Area */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              {inputType === "text" ? "Content" : "Files"}
            </label>

            {inputType === "text" && (
              <textarea
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                placeholder="Enter or paste text content..."
                disabled={isProcessing}
                className={cn(
                  "w-full h-40 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] resize-none focus:outline-none focus:border-[var(--text-muted)]",
                  isProcessing && "opacity-50 cursor-not-allowed"
                )}
              />
            )}

            {inputType === "file" && (
              <div className="space-y-3">
                {/* Drop Zone with Accessibility */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => !isProcessing && inputRef.current?.click()}
                  onKeyDown={(e) => {
                    if ((e.key === "Enter" || e.key === " ") && !isProcessing) {
                      e.preventDefault();
                      inputRef.current?.click();
                    }
                  }}
                  role="button"
                  tabIndex={isProcessing ? -1 : 0}
                  aria-label="Drop files here or click to browse. Supports 30+ file formats."
                  aria-disabled={isProcessing}
                  className={cn(
                    "flex flex-col items-center justify-center p-6 border-2 border-dashed rounded cursor-pointer transition-all",
                    "focus:outline-none focus:ring-2 focus:ring-[var(--text-primary)] focus:ring-offset-2 focus:ring-offset-[var(--bg-base)]",
                    isDragging 
                      ? "border-[var(--text-primary)] bg-[var(--text-primary)]/5 scale-[1.02]" 
                      : "border-[var(--border-subtle)] bg-[var(--bg-surface)] hover:border-[var(--border-default)]",
                    isProcessing && "opacity-50 cursor-not-allowed"
                  )}
                >
                  <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept={ACCEPTED_FILE_TYPES.join(",")}
                    onChange={handleFileSelect}
                    className="hidden"
                    disabled={isProcessing}
                    aria-hidden="true"
                  />
                  <Upload size={20} className={cn(
                    "mb-2 transition-transform",
                    isDragging ? "text-[var(--text-primary)] -translate-y-1" : "text-[var(--text-muted)]"
                  )} />
                  <p className="text-sm text-[var(--text-secondary)]">
                    {isDragging ? "Release to upload" : "Drop files here or click to browse"}
                  </p>
                  <p className="text-xs text-[var(--text-muted)] mt-1">Supports 30+ file formats</p>
                </div>

                {/* File List */}
                {files.length > 0 && (
                  <div className="space-y-2">
                    {files.map((file, index) => (
                      <FileItem 
                        key={`${file.file.name}-${index}`}
                        file={file} 
                        onRemove={() => removeFile(index)} 
                      />
                    ))}
                  </div>
                )}
              </div>
            )}

          </div>

          {/* Options */}
          <div className="flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <div>
              <span className="text-xs text-[var(--text-secondary)]">Incremental Loading</span>
              <p className="text-[10px] text-[var(--text-muted)]">Skip already processed files</p>
            </div>
            <Switch
              checked={incrementalLoading}
              onCheckedChange={setIncrementalLoading}
              disabled={isProcessing}
            />
          </div>

          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={isProcessing}
            className={cn(
              "w-full flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded text-left",
              isProcessing && "opacity-50 cursor-not-allowed"
            )}
          >
            <div className="flex items-center gap-2">
              <Settings2 size={14} className="text-[var(--text-muted)]" />
              <span className="text-xs text-[var(--text-secondary)]">Advanced Options</span>
            </div>
            {showAdvanced ? (
              <ChevronUp size={14} className="text-[var(--text-muted)]" />
            ) : (
              <ChevronDown size={14} className="text-[var(--text-muted)]" />
            )}
          </button>

          {/* Advanced Options Panel */}
          {showAdvanced && (
            <div className="space-y-4 p-4 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
              {/* Node Set */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Tag size={12} className="text-[var(--text-muted)]" />
                  <span className="text-xs text-[var(--text-secondary)]">Node Set</span>
                  <div className="group relative">
                    <Info size={12} className="text-[var(--text-muted)] cursor-help" />
                    <div className="absolute bottom-full left-0 mb-1 px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] w-52 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      Graph node identifiers for organizing and filtering data in the knowledge graph.
                    </div>
                  </div>
                </div>
                <CompactTagInput
                  tags={nodeSet}
                  onAdd={addNode}
                  onRemove={removeNode}
                  placeholder="Node name..."
                  maxTags={5}
                  disabled={isProcessing}
                />
                <p className="text-[10px] text-[var(--text-muted)]">
                  Assign nodes to organize data for filtering in the knowledge graph
                </p>
              </div>
            </div>
          )}

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={cn(
              "w-full py-2.5 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded transition-opacity flex items-center justify-center gap-2",
              canSubmit ? "hover:opacity-90" : "opacity-50 cursor-not-allowed"
            )}
          >
            {isProcessing && <Loader2 size={14} className="animate-spin" />}
            {isProcessing ? "Adding data..." : "Run Add"}
          </button>

          {/* Dataset Info */}
          <p className="text-xs text-[var(--text-muted)]">
            Target: <span className="text-[var(--text-secondary)]">{datasetName}</span>
          </p>

          {/* Supported Formats */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Supported Formats
            </label>
            <div className="flex flex-wrap gap-1">
              {FORMATS.map((f) => (
                <span key={f.name} className="px-2 py-0.5 text-[10px] bg-[var(--bg-surface)] text-[var(--text-muted)] rounded border border-[var(--border-subtle)]" title={f.note || ""}>
                  {f.name}{f.note ? " *" : ""}
                </span>
              ))}
            </div>
            <p className="text-[9px] text-[var(--text-muted)] mt-1.5">
              * PDF requires <code className="bg-[var(--bg-elevated)] px-0.5 rounded">pypdf</code> package. DOCX requires <code className="bg-[var(--bg-elevated)] px-0.5 rounded">unstructured</code> package.
            </p>
          </div>
        </div>

        {/* Right: Reference */}
        <div className="col-span-5 space-y-5">
          {/* API Example */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              API Reference
            </label>
            <CodeBlock code={API_EXAMPLE} />
          </div>

          {/* Note */}
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <p className="text-xs text-[var(--text-secondary)]">
              <span className="text-[var(--text-primary)]">Note:</span> Add only stores data. 
              Run <code className="bg-[var(--bg-elevated)] px-1 rounded">memorize()</code> to build the knowledge graph.
            </p>
          </div>

          {/* Workflow Info */}
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <p className="text-xs text-[var(--text-muted)] mb-2">Workflow Position</p>
            <div className="flex items-center gap-2 text-xs">
              <span className="px-2 py-0.5 bg-[var(--text-primary)] text-[var(--bg-base)] rounded font-medium">add()</span>
              <span className="text-[var(--text-muted)]">→</span>
              <span className="px-2 py-0.5 bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded">memorize()</span>
            </div>
          </div>

          {/* Parameters */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Parameters
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              {ADD_PARAMS.map((param) => (
                <div key={param.name} className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <code className="text-xs text-[var(--text-primary)]">{param.name}</code>
                    {param.required && <span className="text-[10px] text-[var(--error)]">required</span>}
                  </div>
                  <p className="text-[11px] text-[var(--text-muted)] mt-0.5">{param.description}</p>
                  {param.default && (
                    <p className="text-[10px] text-[var(--text-muted)] mt-0.5 font-mono">
                      default = {param.default}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
