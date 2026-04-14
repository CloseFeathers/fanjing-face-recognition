"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/lib/store";
import { useManualIngest } from "@/hooks/use-api";
import { getActionableErrorMessage } from "@/components/common";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { 
  Plus, 
  Trash2, 
  ChevronDown, 
  ChevronRight, 
  Loader2,
  BookOpen,
  Tag,
  User,
  FileText,
  Check,
  AlertCircle
} from "lucide-react";
import type { 
  ManualEpisodeInput, 
  ManualFacetInput, 
  ManualFacetPointInput,
  ManualConceptInput 
} from "@/types";

// ============================================================================
// API Example
// ============================================================================

const API_EXAMPLE = `import m_flow

await m_flow.manual_ingest(
    episodes=[
        {
            "name": "Project Alpha Launch",
            "summary": "Team launched v1.0 on March 15th",
            "facets": [
                {
                    "facet_type": "decision",
                    "search_text": "Chose React + FastAPI stack"
                }
            ],
            "entities": [
                {
                    "name": "Project Alpha",
                    "description": "Main product launch",
                    "entity_type": "Product"
                }
            ]
        }
    ],
    dataset_name="my_dataset",
)`;

// ============================================================================
// Schema Info
// ============================================================================

const EPISODE_SCHEMA = [
  { name: "name", type: "str", required: true, description: "Episode title" },
  { name: "summary", type: "str", required: true, description: "Main content summary (indexed)" },
  { name: "display_only", type: "str", default: "—", description: "Display text (not indexed, for full content)" },
  { name: "memory_type", type: "str", default: "episodic", description: "'episodic' or 'atomic'" },
  { name: "facets", type: "List[Facet]", description: "Detail anchors (each may contain FacetPoints)" },
  { name: "entities", type: "List[Entity]", description: "Related entities" },
];

const ENTITY_TYPES = ["Person", "Organization", "Location", "Event", "Product", "Entity", "Thing"];
const FACET_TYPES = ["chapter", "article", "decision", "risk", "outcome", "metric", "issue", "plan", "constraint"];

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

// ============================================================================
// Entity Editor
// ============================================================================

interface EntityEditorProps {
  entity: ManualConceptInput;
  onChange: (entity: ManualConceptInput) => void;
  onRemove: () => void;
  disabled?: boolean;
}

function EntityEditor({ entity, onChange, onRemove, disabled }: EntityEditorProps) {
  return (
    <div className="p-3 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <User size={12} className="text-[var(--text-muted)]" />
          <span className="text-xs text-[var(--text-muted)]">Entity</span>
        </div>
        <button 
          onClick={onRemove} 
          disabled={disabled}
          className="text-[var(--text-muted)] hover:text-[var(--error)] disabled:opacity-50"
        >
          <Trash2 size={12} />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <input
          type="text"
          value={entity.name}
          onChange={(e) => onChange({ ...entity, name: e.target.value })}
          placeholder="Entity name *"
          disabled={disabled}
          className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
        />
        <select
          value={entity.entity_type || "Thing"}
          onChange={(e) => onChange({ ...entity, entity_type: e.target.value })}
          disabled={disabled}
          className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
        >
          {ENTITY_TYPES.map(type => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>

      <textarea
        value={entity.description}
        onChange={(e) => onChange({ ...entity, description: e.target.value })}
        placeholder="Entity description *"
        disabled={disabled}
        className="w-full h-16 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-2 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] resize-none focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
      />
    </div>
  );
}

// ============================================================================
// Facet Editor
// ============================================================================

function FacetPointEditor({ point, onChange, onRemove, disabled }: {
  point: ManualFacetPointInput;
  onChange: (p: ManualFacetPointInput) => void;
  onRemove: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="p-2 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[var(--text-muted)] flex items-center gap-1"><FileText size={10} /> FacetPoint</span>
        <button onClick={onRemove} disabled={disabled} className="text-[var(--text-muted)] hover:text-[var(--error)] disabled:opacity-50"><Trash2 size={10} /></button>
      </div>
      <input type="text" value={point.search_text} onChange={(e) => onChange({ ...point, search_text: e.target.value })}
        placeholder="Search text (indexed) *" disabled={disabled}
        className="w-full bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded px-2 py-1 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50" />
      <textarea value={point.description || ""} onChange={(e) => onChange({ ...point, description: e.target.value })}
        placeholder="Description (optional, for detail)" disabled={disabled}
        className="w-full h-10 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded p-2 text-[10px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] resize-none focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50" />
      <input type="text" value={point.display_only || ""} onChange={(e) => onChange({ ...point, display_only: e.target.value || undefined })}
        placeholder="Display text (not indexed, for full content)" disabled={disabled}
        className="w-full bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded px-2 py-1 text-[10px] text-[var(--text-muted)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50" />
    </div>
  );
}

interface FacetEditorProps {
  facet: ManualFacetInput;
  onChange: (facet: ManualFacetInput) => void;
  onRemove: () => void;
  disabled?: boolean;
}

function FacetEditor({ facet, onChange, onRemove, disabled }: FacetEditorProps) {
  const [showPoints, setShowPoints] = useState(false);

  const addPoint = () => {
    onChange({ ...facet, points: [...(facet.points || []), { search_text: "" }] });
  };

  return (
    <div className="p-3 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Tag size={12} className="text-[var(--text-muted)]" />
          <span className="text-xs text-[var(--text-muted)]">Facet</span>
          {facet.points && facet.points.length > 0 && (
            <span className="text-[9px] text-[var(--text-muted)] bg-[var(--bg-elevated)] px-1 rounded">{facet.points.length} points</span>
          )}
        </div>
        <button onClick={onRemove} disabled={disabled} className="text-[var(--text-muted)] hover:text-[var(--error)] disabled:opacity-50">
          <Trash2 size={12} />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <select value={facet.facet_type || "chapter"} onChange={(e) => onChange({ ...facet, facet_type: e.target.value })}
          disabled={disabled}
          className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50">
          {FACET_TYPES.map(type => (<option key={type} value={type}>{type}</option>))}
        </select>
        <input type="text" value={facet.search_text} onChange={(e) => onChange({ ...facet, search_text: e.target.value })}
          placeholder="Search text (indexed) *" disabled={disabled}
          className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50" />
      </div>

      <textarea value={facet.description || ""} onChange={(e) => onChange({ ...facet, description: e.target.value })}
        placeholder="Description (optional)" disabled={disabled}
        className="w-full h-12 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-2 text-xs text-[var(--text-primary)] placeholder:text-[var(--text-muted)] resize-none focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50" />

      <input type="text" value={facet.display_only || ""} onChange={(e) => onChange({ ...facet, display_only: e.target.value || undefined })}
        placeholder="Display text (not indexed, e.g. full chapter text)" disabled={disabled}
        className="w-full bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded px-2 py-1 text-[10px] text-[var(--text-muted)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50" />

      {/* FacetPoints */}
      <div>
        <button onClick={() => setShowPoints(!showPoints)} disabled={disabled}
          className="flex items-center gap-1 text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] disabled:opacity-50">
          {showPoints ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          FacetPoints ({facet.points?.length || 0})
        </button>
        {showPoints && (
          <div className="mt-2 space-y-2 ml-3">
            {facet.points?.map((point, i) => (
              <FacetPointEditor key={i} point={point} disabled={disabled}
                onChange={(p) => { const pts = [...(facet.points || [])]; pts[i] = p; onChange({ ...facet, points: pts }); }}
                onRemove={() => { const pts = [...(facet.points || [])]; pts.splice(i, 1); onChange({ ...facet, points: pts }); }} />
            ))}
            <button onClick={addPoint} disabled={disabled}
              className="w-full py-1 border border-dashed border-[var(--border-subtle)] rounded text-[9px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors disabled:opacity-50">
              + Add FacetPoint
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Episode Editor
// ============================================================================

interface EpisodeEditorProps {
  episode: ManualEpisodeInput;
  index: number;
  onChange: (episode: ManualEpisodeInput) => void;
  onRemove: () => void;
  canRemove: boolean;
  disabled?: boolean;
  showValidation?: boolean;
}

function EpisodeEditor({ episode, index, onChange, onRemove, canRemove, disabled, showValidation = false }: EpisodeEditorProps) {
  const [expanded, setExpanded] = useState(true);
  const [showFacets, setShowFacets] = useState(false);
  const [showEntities, setShowEntities] = useState(false);

  const addFacet = () => {
    onChange({
      ...episode,
      facets: [...(episode.facets || []), { facet_type: "chapter", search_text: "" }]
    });
    setShowFacets(true);
  };

  const addEntity = () => {
    onChange({
      ...episode,
      entities: [...(episode.entities || []), { name: "", description: "", entity_type: "Thing" }]
    });
    setShowEntities(true);
  };

  const updateFacet = (facetIndex: number, facet: ManualFacetInput) => {
    const newFacets = [...(episode.facets || [])];
    newFacets[facetIndex] = facet;
    onChange({ ...episode, facets: newFacets });
  };

  const removeFacet = (facetIndex: number) => {
    onChange({
      ...episode,
      facets: (episode.facets || []).filter((_, i) => i !== facetIndex)
    });
  };

  const updateEntity = (entityIndex: number, entity: ManualConceptInput) => {
    const newEntities = [...(episode.entities || [])];
    newEntities[entityIndex] = entity;
    onChange({ ...episode, entities: newEntities });
  };

  const removeEntity = (entityIndex: number) => {
    onChange({
      ...episode,
      entities: (episode.entities || []).filter((_, i) => i !== entityIndex)
    });
  };

  const isValid = episode.name.trim() && episode.summary.trim();
  const showWarningBorder = showValidation && !isValid;

  return (
    <div className={cn(
      "bg-[var(--bg-surface)] border rounded transition-colors",
      showWarningBorder ? "border-[var(--warning)]" : "border-[var(--border-subtle)]"
    )}>
      {/* Header */}
      <div 
        className="flex items-center justify-between p-3 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          {expanded ? (
            <ChevronDown size={14} className="text-[var(--text-muted)]" />
          ) : (
            <ChevronRight size={14} className="text-[var(--text-muted)]" />
          )}
          <BookOpen size={14} className="text-[var(--text-muted)]" />
          <span className="text-xs text-[var(--text-secondary)]">
            Episode {index + 1}
            {episode.name && <span className="text-[var(--text-primary)]"> — {episode.name}</span>}
          </span>
          {isValid && <Check size={12} className="text-[var(--success)]" />}
        </div>
        {canRemove && (
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
            disabled={disabled}
            className="text-[var(--text-muted)] hover:text-[var(--error)] disabled:opacity-50"
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>

      {/* Content */}
      {expanded && (
        <div className="p-3 pt-0 space-y-4">
          {/* Basic Fields */}
          <div className="space-y-3">
            <input
              type="text"
              value={episode.name}
              onChange={(e) => onChange({ ...episode, name: e.target.value })}
              placeholder="Episode name * (e.g., 'Q4 Planning Meeting')"
              disabled={disabled}
              className="w-full bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded px-3 py-2 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
            />
            
            <textarea
              value={episode.summary}
              onChange={(e) => onChange({ ...episode, summary: e.target.value })}
              placeholder="Episode summary * (indexed for search)"
              disabled={disabled}
              className="w-full h-24 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded p-3 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] resize-none focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
            />

            <textarea
              value={episode.display_only || ""}
              onChange={(e) => onChange({ ...episode, display_only: e.target.value || undefined })}
              placeholder="Display text (not indexed, e.g. full document text for display purposes)"
              disabled={disabled}
              className="w-full h-16 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded p-3 text-[10px] text-[var(--text-muted)] placeholder:text-[var(--text-muted)] resize-none focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
            />

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider block mb-1">
                  Memory Type
                </label>
                <select
                  value={episode.memory_type || "episodic"}
                  onChange={(e) => onChange({ ...episode, memory_type: e.target.value })}
                  disabled={disabled}
                  className="w-full bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
                >
                  <option value="episodic">Episodic (event)</option>
                  <option value="atomic">Atomic (single fact)</option>
                </select>
              </div>
              <div>
                <label className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider block mb-1">
                  Status
                </label>
                <select
                  value={episode.status || "open"}
                  onChange={(e) => onChange({ ...episode, status: e.target.value })}
                  disabled={disabled}
                  className="w-full bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded px-2 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--text-muted)] disabled:opacity-50"
                >
                  <option value="open">Open</option>
                  <option value="closed">Closed</option>
                </select>
              </div>
            </div>
          </div>

          {/* Facets Section */}
          <div>
            <button
              onClick={() => setShowFacets(!showFacets)}
              className="flex items-center gap-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] mb-2"
            >
              {showFacets ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <Tag size={12} />
              Facets ({episode.facets?.length || 0})
            </button>
            
            {showFacets && (
              <div className="space-y-2 ml-4">
                {episode.facets?.map((facet, i) => (
                  <FacetEditor
                    key={i}
                    facet={facet}
                    onChange={(f) => updateFacet(i, f)}
                    onRemove={() => removeFacet(i)}
                    disabled={disabled}
                  />
                ))}
                <button
                  onClick={addFacet}
                  disabled={disabled}
                  className="w-full py-1.5 border border-dashed border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors disabled:opacity-50"
                >
                  + Add Facet
                </button>
              </div>
            )}
          </div>

          {/* Entities Section */}
          <div>
            <button
              onClick={() => setShowEntities(!showEntities)}
              className="flex items-center gap-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] mb-2"
            >
              {showEntities ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <User size={12} />
              Entities ({episode.entities?.length || 0})
            </button>
            
            {showEntities && (
              <div className="space-y-2 ml-4">
                {episode.entities?.map((entity, i) => (
                  <EntityEditor
                    key={i}
                    entity={entity}
                    onChange={(e) => updateEntity(i, e)}
                    onRemove={() => removeEntity(i)}
                    disabled={disabled}
                  />
                ))}
                <button
                  onClick={addEntity}
                  disabled={disabled}
                  className="w-full py-1.5 border border-dashed border-[var(--border-subtle)] rounded text-[10px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors disabled:opacity-50"
                >
                  + Add Entity
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export function ManualIngestPage() {
  const { datasetContext } = useUIStore();
  const manualIngest = useManualIngest();

  // Form state
  const [episodes, setEpisodes] = useState<ManualEpisodeInput[]>([
    { name: "", summary: "", memory_type: "episodic", status: "open" }
  ]);
  const [embedTriplets, setEmbedTriplets] = useState(true);
  const [hasAttemptedSubmit, setHasAttemptedSubmit] = useState(false);

  const isProcessing = manualIngest.isPending;
  const datasetName = datasetContext.datasetName || "main_dataset";

  // Validation
  const isValid = episodes.every(ep => ep.name.trim() && ep.summary.trim());
  const validEpisodesCount = episodes.filter(ep => ep.name.trim() && ep.summary.trim()).length;

  const addEpisode = () => {
    setEpisodes([...episodes, { name: "", summary: "", memory_type: "episodic", status: "open" }]);
  };

  const updateEpisode = (index: number, episode: ManualEpisodeInput) => {
    const newEpisodes = [...episodes];
    newEpisodes[index] = episode;
    setEpisodes(newEpisodes);
  };

  const removeEpisode = (index: number) => {
    if (episodes.length > 1) {
      setEpisodes(episodes.filter((_, i) => i !== index));
    }
  };

  const handleSubmit = async () => {
    setHasAttemptedSubmit(true);
    
    if (!isValid) {
      toast.error("Please fill in all required fields (name and summary) for each episode");
      return;
    }

    try {
      const result = await manualIngest.mutateAsync({
        episodes: episodes.map(ep => ({
          ...ep,
          facets: ep.facets?.filter(f => f.search_text.trim()).map(f => ({
            ...f,
            points: f.points?.filter(p => p.search_text.trim()) || undefined,
          })) || undefined,
          entities: ep.entities?.filter(e => e.name.trim() && e.description.trim()) || undefined,
        })),
        dataset_name: datasetName,
        embed_triplets: embedTriplets,
      });

      toast.success(
        `Created ${result.episodes_created} episodes, ${result.facets_created} facets, ${result.entities_created} entities`
      );

      // Reset form
      setEpisodes([{ name: "", summary: "", memory_type: "episodic", status: "open" }]);
      setHasAttemptedSubmit(false);

    } catch (error) {
      const actionableMessage = getActionableErrorMessage(error instanceof Error ? error : "Manual ingest failed");
      toast.error(actionableMessage);
      console.error("Manual ingest error:", error);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-lg font-medium text-[var(--text-primary)]">Structured Import</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Directly define and import structured episodes, entities, and facets into your knowledge graph. Bypasses LLM extraction for precise manual control over memory structure.
        </p>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Episode Editor */}
        <div className="col-span-7 space-y-4">
          {/* Validation Warning - only show after user attempts to submit */}
          {hasAttemptedSubmit && !isValid && episodes.length > 0 && (
            <div className="flex items-center gap-2 p-3 bg-[var(--warning)]/10 border border-[var(--warning)]/30 rounded text-xs text-[var(--warning)]">
              <AlertCircle size={14} />
              <span>
                {validEpisodesCount} of {episodes.length} episodes have required fields filled
              </span>
            </div>
          )}

          {/* Episodes */}
          {episodes.map((episode, index) => (
            <EpisodeEditor
              key={index}
              episode={episode}
              index={index}
              onChange={(ep) => updateEpisode(index, ep)}
              onRemove={() => removeEpisode(index)}
              canRemove={episodes.length > 1}
              disabled={isProcessing}
              showValidation={hasAttemptedSubmit}
            />
          ))}

          {/* Add Episode Button */}
          <button
            onClick={addEpisode}
            disabled={isProcessing}
            className="w-full py-2 border border-dashed border-[var(--border-subtle)] rounded text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            <Plus size={14} />
            Add Episode
          </button>

          {/* Options */}
          <div className="flex items-center justify-between p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <div>
              <span className="text-xs text-[var(--text-secondary)]">Embed Triplets</span>
              <p className="text-[10px] text-[var(--text-muted)]">Create embeddings for relationships</p>
            </div>
            <Switch
              checked={embedTriplets}
              onCheckedChange={setEmbedTriplets}
              disabled={isProcessing}
            />
          </div>

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={isProcessing || !isValid}
            className={cn(
              "w-full py-2.5 bg-[var(--text-primary)] text-[var(--bg-base)] text-sm font-medium rounded transition-opacity flex items-center justify-center gap-2",
              (isProcessing || !isValid) ? "opacity-50 cursor-not-allowed" : "hover:opacity-90"
            )}
          >
            {isProcessing && <Loader2 size={14} className="animate-spin" />}
            {isProcessing ? "Ingesting..." : `Submit ${validEpisodesCount} Episode${validEpisodesCount !== 1 ? "s" : ""}`}
          </button>

          {/* Dataset Info */}
          <p className="text-xs text-[var(--text-muted)]">
            Target: <span className="text-[var(--text-secondary)]">{datasetName}</span>
          </p>
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

          {/* Episode Schema */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Episode Schema
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded divide-y divide-[var(--border-subtle)]">
              {EPISODE_SCHEMA.map((param) => (
                <div key={param.name} className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <code className="text-xs text-[var(--text-primary)]">{param.name}</code>
                    {param.required && <span className="text-[10px] text-[var(--error)]">required</span>}
                  </div>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{param.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Use Cases */}
          <div>
            <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider block mb-2">
              Use Cases
            </label>
            <div className="bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded p-3 space-y-1 text-xs text-[var(--text-secondary)]">
              <p>• User preferences and settings</p>
              <p>• Known facts and constraints</p>
              <p>• Corrections to extracted data</p>
              <p>• Domain-specific knowledge</p>
              <p>• Meeting notes and decisions</p>
            </div>
          </div>

          {/* Structure Info */}
          <div className="p-3 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded">
            <p className="text-xs text-[var(--text-secondary)]">
              <span className="text-[var(--text-primary)]">Structure:</span> Episode → Facets → FacetPoints, plus Entities
            </p>
            <div className="mt-2 flex flex-wrap gap-1">
              <span className="px-2 py-0.5 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded border border-[var(--border-subtle)] flex items-center gap-1">
                <BookOpen size={10} /> Episode
              </span>
              <span className="px-2 py-0.5 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded border border-[var(--border-subtle)] flex items-center gap-1">
                <Tag size={10} /> Facet
              </span>
              <span className="px-2 py-0.5 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded border border-[var(--border-subtle)] flex items-center gap-1">
                <FileText size={10} /> FacetPoint
              </span>
              <span className="px-2 py-0.5 text-[10px] bg-[var(--bg-elevated)] text-[var(--text-muted)] rounded border border-[var(--border-subtle)] flex items-center gap-1">
                <User size={10} /> Entity
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
