/**
 * Default LLM prompts used throughout the application.
 * These serve as fallback values and can be customized by users.
 */

export const DEFAULT_PROMPTS = {
  /**
   * Knowledge Graph Extraction - Used during memorization
   * Controls how entities and relationships are extracted from text
   */
  graphExtraction: `You are a structured information extraction system that produces knowledge graph nodes from text input.

Your task: identify and extract entities (nodes) only. The system handles relationship inference automatically from document structure — do NOT extract edges. Always return edges as an empty array [].

## Node Extraction Rules

### Labeling
- Use fundamental, broad type labels: **"Person"**, **"Organization"**, **"Location"**, **"Date"**, etc.
- Avoid overly specific labels (use "Person" not "Mathematician"; record specifics as properties instead).
- Avoid overly generic labels (do not use "Thing" or "Item").

### Node Identifiers
- Use human-readable names from the text as node IDs — never use numeric IDs.
- When an entity appears under multiple names or pronouns (e.g., "John Doe", "Joe", "he"), always use the most complete form ("John Doe") as the canonical ID.

### Dates and Numbers
- Date entities must have type **"Date"** with format "YYYY-MM-DD" when possible.
- If only partial date information is available, extract what exists (year, month, or both).

### Properties
- Key-value format only.
- Use snake_case for property names.
- Never include escaped quotes inside property values.

## Output Requirements

**Return actual extracted data, not a JSON schema.**
- Correct: \`{"nodes": [{"id": "Marie Curie", "name": "Marie Curie", "type": "Person", "description": "Physicist and chemist"}], "edges": []}\`
- Wrong: \`{"$defs": {...}, "properties": {...}}\` — this is a schema definition, not extracted data.

## Language
Produce output in the same language as the input text.`,

  /**
   * Question Answering - Used during search
   * Controls how the LLM generates answers from retrieved context
   */
  questionAnswering: `Given the context below, provide a concise and direct answer to the question. Keep your response short and focused on the key facts.`,

  /**
   * Procedural Question Answering - Used for how-to queries
   * Controls formatting of step-by-step responses
   */
  proceduralAnswering: `Given the context below, provide step-by-step instructions to answer the question. Format your response as a numbered list of clear, actionable steps.`,

  /**
   * Triplet Completion - Used for knowledge graph queries
   * Controls how triplet relationships are explained
   */
  tripletCompletion: `Based on the knowledge graph data provided, complete the triplet query and explain the relationships found. Be precise about the connections between entities.`,
} as const;

export type PromptKey = keyof typeof DEFAULT_PROMPTS;

/**
 * Get a default prompt by key
 */
export function getDefaultPrompt(key: PromptKey): string {
  return DEFAULT_PROMPTS[key];
}

/**
 * Check if a prompt value differs from default
 */
export function isPromptModified(key: PromptKey, value: string): boolean {
  if (!value) return false;
  return value.trim() !== DEFAULT_PROMPTS[key].trim();
}
