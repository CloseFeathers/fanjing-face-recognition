#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# M-flow helper — Convert standalone workflow triggers to workflow_call.
#
# Purpose: Ensures the listed workflows can only run when invoked from the
#          master test-suites orchestrator, preventing accidental
#          independent execution that bypasses the CI gate.
#
# Usage:   bash .github/workflows/disable_independent_workflows.sh
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Workflow files that should only be invoked via test_suites.yml
readonly TARGET_WORKFLOWS=(
  test_chromadb.yml
  test_weaviate.yml
  test_kuzu.yml
  test_multimetric_qa_eval_run.yaml
  test_graphrag_vs_rag_notebook.yml
  test_llms.yml
  test_multimedia_example.yaml
  test_deduplication.yml
  test_eval_framework.yml
  test_descriptive_graph_metrics.yml
  test_llama_index_m_flow_integration_notebook.yml
  test_m_flow_llama_index_notebook.yml
  test_m_flow_multimedia_notebook.yml
  test_m_flow_server_start.yml
  test_telemetry.yml
  test_neo4j.yml
  test_pgvector.yml
  test_ollama.yml
  test_notebook.yml
  test_simple_example.yml
  test_code_graph_example.yml
)

converted=0
skipped=0

for wf in "${TARGET_WORKFLOWS[@]}"; do
  if [[ ! -f "$wf" ]]; then
    echo "⏭  $wf — not found, skipping"
    (( skipped++ )) || true
    continue
  fi

  # Already converted?
  if grep -q 'workflow_call:' "$wf"; then
    echo "✔  $wf — already uses workflow_call, skipping"
    (( skipped++ )) || true
    continue
  fi

  # Locate the line number of the top-level `on:` key
  on_line="$(grep -n '^on:' "$wf" | head -1 | cut -d: -f1)"
  if [[ -z "$on_line" ]]; then
    echo "⚠  $wf — no top-level 'on:' key found, skipping"
    (( skipped++ )) || true
    continue
  fi

  # Find the next top-level section after `on:`
  next_section="$(awk -v start="$on_line" \
    'NR > start && /^[a-z]/ { print NR; exit }' "$wf")"
  if [[ -z "$next_section" ]]; then
    next_section=$(( $(wc -l < "$wf") + 1 ))
  fi

  # Rewrite: keep everything before `on:`, inject workflow_call, append rest
  {
    head -n $(( on_line - 1 )) "$wf"
    printf 'on:\n  workflow_call:\n    secrets:\n      inherit: true\n'
    tail -n +"$next_section" "$wf"
  } > "${wf}.tmp"

  mv "${wf}.tmp" "$wf"
  echo "✅ $wf — converted to workflow_call"
  (( converted++ )) || true
done

echo ""
echo "Done — converted: ${converted}, skipped: ${skipped}"
