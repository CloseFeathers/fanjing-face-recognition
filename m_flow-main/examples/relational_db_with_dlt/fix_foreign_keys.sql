-- M-flow: Foreign Key Corrections for dlt-generated schemas
-- Applied after initial data load to establish proper relationships
--
-- dlt (data load tool) creates tables without foreign keys by default.
-- This script adds the necessary constraints for M-flow's graph builder
-- to correctly identify entity relationships.

-- Enable WAL mode for better concurrent access
PRAGMA journal_mode=WAL;

-- Fix: Link pokemon_details to pokemon_list
ALTER TABLE pokedex.pokemon_details
    ADD CONSTRAINT fk_pokemon_name
    FOREIGN KEY (name)
    REFERENCES pokedex.pokemon_list(name)
    ON DELETE CASCADE;

-- Create indices for M-flow graph traversal performance
CREATE INDEX IF NOT EXISTS idx_pokemon_name ON pokedex.pokemon_details(name);
CREATE INDEX IF NOT EXISTS idx_pokemon_type ON pokedex.pokemon_details(types);

-- Verify constraints
SELECT sql FROM sqlite_master WHERE type='table' AND name LIKE 'pokemon%';
