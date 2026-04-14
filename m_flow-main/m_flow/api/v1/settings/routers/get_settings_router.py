"""
Settings Router

REST API endpoints for viewing and modifying M-flow system configuration.
Includes LLM provider settings, vector database configuration, and embedding settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import Field

from m_flow.api.DTO import InDTO, OutDTO
from m_flow.llm.config import LLMConfig
from m_flow.adapters.vector.config import VectorConfig
from m_flow.adapters.vector.embeddings.config import EmbeddingConfig

if TYPE_CHECKING:
    from m_flow.auth.models import User


# ---------------------------------------------------------------------------
# Response DTOs
# ---------------------------------------------------------------------------


class LLMSettingsOut(OutDTO, LLMConfig):
    """LLM configuration output wrapper."""

    pass


class VectorDBSettingsOut(OutDTO, VectorConfig):
    """Vector database configuration output wrapper."""

    pass


class EmbeddingSettingsOut(OutDTO):
    """Embedding configuration output wrapper."""

    embedding_provider: Optional[str] = Field(default=None, description="Embedding provider")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model name")
    embedding_dimensions: Optional[int] = Field(default=None, description="Vector dimensions")
    embedding_endpoint: Optional[str] = Field(default=None, description="Custom API endpoint")
    # Note: API key is not returned for security


class SystemSettingsOut(OutDTO):
    """Complete system settings response."""

    llm: LLMSettingsOut
    vector_db: VectorDBSettingsOut
    embedding: Optional[EmbeddingSettingsOut] = None


# ---------------------------------------------------------------------------
# Request DTOs
# ---------------------------------------------------------------------------

LLMProviderType = Literal["openai", "ollama", "anthropic", "gemini", "mistral", "bedrock", "custom"]
VectorDBProviderType = Literal["lancedb", "chromadb", "pgvector"]
EmbeddingProviderType = Literal["openai", "ollama", "fastembed", "azure"]


class LLMSettingsIn(InDTO):
    """LLM configuration update payload."""

    provider: LLMProviderType = Field(..., description="LLM provider identifier")
    model: str = Field(..., description="Model name/identifier")
    api_key: str = Field(default="", description="Provider API key")


class VectorDBSettingsIn(InDTO):
    """Vector database configuration update payload."""

    provider: VectorDBProviderType = Field(..., description="Vector DB provider")
    url: str = Field(default="", description="Connection URL")
    api_key: str = Field(default="", description="Database API key")


class EmbeddingSettingsIn(InDTO):
    """Embedding configuration update payload."""

    provider: EmbeddingProviderType = Field(..., description="Embedding provider")
    model: str = Field(..., description="Model name/identifier")
    dimensions: Optional[int] = Field(default=None, description="Vector dimensions")
    endpoint: Optional[str] = Field(default=None, description="Custom API endpoint")
    api_key: str = Field(default="", description="Provider API key")


class SettingsUpdatePayload(InDTO):
    """Partial settings update request."""

    llm: LLMSettingsIn | None = None
    vector_db: VectorDBSettingsIn | None = None
    embedding: EmbeddingSettingsIn | None = None


# ---------------------------------------------------------------------------
# Auth Dependency
# ---------------------------------------------------------------------------


def _auth():
    """Return authentication dependency."""
    from m_flow.auth.methods import get_authenticated_user

    return get_authenticated_user


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------


def get_settings_router() -> APIRouter:
    """
    Construct settings management router.

    Endpoints:
        GET /  - Retrieve current system settings
        POST / - Update LLM and/or vector database settings
    """
    router = APIRouter()

    @router.get("", response_model=SystemSettingsOut)
    async def retrieve_settings(user: "User" = Depends(_auth())):
        """
        Fetch current system configuration.

        Returns LLM provider settings, vector database connection details,
        and embedding configuration.
        """
        from m_flow.config.settings import get_settings
        from m_flow.adapters.vector.embeddings import get_embedding_config

        base_settings = get_settings()

        # Add embedding config
        emb_cfg = get_embedding_config()
        embedding_out = EmbeddingSettingsOut(
            embedding_provider=emb_cfg.embedding_provider,
            embedding_model=emb_cfg.embedding_model,
            embedding_dimensions=emb_cfg.embedding_dimensions,
            embedding_endpoint=emb_cfg.embedding_endpoint,
        )

        # Convert Pydantic models to dicts for compatibility
        llm_dict = base_settings.llm.model_dump()
        vector_dict = base_settings.vector_db.model_dump()

        return SystemSettingsOut(
            llm=LLMSettingsOut(**{k: v for k, v in llm_dict.items() if k in LLMSettingsOut.model_fields}),
            vector_db=VectorDBSettingsOut(
                **{k: v for k, v in vector_dict.items() if k in VectorDBSettingsOut.model_fields}
            ),
            embedding=embedding_out,
        )

    @router.post("", response_model=None)
    async def update_settings(
        payload: SettingsUpdatePayload,
        user: "User" = Depends(_auth()),
    ):
        """
        Modify system configuration.

        Only supplied sections (llm, vector_db, embedding) are updated.
        Omitted sections retain their current values.
        """
        from m_flow.config.settings import save_llm_config, save_vector_db_config
        from m_flow.config.settings.save_embedding_config import save_embedding_config, EmbeddingConfigDTO

        if payload.llm is not None:
            await save_llm_config(payload.llm)

        if payload.vector_db is not None:
            await save_vector_db_config(payload.vector_db)

        if payload.embedding is not None:
            emb_dto = EmbeddingConfigDTO(
                provider=payload.embedding.provider,
                model=payload.embedding.model,
                dimensions=payload.embedding.dimensions,
                endpoint=payload.embedding.endpoint,
                api_key=payload.embedding.api_key,
            )
            await save_embedding_config(emb_dto)

    return router
