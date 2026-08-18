"""Central configuration for the canonical asthma RAG baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import SourceSpec


@dataclass(frozen=True)
class RagConfig:
    project_root: Path
    sources: tuple[SourceSpec, ...]
    clinical_scope: str
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 5
    embedding_model: str = "nomic-embed-text"
    collection_prefix: str = "pediatric_asthma_guidelines"

    @property
    def source_dir(self) -> Path:
        return self.project_root / "Docs" / "Sources"

    @property
    def artifact_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def chroma_dir(self) -> Path:
        return self.artifact_dir / "chroma_db"

    def validate(self) -> None:
        if not 400 <= self.chunk_size <= 800:
            raise ValueError("chunk_size must stay inside the 400-800 token baseline range")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.sources:
            raise ValueError("At least one approved guideline source is required")


def default_config(project_root: Path | str) -> RagConfig:
    """Return the team's explicit Day-1 baseline; values are hypotheses, not optima."""
    root = Path(project_root).resolve()
    config = RagConfig(
        project_root=root,
        clinical_scope=(
            "Asthma diagnosis and guideline-based management in children and adolescents"
        ),
        sources=(
            SourceSpec(
                file_name="WHO asthma.pdf",
                document="WHO childhood asthma guideline 2026",
                publisher="World Health Organization (WHO)",
                source_url="https://iris.who.int/",
                usage_note=(
                    "Official public WHO guideline; supplied PDF page 4 states "
                    "CC BY-NC-SA 3.0 IGO."
                ),
            ),
            SourceSpec(
                file_name="NICE Asthma.pdf",
                document="NICE asthma guideline NG245",
                publisher="National Institute for Health and Care Excellence (NICE)",
                source_url="https://www.nice.org.uk/guidance/ng245",
                usage_note=(
                    "Official public NICE guidance; reuse remains subject to the "
                    "NICE notice-of-rights terms."
                ),
            ),
        ),
    )
    config.validate()
    return config
