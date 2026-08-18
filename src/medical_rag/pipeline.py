"""Application service coordinating ingestion, chunking, validation, and manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .chunking import SectionAwareChunker
from .config import RagConfig, default_config
from .ingestion import IngestionService
from .models import Chunk, ParsedGuideline


@dataclass(frozen=True)
class CorpusBuild:
    documents: tuple[ParsedGuideline, ...]
    chunks: tuple[Chunk, ...]
    manifest_path: Path
    corpus_fingerprint: str


class CorpusPipeline:
    """Coordinate core services without owning their parsing/chunking algorithms."""

    def __init__(
        self,
        config: RagConfig,
        ingestion: IngestionService | None = None,
        chunker: SectionAwareChunker | None = None,
    ):
        config.validate()
        self.config = config
        self.ingestion = ingestion or IngestionService()
        self.chunker = chunker or SectionAwareChunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )

    def build(self) -> CorpusBuild:
        documents = []
        chunks = []
        for source in self.config.sources:
            pdf_path = self.config.source_dir / source.file_name
            document = self.ingestion.parse(pdf_path, source)
            documents.append(document)
            chunks.extend(self.chunker.chunk(document, self.config.clinical_scope))

        chunks = self._deduplicate(chunks)
        self.validate_chunks(chunks)
        fingerprint = hashlib.sha1(
            "|".join(chunk.chunk_id for chunk in chunks).encode("utf-8")
        ).hexdigest()[:10]
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.artifact_dir / "chunks_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "config": self.experiment_config(fingerprint),
                    "chunks": [chunk.to_dict() for chunk in chunks],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return CorpusBuild(
            documents=tuple(documents),
            chunks=tuple(chunks),
            manifest_path=manifest_path,
            corpus_fingerprint=fingerprint,
        )

    @staticmethod
    def _deduplicate(chunks: list[Chunk]) -> list[Chunk]:
        """Remove exact within-document duplicates while preserving first provenance."""
        unique = []
        seen = set()
        for chunk in chunks:
            normalized = " ".join(chunk.text.casefold().split())
            key = (chunk.document, normalized)
            if key not in seen:
                unique.append(chunk)
                seen.add(key)
        return unique

    def experiment_config(self, fingerprint: str) -> dict[str, object]:
        return {
            "clinical_scope": self.config.clinical_scope,
            "sources": [source.file_name for source in self.config.sources],
            "chunk_strategy": "section-aware-token",
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_model": self.config.embedding_model,
            "distance_metric": "cosine",
            "top_k": self.config.top_k,
            "corpus_fingerprint": fingerprint,
        }

    @staticmethod
    def validate_chunks(chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks were generated")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Chunk IDs must be globally unique")
        for chunk in chunks:
            if not chunk.text.strip():
                raise ValueError(f"Empty chunk: {chunk.chunk_id}")
            if chunk.page_start < 1 or chunk.page_end < chunk.page_start:
                raise ValueError(f"Invalid page range: {chunk.chunk_id}")
            metadata = chunk.metadata()
            required = {
                "document", "section", "page", "page_start", "page_end",
                "chunk_id", "source_url", "publisher", "token_count",
            }
            missing = required - metadata.keys()
            if missing:
                raise ValueError(f"{chunk.chunk_id} missing metadata: {sorted(missing)}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    build = CorpusPipeline(default_config(project_root)).build()
    print(f"Documents: {len(build.documents)}")
    print(f"Chunks: {len(build.chunks)}")
    print(f"Manifest: {build.manifest_path}")
    print(f"Corpus fingerprint: {build.corpus_fingerprint}")


if __name__ == "__main__":
    main()
