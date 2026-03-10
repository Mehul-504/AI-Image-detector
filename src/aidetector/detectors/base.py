from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol

from aidetector.schemas import Category, CategoryScore, MediaType, OverrideHints


@dataclass(frozen=True)
class DetectionContext:
    media_type: MediaType
    media_uri: str | None = None
    provided_scores: Dict[Category, CategoryScore] = field(default_factory=dict)
    override_hints: OverrideHints = field(default_factory=OverrideHints)


@dataclass(frozen=True)
class DetectionResult:
    scores: Dict[Category, CategoryScore] = field(default_factory=dict)
    override_hints: OverrideHints = field(default_factory=OverrideHints)


class Detector(Protocol):
    name: str

    def run(self, context: DetectionContext) -> DetectionResult:
        """Extract category scores and optional override hints."""
