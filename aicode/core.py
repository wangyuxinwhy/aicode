from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Cell:
    id: str
    is_code: bool
    source: str
    rank: int

    @property
    def is_markdown(self) -> bool:
        return not self.is_code


@dataclass
class Notebook:
    id: str
    ancestor_id: str
    parent_id: Optional[str]
    cells: list[Cell]

    @classmethod
    def from_dict(cls, data: dict) -> 'Notebook':
        cells = [Cell(**cell) for cell in data.pop('cells')]
        return cls(**data, cells=cells)
