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

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Cell):
            return self.id == __o.id
        return False

    def __hash__(self) -> int:
        return hash(self.id)


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
