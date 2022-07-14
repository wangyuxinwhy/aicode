from __future__ import annotations

import random

from aicode.core import Notebook


def shuffle_md(notebook: Notebook) -> list[str]:
    md_cells = []
    cell_ids = []
    for cell in notebook.cells:
        if cell.is_markdown:
            md_cells.append(cell.id)
        else:
            if len(md_cells) > 1:
                random.shuffle(md_cells)
            cell_ids.extend(md_cells)
            md_cells = []
            cell_ids.append(cell.id)
    if md_cells:
        random.shuffle(md_cells)
        cell_ids.extend(md_cells)
    return cell_ids
