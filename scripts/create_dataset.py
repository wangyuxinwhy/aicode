import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Union

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from aicode.core import Cell, Notebook
from aicode.utils import get_notebook_id_cell_order_map

logger = logging.getLogger(__name__)


def read_notebook_json_data(json_dir: Path, notebook_id: str):
    file_path = json_dir / f'{notebook_id}.json'
    with open(file_path) as f:
        data = json.load(f)
    return data


def main(dataset_dir: Union[Path, str]):
    dataset_dir = Path(dataset_dir).resolve()
    logger.info(f'Construct notebooks from {dataset_dir}')

    notebook_id_cell_order_map = get_notebook_id_cell_order_map(
        dataset_dir / 'train_orders.csv'
    )
    df_ancestors = pd.read_csv(dataset_dir / 'train_ancestors.csv').fillna('')
    notebook_id_ancestor_id_map = {
        notebook_id: ancestor_id
        for notebook_id, ancestor_id in zip(
            df_ancestors['id'], df_ancestors['ancestor_id']
        )
    }
    notebook_id_parent_id_map = {
        notebook_id: parent_id
        for notebook_id, parent_id in zip(
            df_ancestors['id'], df_ancestors['parent_id']
        )
    }

    def construct_notebook_from_id(notebook_id: str):
        cell_orders = notebook_id_cell_order_map[notebook_id]
        notebook_data = read_notebook_json_data(dataset_dir / 'train', notebook_id)
        cell_id_type_map = notebook_data['cell_type']
        cell_id_source_map = notebook_data['source']

        cells = []
        for rank, cell_id in enumerate(cell_orders):
            cell = Cell(
                id=cell_id,
                is_code=cell_id_type_map[cell_id] == 'code',
                source=cell_id_source_map[cell_id],
                rank=rank,
            )
            cells.append(cell)

        notebook = Notebook(
            id=notebook_id,
            ancestor_id=notebook_id_ancestor_id_map[notebook_id],
            parent_id=notebook_id_parent_id_map[notebook_id] or None,
            cells=cells,
        )
        return notebook

    notebook_ids = list(notebook_id_cell_order_map.keys())
    notebooks = [
        construct_notebook_from_id(notebook_id)
        for notebook_id in tqdm(
            notebook_ids, desc='Construct Notebook', unit='nb'
        )
    ]
    logger.info(f'Construct notebooks done, got {len(notebooks)} notebooks.')
    
    records = [asdict(notebook) for notebook in notebooks]

    logger.info(f'Create dataset and save to {str(dataset_dir / "aicode")}, please wait...')
    dataset = Dataset.from_pandas(pd.DataFrame(records))
    dataset.save_to_disk(str(dataset_dir / 'aicode'))
    
    logger.info(f'Create debug dataset and save to {str(dataset_dir / "aicode-debug")}, please wait...')
    dataset_debug = dataset.select(range(2000)) 
    dataset_debug.save_to_disk(str(dataset_dir / 'aicode-debug'))


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        'dataset_dir', type=str, help='Path to the dataset directory'
    )

    args = argparser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
    )
    main(args.dataset_dir)
