from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from cotrain import Trainer, after, before, op

from aicode.core import Notebook
from aicode.metric import kendall_tau
from aicode.reg_with_code.data import RegSample, RegWithCodeDataset

logger = logging.getLogger(__name__)


@dataclass
class RegResult:
    notebook_id: str
    cell_id: str
    rank: int
    pred_score: float


class Scorer:
    def __init__(self, notebooks: list[Notebook]) -> None:
        self.notebooks = notebooks
        self.groud_truth_cell_orders_map = {
            notebook.id: [cell.id for cell in notebook.cells]
            for notebook in self.notebooks
        }
        self.pred_scores: list[float] = []
        self.results: list[RegResult] = []

    @after(Trainer.run_valid_batch)
    def collect_result(self, trainer: Trainer) -> None:
        pred = trainer.stage.batch_output['pred_scores']
        self.pred_scores.extend(pred.detach().cpu().numpy().ravel().tolist())

    @after(Trainer.collect_stage_metrics)
    def score(self, trainer: Trainer):
        if trainer.in_train_stage:
            return

        if trainer.valid_dataloader is None:
            raise ValueError('valid_dataloader is not specified')
        assert len(trainer.valid_dataloader.dataset) == len(  # type: ignore
            self.pred_scores
        )

        results: list[RegResult] = []
        for reg_sample, pred_score in zip(trainer.valid_dataloader.dataset, self.pred_scores):  # type: ignore
            reg_sample = cast(RegSample, reg_sample)
            results.append(
                RegResult(
                    notebook_id=reg_sample.notebook_id,
                    cell_id=reg_sample.cell_id,
                    rank=reg_sample.rank,
                    pred_score=pred_score,
                )
            )

        pred_cell_orders_map = self.construct_pred_cell_orders(results)

        preds = [
            pred_cell_orders_map[k]
            for k in self.groud_truth_cell_orders_map.keys()
        ]
        targets = [v for _, v in self.groud_truth_cell_orders_map.items()]
        score = kendall_tau(targets, preds)
        self.refresh()
        return op.dict_update, {'score': score}

    def refresh(self) -> None:
        self.pred_scores: list[float] = []

    def construct_pred_cell_orders(
        self, results: list[RegResult]
    ) -> dict[str, list[str]]:

        pred_cell_orders_map = {}
        for result in results:
            if result.notebook_id not in pred_cell_orders_map:
                pred_cell_orders_map[result.notebook_id] = []
            pred_cell_orders_map[result.notebook_id].append(
                (result.cell_id, result.pred_score)
            )
        for notebook in self.notebooks:
            code_cells = [cell.id for cell in notebook.cells if cell.is_code]
            num_codes = len(code_cells)
            code_cells = [
                (cell, idx / num_codes) for idx, cell in enumerate(code_cells)
            ]
            pred_cell_orders_map[notebook.id].extend(code_cells)
        for k, v in pred_cell_orders_map.items():
            pred_cell_orders_map[k] = [
                i[0] for i in sorted(v, key=lambda x: x[1])
            ]
        return pred_cell_orders_map


class RegWithCodeDatasetRefresh:
    @before(Trainer.run_epoch)
    def print_first_sample(self, trainer: Trainer) -> None:
        print(trainer.train_dataloader.dataset[0])
        print(trainer.train_dataloader.dataset[1])

    @after(Trainer.run_epoch)
    def refresh(self, trainer: Trainer) -> None:
        logger.info('Refresh dataset')
        dataset = trainer.train_dataloader.dataset
        dataset = cast(RegWithCodeDataset, dataset)
        dataset.refresh()
