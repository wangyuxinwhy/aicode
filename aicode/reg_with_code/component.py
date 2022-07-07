from dataclasses import dataclass
from typing import cast

from cotrain import Trainer, after, op

from aicode.metric import kendall_tau
from aicode.reg_with_code.data import RegSample


@dataclass
class RegResult:
    notebook_id: str
    cell_id: str
    rank: int
    pred_score: float


class Scorer:
    def __init__(self) -> None:
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
        assert len(trainer.valid_dataloader.dataset) == len(
            self.pred_scores
        )   # type: ignore

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

        _map = {}
        for result in results:
            if result.notebook_id not in _map:
                _map[result.notebook_id] = {'pred': [], 'ground_truth': []}
            _map[result.notebook_id]['pred'].append(
                (result.cell_id, result.pred_score)
            )
            _map[result.notebook_id]['ground_truth'].append(
                (result.cell_id, result.rank)
            )
        for v in _map.values():
            v['pred'] = sorted(v['pred'], key=lambda x: x[1])
            v['ground_truth'] = sorted(v['ground_truth'], key=lambda x: x[1])
        gt = [[i[0] for i in v['ground_truth']] for v in _map.values()]
        pred = [[i[0] for i in v['pred']] for v in _map.values()]
        score = kendall_tau(gt, pred)
        self.pred_scores: list[float] = []
        return op.dict_update, {'score': score}
