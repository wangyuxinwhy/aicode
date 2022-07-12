from __future__ import annotations

from bisect import bisect


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth: list[list[str]], predictions: list[list[str]]):
    total_inversions = 0
    total_2max = (
        0  # twice the maximum possible inversions across all instances
    )
    assert len(ground_truth) == len(predictions)
    for gt, pred in zip(ground_truth, predictions):
        assert len(gt) == len(pred)
        ranks = [
            gt.index(x) for x in pred
        ]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max
