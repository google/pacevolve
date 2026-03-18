"""Public-data MULTI-evolve extrapolation baseline."""

# RegexTagCustomPruningAlgorithmStart

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


def _parse_mutation(mutation: str) -> List[str]:
    text = str(mutation).strip()
    if not text or text == "WT" or text.lower() == "nan":
        return []
    return [token for token in text.replace(":", "/").split("/") if token and token != "WT"]


def _sorted_pair(token_a: str, token_b: str) -> Tuple[str, str]:
    return tuple(sorted((token_a, token_b)))


class SparseMutationRidge:
    """Ridge regression over sparse single and pairwise mutation indicators."""

    def __init__(self, alpha: float = 2.0, pair_scale: float = 0.75):
        self.alpha = alpha
        self.pair_scale = pair_scale
        self.single_index: Dict[str, int] = {}
        self.pair_index: Dict[Tuple[str, str], int] = {}
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)

    def fit(self, train_df: pd.DataFrame) -> None:
        parsed = [_parse_mutation(mutation) for mutation in train_df["mutant"].tolist()]

        singles = sorted({token for tokens in parsed for token in tokens})
        self.single_index = {token: idx for idx, token in enumerate(singles)}

        observed_pairs = set()
        for tokens in parsed:
            if len(tokens) == 2:
                observed_pairs.add(_sorted_pair(tokens[0], tokens[1]))
        self.pair_index = {pair: idx for idx, pair in enumerate(sorted(observed_pairs))}

        x_train = self.transform_mutations(parsed)
        y_train = train_df["fitness"].to_numpy(dtype=float)

        load_counts = train_df["num_mutations"].value_counts().to_dict()
        sample_weight = np.array(
            [1.0 / max(1, load_counts.get(int(load), 1)) for load in train_df["num_mutations"]],
            dtype=float,
        )
        sample_weight *= len(sample_weight) / np.sum(sample_weight)
        self.model.fit(x_train, y_train, sample_weight=sample_weight)

    def transform_mutations(self, parsed_mutations: List[List[str]]) -> sparse.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        load_values = np.zeros((len(parsed_mutations), 2), dtype=float)

        single_offset = 0
        pair_offset = len(self.single_index)
        load_offset = pair_offset + len(self.pair_index)

        for row_idx, tokens in enumerate(parsed_mutations):
            for token in tokens:
                if token in self.single_index:
                    rows.append(row_idx)
                    cols.append(single_offset + self.single_index[token])
                    data.append(1.0)

            all_pairs = list(combinations(sorted(tokens), 2))
            if all_pairs:
                weight = self.pair_scale / len(all_pairs)
                for pair in all_pairs:
                    if pair in self.pair_index:
                        rows.append(row_idx)
                        cols.append(pair_offset + self.pair_index[pair])
                        data.append(weight)

            load_values[row_idx, 0] = float(len(tokens))
            load_values[row_idx, 1] = float(len(all_pairs))

        width = load_offset + load_values.shape[1]
        mutation_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(parsed_mutations), width))
        load_matrix = sparse.csr_matrix(load_values)
        return sparse.hstack([mutation_matrix[:, :load_offset], load_matrix], format="csr")

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        parsed = [_parse_mutation(mutation) for mutation in test_df["mutant"].tolist()]
        x_test = self.transform_mutations(parsed)
        return self.model.predict(x_test)


def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_context: dict) -> np.ndarray:
    del dataset_context
    model = SparseMutationRidge(alpha=2.0, pair_scale=0.75)
    model.fit(train_df)
    return model.predict(test_df)

# RegexTagCustomPruningAlgorithmEnd

__all__ = ["fit_and_predict"]
