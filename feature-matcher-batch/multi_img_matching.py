import numpy as np
import torch
from typing import List, Tuple, Any


def cosine_similarity_score(
        query: np.ndarray = None,
        candidates: np.ndarray = None):
    """ Computes the cosine similarity score between the
        query feature and the candidate features.
    @param query: Feature map of dimension
        [1, n_feat_dim] representing the query.
    @param candidates: Feature map of dimension
        [n_candidates, n_feat_dim] representing the candidate for match.
    """
    sim_measure = np.matmul(query, candidates.T)
    return sim_measure


def tensor_reshape(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        if len(data.shape) > 2:
            data = data.squeeze(0)

    if isinstance(data, List):
        if len(data[0].shape) > 2:
            temp = [x.squeeze(0) for x in data]
            data = torch.cat(temp, dim=0)
        else:
            data = torch.cat(data, dim=0)

    return data


def match_query_to_targets(query_feats: List,
                           candidate_feats: List,
                           avg_mode: bool = False) -> Tuple[int, float]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    query_feats, candidate_feats = \
        tensor_reshape(query_feats).to(device), tensor_reshape(candidate_feats).to(device)

    if avg_mode:
        # average query_feats
        query_feats = torch.mean(query_feats, dim=0).unsqueeze(0)

    # compare features
    sim_dist = torch.mm(query_feats, candidate_feats.t())

    max_val, idx = torch.max(sim_dist, dim=1)
    global_max_val, max_index = torch.max(max_val, dim=0)
    match_id = idx[max_index].item()

    return match_id, global_max_val.item()
