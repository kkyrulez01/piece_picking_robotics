#!/usr/bin/env python3

from dataclasses import replace
import numpy as np

def normalize_sim_scores_per_object(
    candidates_by_object, 
    lower_percentile=5.0,
    upper_percentile=95.0
):
    """
    Normalize Sim-Suction scores independently for each object.

    Uses percentile normalization instead of raw min/max so that a
    single extreme Sim-Suction score does not compress all other
    candidates.

    Values are clipped to [0, 1].
    """
    normalized_by_object = {}

    for object_id, candidates in candidates_by_object.items():
        if not candidates:
            normalized_by_object[object_id] = []
            continue

        raw_scores = np.asarray([candidate.sim_score for candidate in candidates], dtype=np.float64)

        lower_score = float(np.percentile(raw_scores, lower_percentile))
        higher_score = float(np.percentile(raw_scores, upper_percentile))
        score_range = higher_score - lower_score

        normalized_candidates = []
        for candidate in candidates:
            if score_range > 1e-12:
                normalized_score = (candidate.sim_score - lower_score) / score_range
                normalized_score = float(np.clip(normalized_score, 0.0, 1.0))

            else:
                # Sim-Suction cannot meaningfully distinguish candidates
                # on this object
                normalized_score = 0.5

            normalized_candidates.append(
                replace(
                    candidate,
                    sim_score_normalized=normalized_score
                )
            )

        normalized_by_object[object_id] = normalized_candidates

    return normalized_by_object

def calculate_weighted_ranking_scores(
    candidates_by_object,
    sim_suction_weight,
    boundary_weight,
    surface_geometry_weight,
):
    """
    Calculate final candidate ranking from normalized quality scores.

    All input qualities are expected to be in [0, 1].
    """
    weights = [
        (sim_suction_weight, "sim_score_normalized"),
        (boundary_weight, "boundary_quality"),
        (surface_geometry_weight, "surface_quality"),
    ]

    # Ignore disabled scoring terms
    active_weights = [
        (float(weight), attribute) for weight, attribute in weights if weight > 0.0
    ]

    weight_sum = float(np.sum(weight for weight, _ in active_weights))

    ranked_by_object = {}
    for object_id, candidates in candidates_by_object.items():
        ranked_candidates = []

        for candidate in candidates:
            ranking_score = 0.0
            for weight, attribute in active_weights:
                quality = getattr(candidate, attribute)

                if quality is None:
                    raise ValueError(
                        f"Object {object_id}, candidate {candidate.rank}: "
                        f"{attribute} is None."
                    )

                ranking_score += (weight / weight_sum) * quality

            ranked_candidates.append(
                replace(
                    candidate,
                    ranking_score=float(ranking_score)
                )
            )

        ranked_candidates.sort(key=lambda candidate:candidate.ranking_score, reverse=True)
        ranked_by_object[object_id] = [
            replace(candidate, rank=rank) for rank, candidate in enumerate(ranked_candidates, start=1)
        ]

    return ranked_by_object