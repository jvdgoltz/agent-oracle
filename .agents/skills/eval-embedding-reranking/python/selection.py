"""Deterministic stored-vector selection for portable evaluation datasets."""

from __future__ import annotations

import hashlib

import numpy as np

KMEANS_ITERATIONS = 10


def stable_rank(seed: int, *parts: str) -> str:
    """Return a stable seeded tie-breaker independent of database row ordering."""
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def cluster_nearest_sessions(
    rows: list[dict], clusters: int, sessions_per_cluster: int, seed: int
) -> list[str]:
    """Choose nearest stored-vector sessions from each deterministic k-means cluster."""
    if not rows or clusters <= 0 or sessions_per_cluster <= 0:
        return []
    ordered = sorted(rows, key=lambda row: stable_rank(seed, row["id"]))
    vectors = np.asarray([row["embedding"] for row in ordered], dtype=np.float32)
    cluster_count = min(clusters, len(ordered))
    centers = vectors[:cluster_count].copy()
    assignments = np.zeros(len(ordered), dtype=np.int64)
    for _ in range(KMEANS_ITERATIONS):
        assignments = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        updated = centers.copy()
        for cluster in range(cluster_count):
            members = vectors[assignments == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.array_equal(updated, centers):
            break
        centers = updated
    selected = []
    for cluster in range(cluster_count):
        members = np.flatnonzero(assignments == cluster)
        if len(members) == 0:
            continue
        distances = ((vectors[members] - centers[cluster]) ** 2).sum(axis=1)
        nearest = sorted(
            range(len(members)),
            key=lambda index: (
                distances[index],
                stable_rank(seed, ordered[members[index]]["id"]),
            ),
        )[:sessions_per_cluster]
        selected.extend(ordered[members[index]]["id"] for index in nearest)
    return selected


def farthest_first_backfill(
    rows: list[dict], selected_ids: list[str], count: int, seed: int
) -> list[str]:
    """Fill a selection with sessions farthest from its existing stored vectors."""
    if count <= 0:
        return []
    ordered = sorted(rows, key=lambda row: stable_rank(seed, row["id"]))
    vectors = np.asarray([row["embedding"] for row in ordered], dtype=np.float32)
    positions = {row["id"]: position for position, row in enumerate(ordered)}
    selected_positions = [positions[session_id] for session_id in selected_ids]
    remaining = set(range(len(ordered))) - set(selected_positions)
    backfill = []
    while remaining and len(backfill) < count:
        if selected_positions:
            candidates = sorted(remaining)
            distances = (
                (vectors[candidates, None, :] - vectors[selected_positions][None, :, :]) ** 2
            ).sum(axis=2)
            chosen = candidates[
                max(
                    range(len(candidates)),
                    key=lambda index: (
                        distances[index].min(),
                        stable_rank(seed, ordered[candidates[index]]["id"]),
                    ),
                )
            ]
        else:
            chosen = min(remaining, key=lambda index: stable_rank(seed, ordered[index]["id"]))
        selected_positions.append(chosen)
        remaining.remove(chosen)
        backfill.append(ordered[chosen]["id"])
    return backfill
