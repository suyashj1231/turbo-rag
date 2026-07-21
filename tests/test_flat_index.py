"""Executable spec for index/flat.py."""

import numpy as np

from turbo_rag.index.flat import FlatIndex


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_finds_exact_match_first():
    vecs = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([1, 1, 0])])
    idx = FlatIndex()
    idx.add(vecs)
    ids, scores = idx.search(unit([0, 1, 0]), k=2)
    assert ids[0] == 1
    assert scores[0] == max(scores)
    assert np.isclose(scores[0], 1.0, atol=1e-5)


def test_scores_sorted_descending():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((100, 16)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = FlatIndex()
    idx.add(vecs)
    ids, scores = idx.search(vecs[7], k=10)
    assert list(scores) == sorted(scores, reverse=True)
    assert ids[0] == 7
    assert len(ids) == 10


def test_memory_bytes_is_float32_matrix():
    vecs = np.zeros((10, 384), dtype=np.float32)
    idx = FlatIndex()
    idx.add(vecs)
    assert idx.memory_bytes() == 10 * 384 * 4
