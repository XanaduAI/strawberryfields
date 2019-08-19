# Copyright 2019 Xanadu Quantum Technologies Inc.
r"""
conftest.py file for use in pytest
"""
import numpy as np
import networkx as nx
import pytest


@pytest.fixture
def adj(dim: int):
    """Creates a fixed adjacency matrix of size ``dim`` with alternating zeros and nonzeros in a
    checkerboard pattern.

    The nonzero value is fixed to 0.5. This can, e.g., prevent issues with square roots,
    and also explore any problems with weighted graph edges.

    Args:
        dim (int): dimension of adjacency matrix

    Returns:
        array: NumPy adjacency matrix of dimension ``dim`` with each element a ``float``.
    """
    a = np.zeros((dim, dim), dtype=float)

    a[1::2, ::2] = 0.5
    a[::2, 1::2] = 0.5

    return a


@pytest.fixture
def graph(dim: int):
    """Creates the NetworkX graph of ``dim`` nodes corresponding to the checkerboard adjacency
    matrix in :func:`adj`

    Args:
        dim (int): number of nodes in graph

    Returns: graph: NetworkX graph of dimension ``dim`` corresponding to the adjacency matrix in
    :func:`adj`
    """

    a = np.zeros((dim, dim), dtype=float)

    a[1::2, ::2] = 0.5
    a[::2, 1::2] = 0.5

    return nx.Graph(a)
