"""Huffman tree for hierarchical softmax.
Frequent words get short codes -> cost ~ log2(Unigram-perplexity) < log2(V).
Implements Eq.3: p(w|wI) = prod_j sigma([[n(w,j+1)==ch(n(w,j))]] * v'_n(w,j)^T v_wI)
"""
from __future__ import annotations
import heapq
import numpy as np

class HuffNode:
    __slots__ = ("freq", "idx", "left", "right", "nid")
    def __init__(self, freq, idx=-1, left=None, right=None, nid=-1):
        self.freq = freq
        self.idx = idx  # word idx if leaf else -1
        self.left = left
        self.right = right
        self.nid = nid  # inner node id

def build_huffman(counts: list[int]):
    """Returns (codes, points): for each word idx, code bits list and inner-node path list.
    Inner node ids are 0..(V-2) in creation order; caller allocates W_out rows for V-1 inners.
    """
    V = len(counts)
    if V == 1:
        return [[0]], [[0]]
    heap: list[tuple] = []
    for i, c in enumerate(counts):
        heap.append((c, i, HuffNode(c, idx=i)))
    heapq.heapify(heap)
    # unique tie-breaker
    counter = V
    inner_nodes: list[HuffNode] = []
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        parent = HuffNode(f1 + f2, idx=-1, left=n1, right=n2, nid=len(inner_nodes))
        inner_nodes.append(parent)
        heapq.heappush(heap, (parent.freq, counter, parent))
        counter += 1
    _, _, root = heap[0]
    codes: list[list[int]] = [[] for _ in range(V)]
    points: list[list[int]] = [[] for _ in range(V)]

    def walk(node, code, path):
        if node.idx >= 0:
            codes[node.idx] = code
            points[node.idx] = path
            return
        # left = 0 -> sign -1? convention: [[child==ch]] with ch=left? we fix ch=left.
        # If go left, bit=1 with our sign convention below; keep consistent.
        if node.left is not None:
            walk(node.left, code + [0], path + [node.nid])
        if node.right is not None:
            walk(node.right, code + [1], path + [node.nid])

    walk(root, [], [])
    return codes, points

def hs_loss_and_grad(v_in: np.ndarray, v_out: np.ndarray, code: list[int], point: list[int]):
    """Hierarchical softmax forward+grads for one (input, target-word) pair.
    v_in: (D,), v_out: (num_inner, D). code bits 0/1, point inner ids.
    Returns (loss, grad_in, grad_out_dict).
    Label: bit 1 -> target 1, bit 0 -> target 0 (left child convention).
    p = prod sigma(s * v'_n^T v_in) with s=+1 if bit==1 else -1? We use standard:
    score = v'_n . v_in; prob = sigma(score) if bit==1 else sigma(-score).
    """
    loss = 0.0
    grad_in = np.zeros_like(v_in)
    grads = {}
    for bit, nid in zip(code, point):
        z = float(np.dot(v_out[nid], v_in))
        # sigmoid
        s = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else (np.exp(z) / (1.0 + np.exp(z)))
        if bit == 1:
            g = 1.0 - s
            loss += -np.log(max(s, 1e-12))
        else:
            g = -s
            loss += -np.log(max(1.0 - s, 1e-12))
        grads[nid] = grads.get(nid, 0.0) + 0  # placeholder
        # accumulate
        if nid not in grads or isinstance(grads[nid], float):
            grads[nid] = g * v_in.copy()
        else:
            grads[nid] += g * v_in
        grad_in += g * v_out[nid]
    return loss, grad_in, grads
