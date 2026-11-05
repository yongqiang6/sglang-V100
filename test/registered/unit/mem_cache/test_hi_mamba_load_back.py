"""HiMambaRadixCache load-back regressions.

* The load-back target is the matched node itself, never a shallower
  backed-up ancestor. Restoring the ancestor paired its older mamba state with
  the KV prefix down to the matched node and locked only the ancestor's path,
  so the unlocked KV below it could be evicted and reused while the request
  still read it; the idle check then reported
  ``pool memory leak detected! [full] ... available + evictable > total``.
* A failed load-back shrinks the request's prefix to the node it falls back to.
* Load-back keeps the LRU lists and access times consistent, or the idle check
  aborts with ``Incorrect LRU list, self.mamba=True, x: ... != x_lru: ...``.
"""

import types
import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.mamba_radix_cache import LRUList, MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeController:
    """Hands out device mamba slots for restore transfers; no KV to load."""

    def __init__(self, fail=False):
        self.mem_pool_host = types.SimpleNamespace(entry_map={})
        self.next_mamba_slot = 100
        self.fail = fail

    def load(self, host_indices, node_id, extra_pools=None):
        if self.fail:
            return None
        for transfer in extra_pools or []:
            if transfer.name == PoolName.MAMBA and transfer.device_indices is None:
                n = len(transfer.host_indices)
                transfer.device_indices = torch.arange(
                    self.next_mamba_slot, self.next_mamba_slot + n
                )
                self.next_mamba_slot += n
        return torch.empty((0,), dtype=torch.int64)


def _make_cache(fail_load=False):
    cache = object.__new__(HiMambaRadixCache)
    cache.disable = False
    cache.page_size = 1
    cache.device = "cpu"
    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey([])
    cache.root_node.value = torch.empty((0,), dtype=torch.int64)
    cache.full_lru_list = LRUList(mamba=False)
    cache.mamba_lru_list = LRUList(mamba=True)
    cache.full_evictable_size_ = 0
    cache.full_protected_size_ = 0
    cache.mamba_evictable_size_ = 0
    cache.mamba_protected_size_ = 0
    cache.evictable_full_device_leaves = set()
    cache.evictable_full_host_leaves = set()
    cache.ongoing_load_back = {}
    cache.load_back_threshold = 10
    cache.cache_controller = _FakeController(fail=fail_load)
    return cache


def _add_chain(cache, lengths):
    """Append a chain of device nodes (KV + device mamba) under the root."""
    parent, nodes, slot, token = cache.root_node, [], 0, 0
    for i, n in enumerate(lengths):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(list(range(token, token + n)))
        node.value = torch.arange(slot, slot + n)
        node.mamba_value = torch.tensor([i + 1])
        parent.children[token] = node
        cache.full_lru_list.insert_mru(node)
        cache.mamba_lru_list.insert_mru(node)
        cache.full_evictable_size_ += n
        cache.mamba_evictable_size_ += 1
        parent, slot, token = node, slot + n, token + n
        nodes.append(node)
    return nodes


def _back_up(node):
    node.host_value = node.value.clone()
    if node.mamba_value is not None:
        node.mamba_host_value = node.mamba_value.clone()


def _evict_mamba_to_host(cache, node):
    assert node.mamba_host_value is not None
    cache.mamba_lru_list.remove_node(node)
    cache.mamba_evictable_size_ -= len(node.mamba_value)
    node.mamba_value = None


def _req(prefix_indices, mamba_pool_idx=None):
    return types.SimpleNamespace(
        prefix_indices=prefix_indices,
        mamba_pool_idx=mamba_pool_idx,
        mamba_cow_src_index=None,
        mamba_needs_clear=False,
    )


class TestHiMambaLoadBackTarget(unittest.TestCase):
    def test_unbacked_match_does_not_load_back_an_ancestor(self):
        cache = _make_cache()
        h, b = _add_chain(cache, [4, 4])
        # h is on host and its mamba state was evicted to host; b (e.g. a
        # chunked-prefill node) is not backed up yet but has device mamba.
        _back_up(h)
        _evict_mamba_to_host(cache, h)

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8)))))

        self.assertIs(result.last_device_node, b)
        self.assertIs(result.best_match_node, b)
        self.assertIs(result.last_host_node, h)
        self.assertEqual(result.host_hit_length, 0)
        torch.testing.assert_close(result.device_indices, torch.arange(8))

    def test_matched_node_mamba_on_host_loads_back_that_node(self):
        cache = _make_cache()
        a, b = _add_chain(cache, [4, 4])
        _back_up(a)
        _back_up(b)
        _evict_mamba_to_host(cache, b)

        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8)))))
        self.assertIs(result.best_match_node, b)
        self.assertEqual(result.host_hit_length, 1)

        req = _req(result.device_indices.clone(), mamba_pool_idx=torch.tensor(7))
        new_indices, last_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=result.best_match_node,
                host_hit_length=result.host_hit_length,
                req=req,
            )
        )
        self.assertIs(last_node, b)
        self.assertEqual(new_indices.numel(), 0)
        self.assertIsNotNone(b.mamba_value)
        self.assertEqual(len(req.prefix_indices), 8)

    def test_failed_load_back_shrinks_prefix_to_fallback_node(self):
        cache = _make_cache(fail_load=True)
        a, b = _add_chain(cache, [4, 4])
        _back_up(a)
        _back_up(b)
        _evict_mamba_to_host(cache, b)
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8)))))
        req = _req(result.device_indices.clone(), mamba_pool_idx=torch.tensor(7))

        new_indices, last_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=result.best_match_node,
                host_hit_length=result.host_hit_length,
                req=req,
            )
        )

        # The request falls back to a, so its prefix and mamba state are a's.
        self.assertIs(last_node, a)
        self.assertEqual(new_indices.numel(), 0)
        torch.testing.assert_close(req.prefix_indices, torch.arange(4))
        self.assertIs(req.mamba_cow_src_index, a.mamba_value)
        self.assertFalse(req.mamba_needs_clear)

    def test_failed_load_back_to_root_clears_the_request_state(self):
        cache = _make_cache(fail_load=True)
        (b,) = _add_chain(cache, [4])
        _back_up(b)
        _evict_mamba_to_host(cache, b)
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(4)))))
        req = _req(result.device_indices.clone(), mamba_pool_idx=torch.tensor(7))

        _, last_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=result.best_match_node,
                host_hit_length=result.host_hit_length,
                req=req,
            )
        )

        self.assertIs(last_node, cache.root_node)
        self.assertEqual(req.prefix_indices.numel(), 0)
        self.assertIsNone(req.mamba_cow_src_index)
        self.assertTrue(req.mamba_needs_clear)


class TestHiMambaLruAfterLoadBack(unittest.TestCase):
    def test_restored_ancestor_keeps_lru_order_consistent(self):
        cache = _make_cache()
        a, b, c, d = _add_chain(cache, [1, 1, 1, 1])
        for node in (a, b, c, d):
            _back_up(node)
        params = MatchPrefixParams(key=RadixKey([0, 1, 2, 3]))

        # Mamba eviction takes the state closest to the root first.
        _evict_mamba_to_host(cache, b)
        # The scheduler matches every waiting request before loading any back:
        # request 1 stops at b (its matched node, mamba on host) ...
        cache._match_post_processor(params, [a.value, b.value], b, 2)
        # ... request 2 then walks the whole chain, so b is now older than c, d.
        cache._match_post_processor(params, [n.value for n in (a, b, c, d)], d, 4)
        # Request 1 is admitted and b's mamba state is restored from host.
        self.assertIsNotNone(cache.load_back(b, mem_quota=None, req=None))
        cache.dec_lock_ref(b)
        cache.ongoing_load_back.clear()

        # The scheduler runs this when it goes idle.
        MambaRadixCache.sanity_check(cache)

        # b is now in front of its descendants in both lists and in time.
        self.assertGreater(b.last_access_time, c.last_access_time)
        self.assertIs(cache.mamba_lru_list.head.mamba_next, b)
        self.assertIs(cache.full_lru_list.head.next, b)


if __name__ == "__main__":
    unittest.main()
