"""HiMambaRadixCache keeps its LRU lists and access times consistent across load-back.

Regression for the idle sanity check aborting the scheduler with
``AssertionError: Incorrect LRU list, self.mamba=True, x: ... != x_lru: ...``:
``load_back`` restores the mamba state of the deepest backed-up node, which is
often an *ancestor* of the matched path, and moved it to MRU without giving it
a new access time, so it ended up in front of its own descendants while
keeping an older timestamp.
"""

import types
import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.mamba_radix_cache import LRUList, MambaRadixCache, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeController:
    """Hands out device mamba slots for restore transfers; no KV to load."""

    def __init__(self):
        self.mem_pool_host = types.SimpleNamespace(entry_map={})
        self.next_mamba_slot = 100

    def load(self, host_indices, node_id, extra_pools=None):
        for transfer in extra_pools or []:
            if transfer.name == PoolName.MAMBA and transfer.device_indices is None:
                n = len(transfer.host_indices)
                transfer.device_indices = torch.arange(
                    self.next_mamba_slot, self.next_mamba_slot + n
                )
                self.next_mamba_slot += n
        return torch.empty((0,), dtype=torch.int64)


class TestHiMambaLruAfterLoadBack(unittest.TestCase):
    def _cache(self):
        cache = object.__new__(HiMambaRadixCache)
        cache.disable = False
        cache.page_size = 1
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
        cache.cache_controller = _FakeController()
        return cache

    def _add_chain(self, cache, depth):
        parent, nodes, slot = cache.root_node, [], 0
        for i in range(depth):
            node = TreeNode()
            node.parent = parent
            node.key = RadixKey([i])
            node.value = torch.tensor([slot])
            node.host_value = torch.tensor([slot])
            node.mamba_value = torch.tensor([slot])
            parent.children[i] = node
            cache.full_lru_list.insert_mru(node)
            cache.mamba_lru_list.insert_mru(node)
            cache.full_evictable_size_ += 1
            cache.mamba_evictable_size_ += 1
            parent, slot = node, slot + 1
            nodes.append(node)
        return nodes

    def _evict_mamba_to_host(self, cache, node):
        cache.mamba_lru_list.remove_node(node)
        cache.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_host_value = node.mamba_value.clone()
        node.mamba_value = None

    def test_restored_ancestor_keeps_lru_order_consistent(self):
        cache = self._cache()
        a, b, c, d = self._add_chain(cache, 4)
        params = MatchPrefixParams(key=RadixKey([0, 1, 2, 3]))

        # Mamba eviction takes the state closest to the root first.
        self._evict_mamba_to_host(cache, b)
        # The scheduler matches every waiting request before loading any back:
        # request 1 stops at b (its deepest host-backed node) ...
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
