"""HiCache host round trips for state that lives beside the full KV / mamba slots.

Covers the regressions behind "KV cache looks tampered after enabling HiCache"
on Qwen3.8-Flash-Next (Qwen4-Exp):

* QSA compressed indexer keys follow the full-KV pages through host memory and
  land in the *new* device page on load-back (upstream #39893).
* PLE slot side states (n-gram context, short-conv window) follow the mamba
  checkpoint slot (upstream #39862).
* HiCache D2H / H2D copies wait for forwards already queued on the forward
  stream (upstream #36738 / #42701).

The transfer kernels are replaced by torch reference copies, so this runs on CPU.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.managers import cache_controller as cache_controller_module
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.qsa_host_pool import QSAIndexerPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _rows(tensor: torch.Tensor, item_size: int) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.uint8).view(-1, item_size)


class _ReferenceKernels:
    """Torch stand-ins for the sgl_kernel kvcacheio copies used on these paths."""

    def __init__(self):
        self.ptr_to_tensor = {}

    def register(self, tensors):
        for tensor in tensors:
            self.ptr_to_tensor[tensor.data_ptr()] = tensor

    def transfer_kv_per_layer_mla(self, src, dst, src_indices, dst_indices, item_size):
        src_rows = _rows(src, item_size)
        dst_rows = _rows(dst, item_size)
        dst_rows[dst_indices.long().cpu()] = src_rows[src_indices.long().cpu()]

    def transfer_kv_all_layer_mla(
        self, src_layers, dst_layers, src_indices, dst_indices, item_size, num_layers
    ):
        for layer in range(num_layers):
            self.transfer_kv_per_layer_mla(
                self.ptr_to_tensor[int(src_layers[layer])],
                self.ptr_to_tensor[int(dst_layers[layer])],
                src_indices,
                dst_indices,
                item_size,
            )

    def transfer_kv_direct(
        self, src_layers, dst_layers, src_indices, dst_indices, page_size
    ):
        for src, dst in zip(src_layers, dst_layers):
            dst[dst_indices.long().cpu()] = src[src_indices.long().cpu()]

    def patch(self):
        return mock.patch.multiple(
            memory_pool_host,
            create=True,
            transfer_kv_per_layer_mla=self.transfer_kv_per_layer_mla,
            transfer_kv_all_layer_mla=self.transfer_kv_all_layer_mla,
            transfer_kv_direct=self.transfer_kv_direct,
        )


def _page_tokens(pages, page_size):
    return torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size, dtype=torch.int64)
            for p in pages
        ]
    )


class TestQSAIndexerHostRoundTrip(unittest.TestCase):
    page_size = 8
    ratio = 4
    num_layers = 2
    full_slots = 6 * 8  # six full-KV pages
    heads, head_dim = 1, 16

    def _device_pool(self):
        capacity = -((self.full_slots + self.page_size) // -self.ratio)
        flat = torch.randn(self.num_layers, capacity * self.heads * self.head_dim).to(
            torch.float16
        )
        return types.SimpleNamespace(
            qsa_compress_ratio=self.ratio,
            qsa_compressed_k_buffer_pool=[
                flat[i].view(capacity, self.heads, self.head_dim)
                for i in range(self.num_layers)
            ],
        )

    def _host_pool(self, device_pool, layout="layer_first"):
        anchor = types.SimpleNamespace(
            page_size=self.page_size,
            page_num=10,
            layout=layout,
            device="cpu",
            pin_memory=False,
        )
        return QSAIndexerPoolHost(device_pool, anchor)

    def _group_rows(self, buffer, page):
        per_page = self.page_size // self.ratio
        return buffer[page * per_page : (page + 1) * per_page].clone()

    def _roundtrip(self, io_backend):
        device_pool = self._device_pool()
        host = self._host_pool(device_pool)
        kernels = _ReferenceKernels()
        kernels.register(host.device_buffers)
        kernels.register(host.data_refs)

        src_pages, host_pages, dst_pages = [1, 2], [5, 3], [4, 0]
        expected = [
            [self._group_rows(b, p) for p in src_pages]
            for b in device_pool.qsa_compressed_k_buffer_pool
        ]
        with kernels.patch():
            host.backup_from_device_all_layer(
                device_pool,
                _page_tokens(host_pages, self.page_size),
                _page_tokens(src_pages, self.page_size),
                io_backend,
            )
            for buffer in device_pool.qsa_compressed_k_buffer_pool:
                buffer.zero_()
            for layer in range(self.num_layers):
                host.load_to_device_per_layer(
                    device_pool,
                    _page_tokens(host_pages, self.page_size),
                    _page_tokens(dst_pages, self.page_size),
                    layer,
                    io_backend,
                )

        for layer, buffer in enumerate(device_pool.qsa_compressed_k_buffer_pool):
            for want, page in zip(expected[layer], dst_pages):
                torch.testing.assert_close(self._group_rows(buffer, page), want)
            # Pages that were not loaded stay untouched (zero here).
            self.assertTrue(torch.all(self._group_rows(buffer, 1) == 0))

    def test_kernel_layer_first_roundtrip_lands_in_new_page(self):
        self._roundtrip("kernel")

    def test_direct_layer_first_roundtrip_lands_in_new_page(self):
        self._roundtrip("direct")

    def test_empty_transfer_is_a_noop(self):
        device_pool = self._device_pool()
        host = self._host_pool(device_pool)
        empty = torch.empty((0,), dtype=torch.int64)
        with mock.patch.object(
            memory_pool_host, "transfer_kv_per_layer_mla", create=True
        ) as kernel:
            host.load_to_device_per_layer(device_pool, empty, empty, 0, "kernel")
            host.backup_from_device_all_layer(device_pool, empty, empty, "kernel")
        kernel.assert_not_called()

    def test_rejects_page_not_divisible_by_ratio(self):
        device_pool = self._device_pool()
        anchor = types.SimpleNamespace(
            page_size=6,
            page_num=4,
            layout="layer_first",
            device="cpu",
            pin_memory=False,
        )
        with self.assertRaises(ValueError):
            QSAIndexerPoolHost(device_pool, anchor)


class TestMambaSideStateHostRoundTrip(unittest.TestCase):
    slots = 8
    num_layers = 2

    def _device_pool(self):
        conv = [torch.randn(self.num_layers, self.slots + 1, 4, 3)]
        temporal = torch.randn(self.num_layers, self.slots + 1, 2, 4)
        short_conv = ShortConvPool(
            size=self.slots,
            state_shape=(4, 2),
            layer_ids=[0, 3],
            dtype=torch.float16,
            device="cpu",
        )
        short_conv.conv_state.copy_(torch.randn_like(short_conv.conv_state))
        ngram = NGramPool(size=self.slots, context_len=2, eos_token_id=7, device="cpu")
        ngram.context.copy_(torch.randint(0, 1000, ngram.context.shape))
        pool = types.SimpleNamespace(
            num_mamba_layers=self.num_layers,
            mamba_cache=types.SimpleNamespace(conv=conv, temporal=temporal),
            size=self.slots,
            device="cpu",
            _slot_siblings=[short_conv, ngram],
        )
        return pool, short_conv, ngram

    def test_side_states_follow_the_mamba_slot(self):
        pool, short_conv, ngram = self._device_pool()
        host = MambaPoolHost(
            pool, host_to_device_ratio=2.0, host_size=0, pin_memory=False
        )
        self.assertEqual(len(host.slot_state_buffers), 3)  # 2 short-conv layers + ngram

        src, host_slots, dst = (
            torch.tensor([2, 3]),
            torch.tensor([9, 4]),
            torch.tensor([6, 1]),
        )
        want_conv = short_conv.conv_state[:, src].clone()
        want_ctx = ngram.context[src].clone()
        want_temporal = pool.mamba_cache.temporal[:, src].clone()

        kernels = _ReferenceKernels()
        with kernels.patch():
            host.backup_from_device_all_layer(pool, host_slots, src, "kernel")
            short_conv.reset_slots(torch.arange(self.slots + 1))
            ngram.reset_slots(torch.arange(self.slots + 1))
            pool.mamba_cache.temporal.zero_()
            for layer in range(self.num_layers):
                host.load_to_device_per_layer(pool, host_slots, dst, layer, "kernel")

        torch.testing.assert_close(short_conv.conv_state[:, dst], want_conv)
        torch.testing.assert_close(ngram.context[dst], want_ctx)
        torch.testing.assert_close(pool.mamba_cache.temporal[:, dst], want_temporal)

    def test_size_per_token_counts_side_state_once(self):
        pool, short_conv, ngram = self._device_pool()
        host = MambaPoolHost(
            pool, host_to_device_ratio=2.0, host_size=0, pin_memory=False
        )
        base = (4 * 3 + 2 * 4) * 4 * self.num_layers
        side = 2 * (4 * 2 * 2) + 2 * 8
        self.assertEqual(host.get_size_per_token(), base + side)

    def test_flat_page_roundtrip_includes_side_states(self):
        pool, short_conv, ngram = self._device_pool()
        host = MambaPoolHost(
            pool, host_to_device_ratio=2.0, host_size=0, pin_memory=False
        )
        for buf in host.slot_state_buffers:
            buf.copy_(torch.randint(0, 100, buf.shape).to(buf.dtype))
        page = host.get_data_page(3)
        self.assertEqual(page.numel(), host.size_per_token)
        want = [buf[3].clone() for buf in host.slot_state_buffers]
        for buf in host.slot_state_buffers:
            buf.zero_()
        host.set_from_flat_data_page(5, page)
        for buf, w in zip(host.slot_state_buffers, want):
            torch.testing.assert_close(buf[5], w)
        with self.assertRaises(ValueError):
            host.set_from_flat_data_page(5, page[:-8])

    def test_unknown_sibling_is_rejected(self):
        pool, _, _ = self._device_pool()
        pool._slot_siblings = [object()]
        with self.assertRaises(ValueError):
            MambaPoolHost(pool, host_to_device_ratio=2.0, host_size=0, pin_memory=False)


class TestSideStateReadersWait(unittest.TestCase):
    def _pool(self):
        pool = object.__new__(HybridReqToTokenPool)
        pool.layer_transfer_counter = mock.Mock()
        pool.mamba_map = {1: 0, 2: 1, 5: 2}
        pool.start_layer = 0
        return pool

    def test_ngram_reader_waits_for_first_mamba_layer(self):
        pool = self._pool()
        pool._wait_for_slot_side_states()
        pool.layer_transfer_counter.wait_until.assert_called_once_with(1)

    def test_short_conv_reader_waits_for_its_layer_and_the_restore(self):
        pool = self._pool()
        pool._wait_for_slot_side_states(0)
        pool.layer_transfer_counter.wait_until.assert_called_once_with(1)
        pool.layer_transfer_counter.reset_mock()
        pool._wait_for_slot_side_states(4)
        pool.layer_transfer_counter.wait_until.assert_called_once_with(4)


class TestHiMambaSidecarTransfers(unittest.TestCase):
    def _cache(self, names):
        cache = object.__new__(HiMambaRadixCache)
        cache.cache_controller = types.SimpleNamespace(
            mem_pool_host=types.SimpleNamespace(entry_map={n: None for n in names})
        )
        return cache

    def test_indexer_sidecars_follow_kv_and_keep_mamba_first(self):
        cache = self._cache([PoolName.KV, PoolName.MAMBA, PoolName.INDEXER])
        mamba = [PoolTransfer(name=PoolName.MAMBA)]
        transfers = cache._with_kv_sidecar_transfers(mamba)
        self.assertIs(transfers[0], mamba[0])
        self.assertEqual(
            [(t.name, t.indices_from_pool) for t in transfers[1:]],
            [(PoolName.INDEXER, PoolName.KV)],
        )
        cache = self._cache(
            [PoolName.KV, PoolName.MAMBA, PoolName.INDEXER, PoolName.DRAFT_INDEXER]
        )
        names = [t.name for t in cache._with_kv_sidecar_transfers(None)]
        self.assertEqual(names, [PoolName.INDEXER, PoolName.DRAFT_INDEXER])

    def test_no_sidecars_keeps_transfers_unchanged(self):
        cache = self._cache([PoolName.KV, PoolName.MAMBA])
        self.assertIsNone(cache._with_kv_sidecar_transfers(None))
        mamba = [PoolTransfer(name=PoolName.MAMBA)]
        self.assertIs(cache._with_kv_sidecar_transfers(mamba), mamba)


class _RecordingStream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_stream(self, other):
        self.events.append(f"{self.name} waits {other.name}")


class _FakeEvent:
    def record(self, *args):
        pass

    def wait(self, *args):
        pass


class _FakeDeviceModule:
    Event = _FakeEvent

    @staticmethod
    def stream(_stream):
        return mock.MagicMock()


class TestTransferFences(unittest.TestCase):
    def _controller(self, events):
        controller = object.__new__(HiCacheController)
        controller.write_queue = []
        controller.load_queue = []
        controller.ack_write_queue = []
        controller.ack_load_queue = []
        controller.io_backend = "direct"
        controller.has_draft = False
        controller.layer_num = 1
        controller.write_stream = _RecordingStream("d2h", events)
        controller.load_stream = _RecordingStream("h2d", events)
        controller.write_fence_stream = _RecordingStream("forward", events)
        controller.load_fence_stream = _RecordingStream("forward", events)
        controller.move_indices = lambda h, d: (h, d)
        controller.mem_pool_device = None
        controller.mem_pool_host = mock.Mock()
        controller.mem_pool_host.backup_from_device_all_layer.side_effect = (
            lambda *a, **k: events.append("copy d2h")
        )
        controller.mem_pool_host.load_to_device_per_layer.side_effect = (
            lambda *a, **k: events.append("copy h2d")
        )
        counter = mock.Mock()
        counter.update_producer.return_value = 0
        counter.events = [mock.Mock()]
        controller.layer_done_counter = counter
        return controller

    def test_write_copy_waits_for_queued_forwards(self):
        events = []
        controller = self._controller(events)
        controller.mem_pool_host.alloc.return_value = torch.arange(4)
        with mock.patch.object(
            cache_controller_module, "device_module", _FakeDeviceModule
        ):
            controller.write(torch.arange(4, 8), node_id=1)
        self.assertEqual(events, ["d2h waits forward", "copy d2h"])

    def test_load_copy_waits_for_queued_forwards(self):
        events = []
        controller = self._controller(events)
        controller.mem_pool_device_allocator = mock.Mock()
        controller.mem_pool_device_allocator.alloc.return_value = torch.arange(4)
        with mock.patch.object(
            cache_controller_module, "device_module", _FakeDeviceModule
        ):
            controller.load(torch.arange(4, 8), node_id=1)
            controller.start_loading()
        self.assertEqual(events, ["h2d waits forward", "copy h2d"])

    def test_no_fence_without_forward_stream(self):
        events = []
        controller = self._controller(events)
        controller.write_fence_stream = None
        controller.mem_pool_host.alloc.return_value = torch.arange(4)
        with mock.patch.object(
            cache_controller_module, "device_module", _FakeDeviceModule
        ):
            controller.write(torch.arange(4, 8), node_id=1)
        self.assertEqual(events, ["copy d2h"])


if __name__ == "__main__":
    unittest.main()
