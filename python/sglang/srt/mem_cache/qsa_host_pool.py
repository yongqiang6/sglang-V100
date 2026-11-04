"""HiCache host mirror for the QSA compressed-key indexer cache.

The compressed QSA indexer keeps one compressed key per ``compress_ratio``
full-KV slots, addressed as ``full_slot // compress_ratio`` (see
``QSATokenToKVPool``).  Its lifetime follows the full-KV pages, so when
HiCache moves a full-KV page to host memory and later loads it back into a
*different* device page, the compressed keys of that page must make the same
round trip.  Without it the indexer reads whatever the new device page held
before (another request's keys), selects the wrong top-k blocks, and the
model attends to unrelated context.

Back-ported from upstream sglang #39893 ("preserve QSA indexer state through
HiCache") and adapted to this fork's ``HostPoolGroup`` sidecar plumbing.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


def _page_rows(buffer: torch.Tensor, item_bytes: int) -> torch.Tensor:
    """View one layer's compressed buffer as ``[full_kv_page, item_bytes]``.

    The compressed capacity is ``ceil((size + page_size) / ratio)`` slots, so
    it can end with a partial page that no full-KV page maps to. Only whole
    pages are exposed to the transfer kernels.
    """
    if not buffer.is_contiguous():
        raise ValueError("QSA HiCache requires contiguous compressed-key buffers")
    flat = buffer.reshape(-1).view(torch.uint8)
    num_rows = flat.numel() // item_bytes
    return flat[: num_rows * item_bytes].view(num_rows, item_bytes)


class QSAIndexerPoolHost(DeepSeekV4PagedHostPool):
    """Page-row mirror of QSA compressed keys, indexed by full-KV token slots.

    A full-KV page and its ``page_size // compress_ratio`` compressed groups
    share ownership, so the groups are packed into one byte row per page and
    moved with the same page-granular kernels, using the full-KV page number
    as the row index on both the device and the host side. The transfer
    indices are the full-KV ones (sidecar of ``PoolName.KV``), so this pool
    never allocates host slots of its own.
    """

    def __init__(
        self,
        device_pool: QSATokenToKVPool,
        anchor_host,
        *,
        allocator_type: str = "default",
    ):
        page_size = anchor_host.page_size
        ratio = device_pool.qsa_compress_ratio
        if page_size % ratio:
            raise ValueError(
                "QSA HiCache pages must be divisible by the compress ratio: "
                f"page_size={page_size}, ratio={ratio}"
            )
        buffers = list(device_pool.qsa_compressed_k_buffer_pool)
        if not buffers:
            raise ValueError("QSA HiCache requires at least one local indexer layer")
        item_bytes = buffers[0][0].nbytes * (page_size // ratio)
        if any(b[0].nbytes * (page_size // ratio) != item_bytes for b in buffers):
            raise ValueError("QSA indexer layers must share the same page shape")

        self.device_pool = device_pool
        super().__init__(
            pool_name="qsa_indexer",
            device_buffers=[_page_rows(b, item_bytes) for b in buffers],
            item_bytes=item_bytes,
            num_host_pages=anchor_host.page_num,
            slot_page_size=page_size,
            layout=anchor_host.layout,
            device=anchor_host.device,
            pin_memory=anchor_host.pin_memory,
            allocator_type=allocator_type,
        )
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.size_per_token = self.get_size_per_token()

    def get_size_per_token(self):
        return self.layer_num * self.item_bytes // self.slot_page_size

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        if host_indices.numel() == 0:
            return
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        if host_indices.numel() == 0:
            return
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
