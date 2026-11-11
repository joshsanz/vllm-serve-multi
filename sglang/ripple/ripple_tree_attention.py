"""Experimental exact-depth masks around stock SGLang Triton paged attention.

Only target TARGET_VERIFY of the configured tree width is changed. Native KV
writes, physical SWA translation, unified paged-index construction and attention
kernels are reused. The stock flat-row SWA cutoff is replaced by a custom mask
whose query position is tree depth, never the flattened sibling index.
"""
import functools
import torch
import triton
import triton.language as tl


@triton.jit
def _mask_kernel(PREFIX, DEPTH, ANC, MASK, PTR, B: tl.constexpr, N: tl.constexpr,
                 WINDOW: tl.constexpr, TILE: tl.constexpr, MAX_BATCH: tl.constexpr = 4):
    b = tl.program_id(0)
    q = tl.program_id(1)
    rows = tl.arange(0, MAX_BATCH)
    lens = tl.load(PREFIX + rows, mask=rows < B, other=0)
    before = tl.sum(tl.where(rows < b, (lens + N) * N, 0))
    plen = tl.load(PREFIX + b)
    cols = plen + N
    if q == 0:
        tl.store(PTR + b, before)
        if b == B - 1:
            tl.store(PTR + B, before + cols * N)
    depth = tl.load(DEPTH + b * N + q)
    offsets = tl.arange(0, TILE)
    for start in range(0, cols, TILE):
        col = start + offsets
        is_prefix = col < plen
        node = col - plen
        ancestor = tl.load(ANC + b*N*N + q*N + tl.maximum(node, 0),
                           mask=(node >= 0) & (node < N), other=0)
        visible_prefix = is_prefix
        if WINDOW >= 0:
            # Both sides subtract the original prefix length L:
            # L+depth <= (L-plen+col)+window.
            visible_prefix &= depth + plen <= col + WINDOW
            key_depth = tl.load(DEPTH + b*N + tl.maximum(node, 0), mask=(node >= 0) & (node < N), other=0)
            ancestor &= depth - key_depth <= WINDOW
        visible = tl.where(is_prefix, visible_prefix, ancestor)
        tl.store(MASK + before + q*cols + col, visible, mask=col < cols)


def stage_tree(backend, proposal):
    """Copy topology into capture-stable backend buffers, including padded slots."""
    if type(backend).__name__ != 'TritonAttnBackend':
        raise ValueError('Prototype requires one native deterministic Triton backend')
    if not hasattr(backend, '_ripple_tree_depths'):
        raise RuntimeError('Tree attention adapter was not initialized')
    b, n = proposal.tokens.shape
    if not 1 <= b <= backend._ripple_tree_max_batch_size or n != backend._ripple_tree_nodes:
        raise ValueError('Tree shape does not match captured target attention')
    backend._ripple_tree_mask_ready.clear()
    backend._ripple_tree_depths.zero_()
    backend._ripple_tree_ancestors.zero_()
    backend._ripple_tree_ancestors[:, :, 0] = True
    backend._ripple_tree_depths[:b].copy_(proposal.depths)
    backend._ripple_tree_ancestors[:b].copy_(proposal.ancestors)


def install_attention(node_budget, max_batch_size=4):
    if type(max_batch_size) is not int or max_batch_size not in (4, 8):
        raise ValueError("Tree attention capacity must be 4 or 8")
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.speculative import dflash_utils
    if getattr(TritonAttnBackend, '_ripple_tree_installed', False):
        if TritonAttnBackend._ripple_tree_attention_config != (node_budget, max_batch_size):
            raise ValueError('Different tree attention capacity is already installed')
        return
    original_init = TritonAttnBackend.__init__
    original_forward = TritonAttnBackend._forward_extend_unified
    original_policy = dflash_utils.resolve_dflash_verify_mask_policy

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.is_draft_runner:
            return
        if not self.enable_deterministic:
            raise ValueError('Tree adapter requires deterministic Triton attention')
        n = int(node_budget)
        self._ripple_tree_nodes = n
        self._ripple_tree_max_batch_size = max_batch_size
        self._ripple_tree_depths = torch.zeros((max_batch_size,n), dtype=torch.int64, device=self.device)
        self._ripple_tree_ancestors = torch.eye(n, dtype=torch.bool, device=self.device).expand(max_batch_size,-1,-1).clone()
        self._ripple_tree_masks = {
            False: torch.empty(max_batch_size*n*(self.max_context_len+n), dtype=torch.uint8, device=self.device),
            True: torch.empty(max_batch_size*n*(min(self.max_context_len, self.sliding_window_size or self.max_context_len)+n), dtype=torch.uint8, device=self.device),
        }
        self._ripple_tree_mask_ptrs = {key: torch.zeros(max_batch_size+1, dtype=torch.int64, device=self.device) for key in (False,True)}
        self._ripple_tree_active = False
        self._ripple_tree_mask_ready = set()
        native = self.extend_attention_fwd_unified

        def attention(*a, **kw):
            if not self._ripple_tree_active:
                return native(*a, **kw)
            window = int(kw.get('sliding_window_size', -1))
            kind = window >= 0
            b = a[6].numel()-1
            if not 1 <= b <= max_batch_size or a[10] != n:
                raise RuntimeError('Unsupported tree target attention shape')
            if kind not in self._ripple_tree_mask_ready:
                _mask_kernel[(b,n)](a[9], self._ripple_tree_depths,
                    self._ripple_tree_ancestors, self._ripple_tree_masks[kind],
                    self._ripple_tree_mask_ptrs[kind], B=b, N=n, WINDOW=window, TILE=256, MAX_BATCH=max_batch_size)
                self._ripple_tree_mask_ready.add(kind)
            kw['custom_mask'] = self._ripple_tree_masks[kind]
            kw['mask_indptr'] = self._ripple_tree_mask_ptrs[kind][:b+1]
            kw['sliding_window_size'] = -1
            kw['window_start_pos'] = None
            kw['is_causal'] = False
            return native(*a, **kw)
        self.extend_attention_fwd_unified = attention

    @functools.wraps(original_forward)
    def forward(self, q, o, layer, forward_batch, *args, **kwargs):
        if self.is_draft_runner:
            return original_forward(self, q, o, layer, forward_batch, *args, **kwargs)
        spec = forward_batch.spec_info
        active = (not self.is_draft_runner and forward_batch.forward_mode.is_target_verify()
                  and spec is not None and spec.draft_token_num == node_budget
                  and spec.custom_mask is not None)
        if active and layer.layer_id == 0:
            self._ripple_tree_mask_ready.clear()
        previous = self._ripple_tree_active
        self._ripple_tree_active = active
        try:
            return original_forward(self, q, o, layer, forward_batch, *args, **kwargs)
        finally:
            self._ripple_tree_active = previous

    def mask_policy(backend):
        name, native = original_policy(backend)
        return name, True if name == 'TritonAttnBackend' else native

    TritonAttnBackend.__init__ = init
    TritonAttnBackend._forward_extend_unified = forward
    TritonAttnBackend._ripple_tree_attention_config = (node_budget, max_batch_size)
    TritonAttnBackend._ripple_tree_installed = True
    dflash_utils.resolve_dflash_verify_mask_policy = mask_policy
