"""Isolated DFLASH tree prototype. No installed SGLang files are modified.

Subclass reuses native prefill, scheduler state, draft model/graphs, target model/
graphs, paged allocation, Eagle target-only tree sampling and KV compaction.
Only decode orchestration is replaced. Default scope is TP1/C1-4, deterministic
Triton target+draft, no grammar/Mamba, node budgets 32/64/128 with a 16-token backbone.
Experimental and unqualified; enabled only by the explicit Spark decoding option.
A separate explicit C1-8 benchmark option permits only the 32-node tree.
"""
import torch

from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative import dflash_worker_v2 as native
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache
from types import MappingProxyType
from ripple_tree_proposal import build_tree
from ripple_tree_attention import install_attention, stage_tree


class LatticeSampler:
    def __init__(self, draft_model, width, device, max_batch_size=4):
        self.model = draft_model
        self.width = width
        if type(max_batch_size) is not int or max_batch_size not in (4, 8):
            raise ValueError("Lattice sampler capacity must be 4 or 8")
        k = draft_model.candidate_selector.top_k
        self.candidates = torch.empty((max_batch_size,width-1,k), dtype=torch.int64, device=device)
        self.scores = torch.empty((max_batch_size,width-1,k,k), dtype=torch.float32, device=device)

    def __call__(self, hidden_states, input_ids):
        b = hidden_states.shape[0] // self.width
        if not 1 <= b <= self.candidates.shape[0]:
            raise ValueError("Lattice batch exceeds its captured capacity")
        hidden = hidden_states.view(b,self.width,-1)[:,1:]
        ids, scores = native._selector_lattice(self.model, hidden, input_ids.view(b,self.width)[:,0])
        self.candidates[:b].copy_(ids)
        self.scores[:b].copy_(scores)


class TreeWorker(native.DFlashWorkerV2):
    _tree_config = None

    def __init__(self, *args, **kwargs):
        if native.get_tp_group().world_size != 1:
            raise ValueError('Tree prototype supports TP1 only')
        self.tree_max_batch_size = self._tree_config.get('max_concurrency', 4)
        self.tree_nodes = int(self._tree_config['nodes'])
        width = int(self._tree_config['draft_width'])
        if self.tree_nodes != int(get_spec().speculative_num_draft_tokens):
            raise ValueError('Native verify budget must match the selected tree node budget')
        if not 2 <= width <= self.tree_nodes:
            raise ValueError('Invalid tree draft width')
        if get_spec().speculative_use_rejection_sampling:
            raise ValueError('Tree proposals require native target-only tree sampling')
        if get_spec().speculative_accept_threshold_single != 1.0 or get_spec().speculative_accept_threshold_acc != 1.0:
            raise ValueError('Exact tree sampler thresholds must both be1.0')
        # The target owns N verification slots. Only construction/capture of the
        # separate draft runner sees its actual backbone width.
        with get_spec().override(speculative_num_draft_tokens=width):
            super().__init__(*args, **kwargs)
        self.speculative_num_draft_tokens = self.tree_nodes
        if self.selector is None:
            raise ValueError('Prototype requires selector draft')
        self.tree_children = int(self._tree_config['children'])
        self.tree_temperature = float(self._tree_config['proposal_temperature'])
        self._tree_pos = torch.arange(self.tree_nodes, device=self.device)
        self._tree_dummy_mask = torch.ones(1, dtype=torch.bool, device=self.device)

    def init_attention_backends(self):
        with get_spec().override(speculative_num_draft_tokens=self.block_size):
            result = super().init_attention_backends()
        for backend in (self.target_worker.model_runner.attn_backend, self.draft_model_runner.attn_backend):
            if type(backend).__name__ != 'TritonAttnBackend' or not backend.enable_deterministic:
                raise ValueError('Tree prototype requires deterministic Triton target and draft')
        return result

    def init_cuda_graphs(self):
        with get_spec().override(speculative_num_draft_tokens=self.block_size):
            return super().init_cuda_graphs()

    def _maybe_build_draft_sampler(self):
        sampler = super()._maybe_build_draft_sampler()
        if sampler is None or not hasattr(sampler,'draft_model'):
            raise RuntimeError('Tree draft requires native selector graph hook support')
        return LatticeSampler(sampler.draft_model, self.block_size, self.device, self.tree_max_batch_size)

    def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch or batch.forward_mode.is_idle():
            result = super().forward_batch_generation(batch, on_publish, grammar_barrier, pp_proxy_tensors)
            result.speculative_num_draft_tokens = self.tree_nodes
            return result
        self._validate_phase1_sampling_support(batch)
        if batch.has_grammar or self._need_mamba_verify_commit:
            raise ValueError('Tree prototype does not support grammar or Mamba')
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise TypeError('Expected native DFLASH draft state')
        b = len(batch.seq_lens)
        if not 1 <= b <= self.tree_max_batch_size:
            raise ValueError('Tree batch exceeds the explicitly configured concurrency')
        batch.seq_lens.record_stream(torch.get_device_module(self.device).current_stream())
        prefix = batch.seq_lens
        width, n = self.block_size, self.tree_nodes
        self._ensure_draft_block_buffers(b)
        block_ids = self._draft_block_ids_buf[:b]
        positions = self._draft_block_positions_buf[:b]
        draft_locs = self._draft_verify_out_cache_loc_buf[:b]
        native._prepare_dflash_draft_block_unchecked(
            bonus_tokens=draft_input.bonus_tokens.view(-1), prefix_lens=prefix.view(-1),
            req_pool_indices=batch.req_pool_indices.view(-1),
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            block_ids_out=block_ids, positions_out=positions, cache_loc_out=draft_locs,
            mask_token_id=int(self._mask_token_id))
        target_model = self.target_worker.model_runner.model
        embedding = native.unwrap_lora_layer(native._resolve_dflash_embedding_module(self.draft_model, target_model))
        embeds = embedding(block_ids)
        if self._noise_embed_scale != 1.0:
            embeds = embeds * self._noise_embed_scale
        seq_cpu = self._draft_seq_lens_cpu_buf[:b]
        if self.use_compact_draft_cache:
            draft_prefix = self._compute_compact_draft_seq_lens(prefix)
            self._fill_compact_seq_lens_cpu_bound(
                batch_seq_lens_cpu=batch.seq_lens_cpu, nxt_kv_lens_cpu=draft_input.nxt_kv_lens_cpu,
                draft_prefix_lens=draft_prefix, out=seq_cpu)
            self._rebuild_compact_draft_cache(
                req_pool_indices=batch.req_pool_indices, prefix_lens=prefix,
                draft_prefix_lens=draft_prefix, verify_out_cache_loc_2d=draft_locs,
                bs=b, block_size=width)
        else:
            draft_prefix = prefix
            if batch.seq_lens_cpu is not None:
                seq_cpu.copy_(batch.seq_lens_cpu)
                seq_cpu.add_(width)
            elif draft_input.nxt_kv_lens_cpu is not None:
                seq_cpu.copy_(draft_input.nxt_kv_lens_cpu)
            else:
                seq_cpu.copy_(prefix.to("cpu", dtype=torch.int32))
        draft_batch = native.ForwardBatch(
            forward_mode=native.ForwardMode.TARGET_VERIFY, batch_size=b,
            input_ids=block_ids.flatten(), req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_prefix, out_cache_loc=draft_locs.flatten(), seq_lens_sum=int(seq_cpu.sum()),
            seq_lens_cpu=seq_cpu, positions=positions.flatten(), input_embeds=embeds.reshape(-1,embeds.shape[-1]),
            spec_algorithm=native.SpeculativeAlgorithm.DFLASH, spec_info=self._draft_block_spec_info,
            capture_hidden_mode=native.CaptureHiddenMode.NULL)
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_batch)
        if self._draft_sampler is not None and draft_out.can_run_graph:
            candidates, scores = self._draft_sampler.candidates[:b], self._draft_sampler.scores[:b]
        else:
            lm_head = native.unwrap_lora_layer(target_model.lm_head)
            self.draft_model.lm_head = lm_head
            hidden = draft_out.logits_output.hidden_states.view(b,width,-1)[:,1:]
            candidates, scores = native._selector_lattice(self.draft_model, hidden, block_ids[:,0])
        proposal = build_tree(candidates, scores, block_ids[:,0], node_budget=n,
                              children=self.tree_children, draft_temperature=self.tree_temperature,
                              max_batch_size=self.tree_max_batch_size)
        tree_positions = prefix[:,None] + proposal.depths
        tree_locs = native.assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix, end_offset=prefix+n, batch_size=b,
            draft_token_num=n, device=self.device).view(b,n)
        stage_tree(self.target_worker.model_runner.attn_backend, proposal)
        verify = DFlashVerifyInput(draft_token=proposal.tokens.flatten(), positions=tree_positions.flatten(),
                                  draft_token_num=n, custom_mask=self._tree_dummy_mask,
                                  capture_hidden_mode=native.CaptureHiddenMode.FULL)
        batch.out_cache_loc = tree_locs.flatten()
        cpu_backup, sum_backup = batch.seq_lens_cpu, batch.seq_lens_sum
        # Native overlap may omit the CPU mirror. Match DFLASH's host-bound
        # convention: reserved next-KV lengths are a safe planning upper bound;
        # neither this bound nor the tree's flat width changes GPU prefix P.
        if cpu_backup is not None:
            batch.seq_lens_cpu = cpu_backup + n
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        try:
            forward_batch, _ = verify.prepare_for_verify(batch, self.target_worker)
        finally:
            batch.seq_lens_cpu, batch.seq_lens_sum = cpu_backup, sum_backup
        target_out = self.target_worker.forward_batch_generation(
            batch=None, forward_batch=forward_batch, is_verify=True, skip_attn_backend_init=True)
        logits = target_out.logits_output
        sampling_tree = EagleVerifyInput(
            draft_token=proposal.tokens.flatten(), custom_mask=self._tree_dummy_mask,
            positions=tree_positions.flatten(), retrieve_index=proposal.retrieve_index,
            retrieve_next_token=proposal.retrieve_next_token, retrieve_next_sibling=proposal.retrieve_next_sibling,
            retrieve_cum_len=None, spec_steps=width-1, topk=-1, draft_token_num=n,
            capture_hidden_mode=native.CaptureHiddenMode.FULL, seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu)
        # This native helper applies target penalties, temperature and top-p and
        # exact target-only multi-candidate sampling. No proposal q is substituted.
        predict, commit, accept_index = eagle_sample(sampling_tree, batch, logits)
        commit = commit.to(torch.int32)
        accepted = commit - 1
        if batch.return_logprob:
            native.compute_spec_logprobs(batch, logits, predict, accept_index=accept_index)
        safe = accept_index.long().clamp_min(0)
        path_tokens = predict[safe]
        bonus = path_tokens.gather(1, accepted.long()[:,None]).squeeze(1).to(torch.int64)
        out_tokens = torch.zeros((b,n), dtype=torch.int32, device=self.device)
        out_tokens[:,:width].copy_(path_tokens)
        move_accept_tokens_to_target_kvcache(batch, accept_index, accepted,
                                            self.model_runner.token_to_kv_pool_allocator)
        hidden = logits.hidden_states
        if hidden is None:
            raise RuntimeError('Tree target did not return draft auxiliary hidden states')
        hidden = hidden[safe.flatten()]
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden, cache_loc=draft_locs.flatten(), cache_loc_2d=draft_locs,
            positions=positions.flatten(), commit_lens=commit)
        logits.hidden_states = None
        new_seq_lens = prefix + commit.to(prefix.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)
        next_input = self._make_next_draft_input_decode(bonus_tokens=bonus,new_seq_lens=new_seq_lens)
        return native.GenerationBatchResult(
            logits_output=logits, next_token_ids=out_tokens.flatten(), accept_lens=commit,
            can_run_cuda_graph=target_out.can_run_cuda_graph, next_draft_input=next_input,
            speculative_num_draft_tokens=n, new_seq_lens=new_seq_lens,
            routed_experts_output=target_out.routed_experts_output,
            indexer_topk_output=target_out.indexer_topk_output)


def validate_config(config):
    """Accept only the explicit experimental configuration tested by this adapter."""
    if not isinstance(config, dict) or type(config.get('nodes')) is not int or config['nodes'] not in (32, 64, 128):
        raise ValueError('Tree node budget must be 32, 64 or 128')
    expected = {'nodes': config['nodes'], 'draft_width':16, 'children':16,
                'proposal_temperature':1.5, 'sampler':'native_eagle_target_only',
                'exact_thresholds':[1.0, 1.0]}
    if 'max_concurrency' in config:
        if type(config['max_concurrency']) is not int or config['max_concurrency'] != 8 or config['nodes'] != 32:
            raise ValueError('Experimental C1–8 requires max_concurrency=8 and nodes=32')
        expected['max_concurrency'] = 8
    if config != expected or set(config) != set(expected):
        raise ValueError('Tree backbone 16 / children 16 / temperature 1.5/exact native sampler are fixed')
    if type(config['draft_width']) is not int or type(config['children']) is not int or any(
            type(x) not in (int, float) for x in config['exact_thresholds']):
        raise ValueError('Tree configuration contains invalid numeric types')
    expected['exact_thresholds'] = (1.0, 1.0)
    return MappingProxyType(expected)


def install(config, server_args):
    """Opt in inside one scheduler process, before native workers/graphs are built.

    The installed SGLang package is never changed. A process can use one immutable
    tree configuration; chain and target-only processes do not call this function.
    Native target-only numerical qualification still retains the 0.25-nat FAIL.
    """
    config = validate_config(config)
    expected = {'speculative_algorithm':'DFLASH', 'speculative_num_draft_tokens':config['nodes'],
                'speculative_draft_window_size':16384, 'attention_backend':'triton',
                'speculative_draft_attention_backend':'triton', 'enable_deterministic_inference':True,
                'tp_size':1}
    for key, value in expected.items():
        actual = getattr(server_args, key, None)
        if actual != value or type(actual) is not type(value):
            raise ValueError('Tree adapter received unsupported server setting: '+key)
    concurrency = getattr(server_args, 'max_running_requests', None)
    batch_limit = config.get('max_concurrency', 4)
    if type(concurrency) is not int or not 1 <= concurrency <= batch_limit:
        raise ValueError('Tree adapter concurrency exceeds the explicit configuration')
    installed = getattr(native.DFlashWorkerV2, '_ripple_tree_config', None)
    if installed is not None:
        if installed != config:
            raise ValueError('A different tree configuration is already installed in this worker process')
        return
    if batch_limit == 4:
        install_attention(config['nodes'])
    else:
        install_attention(config['nodes'], max_batch_size=batch_limit)
    TreeWorker._tree_config = config
    TreeWorker._ripple_tree_config = config
    TreeWorker._ripple_tree_worker = True
    native.DFlashWorkerV2 = TreeWorker
