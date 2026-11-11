"""Small transport compatibility fixes for stock SGLang 0.5.19."""
import functools


def patch_request_cleanup():
    """Abort dispatched n=1 requests/native batches before discarding their state.

    SGLang's exception cleanup removes request IDs before its delayed disconnect
    abort runs. The native abort guard then ignores them, leaving GPU work active.
    Parallel-sampling child IDs (n>1) are outside this patch's scope.
    """
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    original = TokenizerManager._discard_pending_req_states
    if getattr(original, "_ripple_abort_before_discard", False):
        return

    @functools.wraps(original)
    def discard(self, obj):
        rids = [obj.rid] if getattr(obj, "is_single", True) else obj.rid
        for rid in dict.fromkeys(rids):
            state = self.rid_to_state.get(rid)
            # The start timestamp precedes dispatch; only the finish timestamp
            # proves dispatch succeeded. Never abort an unknown/predispatch ID:
            # scheduler aborts use prefix matching and could affect another ID.
            if (rid and state is not None and not state.finished
                    and state.time_stats.api_server_dispatch_finish_time > 0):
                self.abort_request(rid)
        return original(self, obj)

    discard._ripple_abort_before_discard = True
    TokenizerManager._discard_pending_req_states = discard
