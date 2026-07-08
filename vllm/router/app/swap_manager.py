import asyncio
import httpx

from .config import MODELS

# Wake-from-sleep is seconds, not minutes (cold boot / graph capture is the
# only minutes-scale operation, and that's gated by the compose healthcheck,
# not this client). 300s is a generous safety margin, not a tuned bound.
ADMIN_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)


class SwapManager:
    def __init__(self, models: dict[str, str]):
        self._models = models
        self._client = httpx.AsyncClient(timeout=ADMIN_TIMEOUT)
        self._cond = asyncio.Condition()
        self._current: str | None = None
        self._inflight: dict[str, int] = {m: 0 for m in models}
        # None means "not yet fetched or last attempt failed" -- retried
        # lazily on the next call rather than cached as a permanent miss,
        # since a model can still be booting when first queried.
        self._model_info: dict[str, dict | None] = {m: None for m in models}

    async def model_info(self, name: str) -> dict | None:
        """vLLM's own /v1/models reports the real root checkpoint,
        launch timestamp, and --max-model-len for this served name
        (ModelCard), even while sleeping -- that endpoint doesn't touch
        GPU state. This is the source of truth so the proxy's /v1/models
        never drifts from what each container was actually launched
        with."""
        if self._model_info[name] is not None:
            return self._model_info[name]
        base = self._models[name]
        try:
            r = await self._client.get(f"{base}/v1/models")
            r.raise_for_status()
            data = r.json()["data"]
            card = data[0] if data else {}
            info = {
                "max_model_len": card.get("max_model_len"),
                "root": card.get("root"),
                "created": card.get("created"),
            }
        except Exception:
            info = None
        self._model_info[name] = info
        return info

    async def startup_reconcile(self) -> None:
        """Run once at FastAPI startup. Vllm containers persist across
        fastapi-router restarts, so don't assume nothing is awake.
        A model unreachable at startup (crash-looping, still booting, not
        started at all) must not take down routing for the other models —
        skip it here; the first real request for it re-checks liveness via
        _swap_to and fails loudly for that request only."""
        awake: list[str] = []
        for name, base in self._models.items():
            try:
                r = await self._client.get(f"{base}/is_sleeping")
                r.raise_for_status()
                if not r.json()["is_sleeping"]:
                    awake.append(name)
            except Exception:
                continue
        for extra in awake[1:]:
            await self._sleep(extra)
        self._current = awake[0] if awake else None

    async def _sleep(self, name: str) -> None:
        base = self._models[name]
        r = await self._client.post(f"{base}/sleep", params={"level": 2})
        r.raise_for_status()

    async def _wake(self, name: str) -> None:
        """Level-2 sleep requires this exact 4-call sequence -- confirmed
        against vLLM's own blog ("For Level 2 sleep, you must call
        reload_weights and reset_prefix_cache after waking") and reproduced
        empirically on this box: skipping /collective_rpc reload_weights
        between the two /wake_up calls reallocates memory but never copies
        weight VALUES into it, producing gibberish output with is_sleeping
        still reporting correctly and HTTP 200 throughout (matches
        vllm-project/vllm#29341). A `reload_weights` disk read on a
        page-cache-cold large checkpoint can take well over a minute --
        that is real work, not a hang; ADMIN_TIMEOUT.read=300.0 accounts
        for it. Do not remove this step."""
        base = self._models[name]
        for path, params, body in (
            ("/wake_up", {"tags": "weights"}, None),
            ("/collective_rpc", None, {"method": "reload_weights"}),
            ("/wake_up", {"tags": "kv_cache"}, None),
            ("/reset_prefix_cache", None, None),
        ):
            if body is not None:
                r = await self._client.post(f"{base}{path}", json=body)
            else:
                r = await self._client.post(f"{base}{path}", params=params)
            r.raise_for_status()

    async def _swap_to(self, target: str) -> None:
        base = self._models[target]
        try:
            r = await self._client.get(f"{base}/is_sleeping")
            r.raise_for_status()
            target_awake = not r.json()["is_sleeping"]
            if self._current is not None and self._current != target:
                await self._sleep(self._current)
            if not target_awake:
                await self._wake(target)
        except Exception:
            self._current = None
            raise
        self._current = target

    async def acquire(self, model: str) -> None:
        if model not in self._models:
            raise KeyError(model)
        async with self._cond:
            while True:
                if self._current == model:
                    self._inflight[model] += 1
                    return
                if self._current is None or self._inflight[self._current] == 0:
                    await self._swap_to(model)
                    self._cond.notify_all()
                    continue
                await self._cond.wait()

    async def release(self, model: str) -> None:
        async with self._cond:
            self._inflight[model] -= 1
            self._cond.notify_all()
