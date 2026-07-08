import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from litellm import Router

from .config import MAX_OUTPUT_TOKENS, MODELS
from .swap_manager import SwapManager

swap_manager = SwapManager(MODELS)

router = Router(
    model_list=[
        {
            "model_name": name,
            "litellm_params": {
                "model": f"hosted_vllm/{name}",
                "api_base": f"{base}/v1",
                "api_key": "EMPTY",
            },
        }
        for name, base in MODELS.items()
    ],
    num_retries=0,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await swap_manager.startup_reconcile()
    yield


app = FastAPI(lifespan=lifespan)


def _model_error(model: str):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"The model '{model}' does not exist or you do not have access to it.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    infos = await asyncio.gather(*(swap_manager.model_info(name) for name in MODELS))
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": (info or {}).get("created") or 0,
                "owned_by": "vllm",
                "root": (info or {}).get("root"),
                "max_model_len": (info or {}).get("max_model_len"),
                "max_output_tokens": MAX_OUTPUT_TOKENS.get(name),
            }
            for name, info in zip(MODELS, infos)
        ],
    }


async def _dispatch(body: dict, call):
    model = body.get("model")
    if model not in MODELS:
        return _model_error(model)

    await swap_manager.acquire(model)
    released = False

    async def release_once():
        nonlocal released
        if not released:
            released = True
            await swap_manager.release(model)

    try:
        if body.get("stream"):
            response_iter = await call(**body)

            async def gen():
                try:
                    async for chunk in response_iter:
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    await release_once()

            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            response = await call(**body)
            return JSONResponse(response.model_dump())
    except Exception:
        await release_once()
        raise
    finally:
        if not body.get("stream"):
            await release_once()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await _dispatch(body, router.acompletion)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    return await _dispatch(body, router.atext_completion)
