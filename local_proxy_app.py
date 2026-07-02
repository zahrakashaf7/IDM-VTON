"""
Local FastAPI proxy for IDM-VTON running on Google Colab.

Runs entirely on your local machine -- no GPU required here. It forwards
requests to the real inference server running on Colab (started via
fastapi_app.py + ngrok, see the Colab notebook's Part 2). Your local clients
always call this stable local address; only this proxy needs to know about
the Colab ngrok URL, which can change between Colab sessions.

Setup:
    pip install fastapi "uvicorn[standard]" python-multipart httpx

Run:
    COLAB_API_URL="https://xxxx.ngrok-free.app" uvicorn local_proxy_app:app --host 0.0.0.0 --port 8000

Or set COLAB_API_URL below directly, or export it as an environment variable
before starting uvicorn. Update it any time the Colab ngrok URL changes --
no need to restart your own clients, just this proxy.
"""

import os
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLAB_API_URL = os.environ.get("COLAB_API_URL", "").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("COLAB_API_TIMEOUT", "300"))  # generation can take a while

app = FastAPI(title="IDM-VTON Local Proxy", description="Forwards requests to a Colab-hosted IDM-VTON API.")


def _require_configured_url():
    if not COLAB_API_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "COLAB_API_URL is not set. Start this proxy with e.g. "
                "COLAB_API_URL=https://xxxx.ngrok-free.app uvicorn local_proxy_app:app ..."
            ),
        )


@app.get("/health")
async def health():
    """Checks both this proxy and the upstream Colab server."""
    _require_configured_url()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{COLAB_API_URL}/health")
            resp.raise_for_status()
            return {"proxy": "ok", "colab_target": COLAB_API_URL, "colab_status": resp.json()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Colab endpoint unreachable: {e}")


@app.post("/tryon")
async def tryon(
    human_img: UploadFile = File(...),
    garm_img: UploadFile = File(...),
    garment_des: str = Form("clothing item"),
    use_auto_mask: bool = Form(True),
    use_auto_crop: bool = Form(False),
    denoise_steps: int = Form(30, ge=1, le=100),
    seed: Optional[int] = Form(42),
    mask_img: Optional[UploadFile] = File(None),
):
    _require_configured_url()

    files = {
        "human_img": (human_img.filename, await human_img.read(), human_img.content_type),
        "garm_img": (garm_img.filename, await garm_img.read(), garm_img.content_type),
    }
    if mask_img is not None:
        files["mask_img"] = (mask_img.filename, await mask_img.read(), mask_img.content_type)

    data = {
        "garment_des": garment_des,
        "use_auto_mask": str(use_auto_mask).lower(),
        "use_auto_crop": str(use_auto_crop).lower(),
        "denoise_steps": str(denoise_steps),
        "seed": "" if seed is None else str(seed),
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{COLAB_API_URL}/tryon", files=files, data=data)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Colab endpoint timed out -- generation may be taking too long.")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Colab endpoint: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return StreamingResponse(iter([resp.content]), media_type=resp.headers.get("content-type", "image/png"))
