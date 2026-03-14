from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import get_engine_manager
from ..errors import APIError, ErrorCode, success_response
from ..schemas import RewriteAutoRequest, RewriteInstructionRequest

router = APIRouter()


@router.post("/auto")
async def rewrite_auto(req: RewriteAutoRequest):
    em = get_engine_manager()
    if em.rewrite_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "RewriteEngine not loaded", 503)
    try:
        result = em.rewrite_engine.auto_rewrite(req.text)
    except Exception as e:
        raise APIError(ErrorCode.ENGINE_ERROR, str(e))
    return success_response({"rewritten_text": result})


@router.post("/instruction")
async def rewrite_instruction(req: RewriteInstructionRequest):
    em = get_engine_manager()
    if em.rewrite_engine is None:
        raise APIError(ErrorCode.ENGINE_ERROR, "RewriteEngine not loaded", 503)
    try:
        result = em.rewrite_engine.rewrite_with_instruction(req.text, req.instruction)
    except Exception as e:
        raise APIError(ErrorCode.ENGINE_ERROR, str(e))
    return success_response({"rewritten_text": result})
