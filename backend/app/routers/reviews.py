"""Review endpoints inherit the per-account session middleware."""
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import human_review
router = APIRouter()

class ReviewDecision(BaseModel):
    action: Literal['accept', 'reject']
    revision: int = Field(ge=0)
    bbox: Optional[list[float]] = None
    class_name: Optional[str] = None

@router.get('/api/reviews/jobs')
def jobs(dataset_id: Optional[str]=None):
    return {'jobs': human_review.list_jobs(dataset_id)}

@router.get('/api/reviews/jobs/{job_id}')
def job(job_id: str):
    return human_review.get_job(job_id)

@router.get('/api/reviews/jobs/{job_id}/candidates/{candidate_id}')
def candidate(job_id: str, candidate_id: str):
    return human_review.candidate_detail(job_id, candidate_id)

@router.get('/api/reviews/jobs/{job_id}/candidates/{candidate_id}/image')
def image(job_id: str, candidate_id: str):
    return FileResponse(human_review.candidate_image(job_id, candidate_id), headers={'Cache-Control': 'private, max-age=60'})

@router.post('/api/reviews/jobs/{job_id}/candidates/{candidate_id}/decision')
def decision(job_id: str, candidate_id: str, body: ReviewDecision):
    try:
        return human_review.decide(job_id, candidate_id, body.action, body.revision, bbox=body.bbox, class_name=body.class_name)
    except OSError as exc:
        raise HTTPException(503, '保存复核记录失败，请刷新后重试') from exc
