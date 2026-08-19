"""
video_triggers.py (FINAL)

Detects when a user's question should trigger a specific instructional
video (e.g. breathing exercise, inhaler technique) and returns it as part
of the FastAPI chat response, so the frontend can play it.

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
1. Put this file next to your demo.py (or wherever your FastAPI app lives).
2. Put your video files in:  ./static/videos/breathing_exercise.mp4
                              ./static/videos/inhaler_technique.mp4
3. In demo.py, add the import + static mount + response field + endpoint
   changes shown at the bottom of this file (under "INTEGRATION -
   COPY INTO demo.py").
4. Frontend: render <video controls autoplay loop src="{video_url}">
   whenever the response includes a video_url.
--------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VideoTrigger:
    trigger_id: str
    keywords: List[str]
    video_path: str   # served path, e.g. "/static/videos/breathing_exercise.mp4"
    title: str


# ---------------------------------------------------------------------------
# Registry - add one entry per video you want to trigger.
# ---------------------------------------------------------------------------
VIDEO_TRIGGERS: List[VideoTrigger] = [
    VideoTrigger(
        trigger_id="breathing_exercise",
        keywords=[
            "breathing exercise",
            "breathing technique",
            "how do i breathe",
            "pursed lip breathing",
            "diaphragmatic breathing",
            "how to calm my breathing",
        ],
        video_path="/static/videos/breathing_exercise.mp4",
        title="Guided Breathing Exercise",
    ),
    VideoTrigger(
        trigger_id="inhaler_technique",
        keywords=[
            "how to use an inhaler",
            "inhaler technique",
            "how do i use my inhaler",
            "inhaler tutorial",
            "using an inhaler",
            "proper inhaler use",
        ],
        video_path="/static/videos/inhaler_technique.mp4",
        title="Correct Inhaler Technique",
    ),
    # Add more triggers here.
]

_COMPILED = [
    (t, re.compile("|".join(re.escape(k) for k in t.keywords), re.IGNORECASE))
    for t in VIDEO_TRIGGERS
]


def detect_video(question: str) -> Optional[VideoTrigger]:
    """Returns the first matching VideoTrigger for a question, or None."""
    if not question:
        return None
    for trigger, pattern in _COMPILED:
        if pattern.search(question):
            return trigger
    return None


if __name__ == "__main__":
    tests = [
        "Can you show me a breathing exercise for an asthma attack?",
        "How do I use my inhaler correctly?",
        "What causes asthma?",
    ]
    for q in tests:
        result = detect_video(q)
        print(f"{q!r} -> {result.trigger_id if result else None}")


# ===========================================================================
# INTEGRATION - COPY INTO demo.py
# ===========================================================================
#
# from fastapi.staticfiles import StaticFiles
# from video_triggers import detect_video
#
# app.mount("/static", StaticFiles(directory="static"), name="static")
#
# class ChatResponse(BaseModel):
#     answer: str
#     sources: list
#     video_url: Optional[str] = None
#     video_title: Optional[str] = None
#
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     answer, sources = run_rag_pipeline(request.question)  # your existing call
#     video = detect_video(request.question)
#
#     return ChatResponse(
#         answer=answer,
#         sources=sources,
#         video_url=video.video_path if video else None,
#         video_title=video.title if video else None,
#     )
#
# ---------------------------------------------------------------------------
# FRONTEND - wherever the response is rendered
# ---------------------------------------------------------------------------
#
# {% if video_url %}
# <video controls autoplay loop src="{{ video_url }}" style="max-width: 100%;"></video>
# {% endif %}
#
# ===========================================================================
