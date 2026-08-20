"""Video Triggers module for instructional clinical videos (Breathing Exercises & Inhaler Techniques)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VideoTrigger:
    trigger_id: str
    keywords: List[str]
    video_path: str   # Served path, e.g. "/static/videos/breathing_exercise.mp4"
    title: str
    title_ar: str


VIDEO_TRIGGERS: List[VideoTrigger] = [
    VideoTrigger(
        trigger_id="breathing_exercise",
        keywords=[
            r"breathing\s+exercise",
            r"breathing\s+technique",
            r"how\s+do\s+i\s+breathe",
            r"pursed\s+lip\s+breathing",
            r"diaphragmatic\s+breathing",
            r"how\s+to\s+calm\s+my\s+breathing",
            r"breathing\s+timer",
            r"تمارين\s+التنفس",
            r"تمرين\s+تنفس",
            r"كيف\s+اتنفس",
            r"طريقة\s+التنفس",
            r"تهدئة\s+التنفس",
        ],
        video_path="https://raw.githubusercontent.com/Jasminex6/MindMesh-RAG/main/static/videos/breathing_exercise.mp4",
        title="Guided Breathing Exercise",
        title_ar="تمرين التنفس الموجه",
    ),
    VideoTrigger(
        trigger_id="inhaler_technique",
        keywords=[
            r"how\s+to\s+use\s+an?\s+inhaler",
            r"inhaler\s+video",
            r"inhaler\s+technique",
            r"how\s+do\s+i\s+use\s+my\s+inhaler",
            r"breathilezer",
            r"breathalizer",
            r"breathalyzer",
            r"nebulizer\s+video",
            r"spacer\s+technique",
            r"spacer\s+video",
            r"طريقة\s+استخدام\s+البخاخ",
            r"فيديو\s+البخاخ",
            r"كيف\s+استخدم\s+البخاخ",
            r"استعمال\s+البخاخ",
            r"استخدام\s+البخاخ",
        ],
        video_path="https://raw.githubusercontent.com/Jasminex6/MindMesh-RAG/main/static/videos/inhaler_technique.mp4",
        title="Correct Inhaler & Spacer Technique",
        title_ar="طريقة استخدام البخاخ والمباعد الصحيحة",
    ),
]

_COMPILED = [
    (t, re.compile("|".join(t.keywords), re.IGNORECASE))
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
