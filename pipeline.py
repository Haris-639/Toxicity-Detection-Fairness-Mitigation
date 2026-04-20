import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from transformers import AutoTokenizer, DistilBertForSequenceClassification


BLOCKLIST = {
    "direct_threat": [
        re.compile(r"\b(i\s*(?:will|'ll|am\s+going\s+to|gonna)\s+)(kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re|\s+are)?\s+going\s+to\s+die\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|'ll|am\s+going\s+to|gonna)\s+find\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bsomeone\s+should\s+(?:kill|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|'ll|am\s+going\s+to|gonna)\s+beat\s+you\s+up\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        re.compile(r"\b(?:go\s+)?kill\s+yourself\b", re.IGNORECASE),
        re.compile(r"\byou\s+should\s+kill\s+yourself\b", re.IGNORECASE),
        re.compile(r"\bnobody\s+would\s+miss\s+you\s+if\s+you\s+died\b", re.IGNORECASE),
        re.compile(r"\bdo\s+everyone\s+a\s+favou?r\s+and\s+disappear\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi\s+know\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:will|'ll|am\s+going\s+to|gonna)\s+post\s+your\s+address\b", re.IGNORECASE),
        re.compile(r"\bi\s+found\s+your\s+real\s+name\b", re.IGNORECASE),
        re.compile(r"\beveryone\s+will\s+know\s+who\s+you\s+really\s+are\b", re.IGNORECASE),
    ],
    "dehumanization": [
        re.compile(r"\b(?:they|these\s+people|those\s+people|[a-z]+)\s+are\s+not\s+(?:human|people|person)\b", re.IGNORECASE),
        re.compile(r"\b(?:they|these\s+people|those\s+people|[a-z]+)\s+are\s+animals\b", re.IGNORECASE),
        re.compile(r"\b(?:they|these\s+people|those\s+people|[a-z]+)\s+should\s+be\s+exterminated\b", re.IGNORECASE),
        re.compile(r"\b(?:they|these\s+people|those\s+people|[a-z]+)\s+are\s+a\s+disease\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        re.compile(r"\beveryone\s+report\s+(?=@?\w+)", re.IGNORECASE),
        re.compile(r"\blet'?s\s+all\s+go\s+after\s+\w+\b", re.IGNORECASE),
        re.compile(r"\b(?:mass\s+report|raid)\s+(?:this\s+)?(?:account|profile)\b", re.IGNORECASE),
    ],
}


def input_filter(text: str) -> Optional[Dict[str, Any]]:
    """Returns a block decision dict if matched, else None."""
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


class ScorePassThroughEstimator(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible estimator for calibrating precomputed model scores."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.asarray(X["score"]).reshape(-1, 1)
        return np.hstack([1 - s, s])


class ModerationPipeline:
    def __init__(
        self,
        model_dir: str,
        lower_threshold: float = 0.4,
        upper_threshold: float = 0.6,
        device: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.lower_threshold = float(lower_threshold)
        self.upper_threshold = float(upper_threshold)

        self.calibrator: Optional[CalibratedClassifierCV] = None

    def set_uncertainty_band(self, lower: float, upper: float) -> None:
        self.lower_threshold = float(lower)
        self.upper_threshold = float(upper)

    def _score_texts(self, texts: List[str], batch_size: int = 64, max_len: int = 128) -> np.ndarray:
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                enc.pop('token_type_ids', None)  # DistilBERT doesn't use token_type_ids
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                probs.extend(p)
        return np.asarray(probs)

    def fit_calibrator(self, calibration_texts: List[str], calibration_labels: List[int], batch_size: int = 64) -> None:
        raw_scores = self._score_texts(calibration_texts, batch_size=batch_size)
        X_cal = pd.DataFrame({"score": raw_scores})
        y_cal = np.asarray(calibration_labels).astype(int)

        calibrator = CalibratedClassifierCV(
            estimator=ScorePassThroughEstimator(),
            method="isotonic",
            cv=3,
        )
        calibrator.fit(X_cal, y_cal)
        self.calibrator = calibrator

    def _calibrated_score(self, raw_score: float) -> float:
        if self.calibrator is None:
            return float(raw_score)
        X = pd.DataFrame({"score": [float(raw_score)]})
        return float(self.calibrator.predict_proba(X)[0, 1])

    def predict(self, text: str) -> Dict[str, Any]:
        pre = input_filter(text)
        if pre is not None:
            return pre

        raw_score = float(self._score_texts([text])[0])
        confidence = self._calibrated_score(raw_score)

        if confidence >= self.upper_threshold:
            return {
                "decision": "block",
                "layer": "model",
                "confidence": confidence,
                "raw_score": raw_score,
            }
        if confidence <= self.lower_threshold:
            return {
                "decision": "allow",
                "layer": "model",
                "confidence": confidence,
                "raw_score": raw_score,
            }
        return {
            "decision": "review",
            "layer": "model",
            "confidence": confidence,
            "raw_score": raw_score,
        }
