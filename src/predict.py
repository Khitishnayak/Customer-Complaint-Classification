"""
Inference module for customer complaint classification and urgency scoring.
Loads trained artifacts and provides real-time single and batch prediction methods.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import joblib
import numpy as np
import pandas as pd


class ComplaintClassifierService:
    def __init__(self, model_path: str = "models/complaint_classifier_pipeline.joblib"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}. Run `src/train.py` first.")

        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.classes = artifact["classes"]
        self.preprocessor = artifact["preprocessor"]

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Classifies a single complaint text.
        Returns:
            - predicted_category
            - confidence
            - top_predictions (top k categories with confidence probabilities)
            - urgency (priority rating and risk keywords detected)
        """
        cleaned = self.preprocessor.clean_text(text)
        if not cleaned:
            return {
                "error": "Input text is empty or contains no valid characters."
            }

        # Predict probabilities
        probs = self.pipeline.predict_proba([cleaned])[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        top_preds = [
            {"category": self.classes[i], "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        urgency_info = self.preprocessor.detect_urgency(text)

        return {
            "predicted_category": top_preds[0]["category"],
            "confidence": top_preds[0]["confidence"],
            "top_candidates": top_preds,
            "urgency": urgency_info,
            "cleaned_text_preview": cleaned[:120] + ("..." if len(cleaned) > 120 else "")
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch classification for multiple complaints (e.g. from a CSV upload)."""
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    service = ComplaintClassifierService()

    sample_complaints = [
        "My escrow account was improperly calculated and the mortgage company is threatening foreclosure on my home!",
        "Someone opened a credit card in my name without my authorization, this is fraud and identity theft and I am calling my lawyer.",
        "I transferred $500 via wire to my family overseas two weeks ago and the cryptocurrency / money transfer never arrived.",
        "The debt collectors keep calling me at work 10 times a day demanding payment for a debt I already paid off.",
    ]

    print("\n--- SAMPLE INFERENCE RUN ---\n")
    for text in sample_complaints:
        result = service.predict(text)
        print(f"Complaint: {text}")
        print(f" -> Predicted Category: {result['predicted_category']} (Confidence: {result['confidence']*100:.1f}%)")
        print(f" -> Urgency / Escalation: {result['urgency']['priority']} (Triggers: {result['urgency']['triggers']})")
        print("-" * 60)
