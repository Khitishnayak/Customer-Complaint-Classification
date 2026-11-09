"""
Model training and evaluation pipeline for customer complaint classification.
Builds an optimized TF-IDF + Calibrated Linear Model pipeline.
Evaluates classification performance (Accuracy, Macro F1, Per-class metrics)
and saves the model artifact for production serving.
"""

import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.preprocessor import prepare_dataset, TextPreprocessor


def train_complaint_classifier(
    data_path: str = "data/processed/complaints_sample.csv",
    models_dir: str = "models",
    max_features: int = 40000,
    ngram_range: tuple = (1, 2)
):
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading preprocessed dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Clean text and split
    X_train, X_test, y_train, y_test, label_encoder = prepare_dataset(df)

    # Build Pipeline: TF-IDF + Calibrated Logistic Regression
    print("\nBuilding NLP Classification Pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=3,
            stop_words="english"
        )),
        ("classifier", LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on unseen test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=" * 60)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1 Score:   {macro_f1 * 100:.2f}%")
    print("=" * 60)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
    print("\nDetailed Classification Report:")
    print(report)

    # Save artifacts
    model_artifact = {
        "pipeline": pipeline,
        "classes": label_encoder.classes_,
        "preprocessor": TextPreprocessor()
    }

    output_model_path = models_path / "complaint_classifier_pipeline.joblib"
    joblib.dump(model_artifact, output_model_path)
    print(f"\nTrained model pipeline successfully saved to: {output_model_path}")

    return pipeline, label_encoder


if __name__ == "__main__":
    train_complaint_classifier()
