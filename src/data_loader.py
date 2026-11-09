"""
Data loader module for customer complaint classification.
Supports:
- Loading the CFPB Consumer Complaint Database (or a balanced sample).
- Standardizing and mapping category taxonomies.
- Adding and merging any new external datasets (CSV, JSON, Parquet).
- Data deduplication and quality filtering.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm

DEFAULT_CFPB_CATEGORY_MAP = {
    # Credit Reporting
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit Reporting",
    "Credit reporting": "Credit Reporting",
    # Debt Collection
    "Debt collection": "Debt Collection",
    # Mortgages
    "Mortgage": "Mortgages & Home Loans",
    # Credit Cards
    "Credit card or prepaid card": "Credit Cards & Prepaid",
    "Credit card": "Credit Cards & Prepaid",
    "Prepaid card": "Credit Cards & Prepaid",
    # Banking & Accounts
    "Bank account or service": "Retail Banking & Accounts",
    "Checking or savings account": "Retail Banking & Accounts",
    # Student Loans
    "Student loan": "Student Loans",
    # Personal / Consumer Loans
    "Consumer Loan": "Personal & Consumer Loans",
    "Payday loan, title loan, or personal loan": "Personal & Consumer Loans",
    "Payday loan": "Personal & Consumer Loans",
    # Vehicle Loans
    "Vehicle loan or lease": "Vehicle Loans",
    # Money Transfers
    "Money transfer, virtual currency, or money service": "Money Transfers & Digital Currency",
    "Money transfers": "Money Transfers & Digital Currency",
    "Virtual currency": "Money Transfers & Digital Currency",
}


class ComplaintDataLoader:
    def __init__(self, raw_data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def load_cfpb_dataset(
        self,
        file_path: Optional[str] = None,
        max_samples_per_category: Optional[int] = 10000,
        category_mapping: Optional[Dict[str, str]] = None,
        chunk_size: int = 50000,
    ) -> pd.DataFrame:
        """
        Loads the CFPB complaint dataset in chunks, filters out empty narratives,
        maps categories, and returns a clean, balanced DataFrame.
        """
        mapping = category_mapping or DEFAULT_CFPB_CATEGORY_MAP
        target_path = Path(file_path) if file_path else self.raw_data_dir / "consumer_complaints_cfpb.csv"

        if not target_path.exists():
            raise FileNotFoundError(f"CFPB dataset not found at {target_path}")

        print(f"Loading CFPB dataset from: {target_path}")
        chunks_collected = []
        category_counts = {cat: 0 for cat in set(mapping.values())}

        reader = pd.read_csv(
            target_path,
            usecols=["Product", "Consumer complaint narrative"],
            chunksize=chunk_size,
            low_memory=False,
        )

        for chunk in tqdm(reader, desc="Processing CFPB chunks"):
            # Drop missing narratives
            valid = chunk[chunk["Consumer complaint narrative"].notna()].copy()
            valid["Consumer complaint narrative"] = valid["Consumer complaint narrative"].astype(str)
            valid = valid[valid["Consumer complaint narrative"].str.strip() != ""]

            # Map category
            valid["category"] = valid["Product"].map(mapping)
            valid = valid.dropna(subset=["category"])
            valid = valid.rename(columns={"Consumer complaint narrative": "text"})
            valid["source"] = "cfpb"

            # Balance sampling if required
            if max_samples_per_category:
                to_keep = []
                for cat, group in valid.groupby("category"):
                    current_count = category_counts.get(cat, 0)
                    needed = max_samples_per_category - current_count
                    if needed > 0:
                        selected = group.head(needed)
                        to_keep.append(selected)
                        category_counts[cat] += len(selected)
                if to_keep:
                    chunks_collected.append(pd.concat(to_keep, ignore_index=True))

                # Check if all categories reached quota
                if all(c >= max_samples_per_category for c in category_counts.values()):
                    print("Reached sample quota for all target categories.")
                    break
            else:
                chunks_collected.append(valid[["text", "category", "source"]])

        df = pd.concat(chunks_collected, ignore_index=True)
        df = self._clean_and_deduplicate(df)
        print(f"Loaded {len(df):,} CFPB complaints across {df['category'].nunique()} categories.")
        return df

    def add_external_dataset(
        self,
        file_path: Union[str, Path],
        text_column: str,
        category_column: str,
        category_mapping: Optional[Dict[str, str]] = None,
        source_name: str = "external",
    ) -> pd.DataFrame:
        """
        Loads and standardizes an arbitrary user dataset (CSV, JSON, Parquet).
        Allows seamless addition of new datasets to the NLP pipeline.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"Adding external dataset from: {file_path}")
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        if text_column not in df.columns or category_column not in df.columns:
            raise ValueError(f"Columns '{text_column}' or '{category_column}' not found in {df.columns.tolist()}")

        standardized = pd.DataFrame()
        standardized["text"] = df[text_column].astype(str).str.strip()
        
        if category_mapping:
            standardized["category"] = df[category_column].map(category_mapping)
        else:
            standardized["category"] = df[category_column].astype(str).str.strip()

        standardized["source"] = source_name
        standardized = standardized[standardized["text"] != ""]
        standardized = standardized.dropna(subset=["category"])
        standardized = self._clean_and_deduplicate(standardized)

        print(f"Added {len(standardized):,} valid samples from {source_name}.")
        return standardized

    def merge_and_save(
        self,
        datasets: List[pd.DataFrame],
        output_name: str = "complaints_master.csv",
    ) -> pd.DataFrame:
        """
        Merges multiple datasets into a single master training dataset,
        deduplicates, and saves to data/processed/.
        """
        merged = pd.concat(datasets, ignore_index=True)
        merged = self._clean_and_deduplicate(merged)
        output_path = self.processed_dir / output_name
        merged.to_csv(output_path, index=False)
        print(f"\nSaved master combined dataset ({len(merged):,} rows) to: {output_path}")
        print("\nClass distribution:")
        print(merged["category"].value_counts())
        return merged

    def _clean_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates and very short/noisy complaint texts."""
        df = df.dropna(subset=["text", "category"])
        # Filter minimum length (e.g. complaints less than 20 chars are usually junk)
        df = df[df["text"].str.len() >= 20]
        # Deduplicate identical complaint narratives
        df = df.drop_duplicates(subset=["text"]).copy()
        return df


if __name__ == "__main__":
    loader = ComplaintDataLoader()
    # Build a fast, balanced benchmark sample (5,000 samples per category)
    df_cfpb = loader.load_cfpb_dataset(max_samples_per_category=5000)
    loader.merge_and_save([df_cfpb], output_name="complaints_sample.csv")
