# data_processing.py
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentiment_model import SentimentModel
from restaurant_utils import (
    extract_restaurant_slug,
    slug_to_display_name,
    restaurant_matches,
)

CHUNK_SIZE = 100
MAX_THREADS = 32


class ReviewDataset:
    """
    Manages loading the Kaggle dataset and running multi-threaded analysis.

    Phase 1:
        - Scan the whole dataset in chunks of 100 rows.
        - Each chunk is processed by one thread that counts restaurants.
        - Returns the list of restaurants + Phase 1 thread traces.

    Phase 2:
        - For a given restaurant name, scan the whole dataset again in chunks
          of 100 rows with multiple threads.
        - Each thread:
            * checks if a row belongs to the restaurant
            * computes a sentiment-based rating (0-5) from review_text
            * collects matching reviews in memory
        - Returns all matching reviews + Phase 2 thread traces.

    Phase 3:
        - For the restaurant's reviews, at most 10 years (from today) are
          considered.
        - For each year, one thread computes the average rating.
        - Returns ratings by year + Phase 3 thread traces.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """
        Lazily load and prepare the dataset.

        We expect at least: yelp_url, date, review_text.
        """
        if self._df is not None:
            return

        with self._lock:
            if self._df is not None:
                return

            df = pd.read_csv(self.csv_path)

            required = {"yelp_url", "date", "review_text"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"CSV must contain columns: {', '.join(sorted(required))}. Missing: {', '.join(sorted(missing))}"
                )

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = df["date"].dt.year

            # Extract restaurant slug from Yelp URL and a prettier name
            df["restaurant_slug"] = df["yelp_url"].apply(extract_restaurant_slug)
            df["restaurant_name"] = df["restaurant_slug"].apply(slug_to_display_name)

            self._df = df

    def _iter_ranges(self, n: int) -> List[tuple]:
        """
        Build index ranges of size CHUNK_SIZE.
        """
        return [(i, min(i + CHUNK_SIZE, n)) for i in range(0, n, CHUNK_SIZE)]

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def get_all_restaurants_with_stats(self) -> Dict[str, Any]:
        """
        Phase 1:
        - Use multiple threads (one per 100 rows) to scan the dataset
          and count how many reviews each restaurant has.
        - Also return traces about each thread.

        Returns JSON-like dict:
        {
          "total_reviews": int,
          "restaurants": [
             {"slug": str, "name": str, "review_count": int},
             ...
          ],
          "threads": [
             {
               "thread_index": int,
               "rows_processed": int,
               "unique_restaurants": int,
               "duration_ms": float
             },
             ...
          ]
        }
        """
        self._ensure_loaded()
        df = self._df
        if df is None:
            raise RuntimeError("Dataset not loaded.")

        total_reviews = int(len(df))
        if total_reviews == 0:
            return {
                "total_reviews": 0,
                "restaurants": [],
                "threads": [],
            }

        ranges = self._iter_ranges(total_reviews)
        all_counts: Counter[str] = Counter()
        thread_traces: List[Dict[str, Any]] = []

        def process_chunk(idx: int, start: int, end: int):
            start_t = time.perf_counter()
            chunk = df.iloc[start:end]
            # count restaurants in this chunk
            local_counter = Counter(chunk["restaurant_slug"].tolist())
            elapsed = time.perf_counter() - start_t
            stats = {
                "thread_index": idx,
                "rows_processed": int(end - start),
                "unique_restaurants": int(len(local_counter)),
                "duration_ms": round(elapsed * 1000.0, 2),
            }
            return stats, local_counter

        max_workers = min(MAX_THREADS, len(ranges)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_chunk, idx, start, end): idx
                for idx, (start, end) in enumerate(ranges)
            }

            for future in as_completed(future_to_idx):
                stats, local_counter = future.result()
                thread_traces.append(stats)
                all_counts.update(local_counter)

        # Map slug -> pretty display name
        slug_to_name_map = (
            df.drop_duplicates("restaurant_slug")
            .set_index("restaurant_slug")["restaurant_name"]
            .to_dict()
        )

        restaurants_list: List[Dict[str, Any]] = []
        for slug, count in all_counts.most_common():
            if not slug:
                continue
            restaurants_list.append(
                {
                    "slug": slug,
                    "name": slug_to_name_map.get(slug, slug_to_display_name(slug)),
                    "review_count": int(count),
                }
            )

        return {
            "total_reviews": total_reviews,
            "restaurants": restaurants_list,
            "threads": thread_traces,
        }

    # ------------------------------------------------------------------
    # Phase 2 & 3
    # ------------------------------------------------------------------

    def analyze_restaurant_with_stats(self, restaurant_name: str) -> Dict[str, Any]:
        """
        Phase 2:
            - Use threads (one per 100 rows) to scan the dataset and
              find all reviews for the target restaurant.
            - Each matching review gets a sentiment-based rating.
        Phase 3:
            - For at most 10 years (from today), each year is handled
              by one thread that computes the average rating.

        Returns a JSON-like dict with:
          - ratings_by_year
          - global_rate (average of yearly ratings)
          - phase2 / phase3 thread traces
        """
        self._ensure_loaded()
        df = self._df
        if df is None:
            raise RuntimeError("Dataset not loaded.")

        total_dataset_reviews = int(len(df))
        if total_dataset_reviews == 0:
            return {
                "restaurant": restaurant_name,
                "search_query": restaurant_name,
                "dataset_total_reviews": 0,
                "restaurant_total_reviews": 0,
                "ratings_by_year": {},
                "global_rate": None,
                "phase2": {
                    "total_reviews_processed": 0,
                    "total_threads": 0,
                    "threads": [],
                },
                "phase3": {
                    "total_year_threads": 0,
                    "threads": [],
                },
            }

        ranges = self._iter_ranges(total_dataset_reviews)
        model = SentimentModel.instance()

        phase2_traces: List[Dict[str, Any]] = []
        matched_reviews: List[Dict[str, Any]] = []

        def process_chunk(idx: int, start: int, end: int):
            start_t = time.perf_counter()
            chunk = df.iloc[start:end]
            rows_processed = int(end - start)
            rows_matched: List[Dict[str, Any]] = []

            for row in chunk.itertuples(index=False):
                slug = getattr(row, "restaurant_slug", "")
                if not restaurant_matches(slug, restaurant_name):
                    continue

                text = getattr(row, "review_text", "") or ""
                score = float(model.score_review(text))
                date = getattr(row, "date", None)
                year = getattr(row, "year", None)
                yelp_url = getattr(row, "yelp_url", "")

                rows_matched.append(
                    {
                        "restaurant_slug": slug,
                        "review_text": text,
                        "review_result": score,
                        "date": date,
                        "year": int(year) if pd.notnull(year) else None,
                        "yelp_url": yelp_url,
                    }
                )

            elapsed = time.perf_counter() - start_t
            stats = {
                "thread_index": idx,
                "rows_processed": rows_processed,
                "matched_reviews": len(rows_matched),
                "duration_ms": round(elapsed * 1000.0, 2),
            }
            return stats, rows_matched

        max_workers = min(MAX_THREADS, len(ranges)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_chunk, idx, start, end): idx
                for idx, (start, end) in enumerate(ranges)
            }

            for future in as_completed(future_to_idx):
                stats, rows_matched = future.result()
                phase2_traces.append(stats)
                matched_reviews.extend(rows_matched)

        total_matched = len(matched_reviews)

        # If no review for this restaurant, still return Phase 2 traces
        if total_matched == 0:
            return {
                "restaurant": restaurant_name,
                "search_query": restaurant_name,
                "dataset_total_reviews": total_dataset_reviews,
                "restaurant_total_reviews": 0,
                "ratings_by_year": {},
                "global_rate": None,
                "phase2": {
                    "total_reviews_processed": total_dataset_reviews,
                    "total_threads": len(ranges),
                    "threads": phase2_traces,
                },
                "phase3": {
                    "total_year_threads": 0,
                    "threads": [],
                },
            }

        df_rest = pd.DataFrame(matched_reviews)

        # ------------------------------------------------------------------
        # Phase 3: one thread per year (max 10 years from today)
        # ------------------------------------------------------------------
        now_year = datetime.utcnow().year
        min_year = now_year - 9  # last 10 years inclusive

        df_rest = df_rest.dropna(subset=["year"])
        df_rest["year"] = df_rest["year"].astype(int)
        df_rest = df_rest[
            (df_rest["year"] >= min_year) & (df_rest["year"] <= now_year)
        ]

        ratings_by_year: Dict[int, float] = {}
        phase3_traces: List[Dict[str, Any]] = []

        if not df_rest.empty:
            years = sorted(df_rest["year"].unique().tolist())

            def compute_year(year: int):
                start_t = time.perf_counter()
                subset = df_rest[df_rest["year"] == year]
                rows_count = int(len(subset))
                if rows_count == 0:
                    elapsed_inner = time.perf_counter() - start_t
                    stats_inner = {
                        "year": int(year),
                        "rows_processed": 0,
                        "duration_ms": round(elapsed_inner * 1000.0, 2),
                    }
                    return year, None, stats_inner

                avg = float(subset["review_result"].mean())
                elapsed_inner = time.perf_counter() - start_t
                stats_inner = {
                    "year": int(year),
                    "rows_processed": rows_count,
                    "duration_ms": round(elapsed_inner * 1000.0, 2),
                }
                return year, avg, stats_inner

            max_workers_years = min(MAX_THREADS, len(years)) or 1
            with ThreadPoolExecutor(max_workers=max_workers_years) as executor:
                future_to_year = {
                    executor.submit(compute_year, year): year for year in years
                }

                for future in as_completed(future_to_year):
                    year, avg, stats = future.result()
                    phase3_traces.append(stats)
                    if avg is not None:
                        ratings_by_year[int(year)] = round(avg, 2)

        # Compute global rate as an average of yearly averages
        if ratings_by_year:
            valid_ratings = list(ratings_by_year.values())
            global_rate = round(sum(valid_ratings) / len(valid_ratings), 2)
        else:
            global_rate = None

        # Try to use a nicer restaurant name if we have a slug
        first_slug = matched_reviews[0].get("restaurant_slug") or ""
        nice_name = slug_to_display_name(first_slug) if first_slug else restaurant_name

        ratings_str_keys = {str(y): r for y, r in sorted(ratings_by_year.items())}

        return {
            "restaurant": nice_name,
            "search_query": restaurant_name,
            "dataset_total_reviews": total_dataset_reviews,
            "restaurant_total_reviews": total_matched,
            "ratings_by_year": ratings_str_keys,
            "global_rate": global_rate,
            "phase2": {
                "total_reviews_processed": total_dataset_reviews,
                "total_threads": len(ranges),
                "threads": phase2_traces,
            },
            "phase3": {
                "total_year_threads": len(phase3_traces),
                "threads": phase3_traces,
            },
        }
