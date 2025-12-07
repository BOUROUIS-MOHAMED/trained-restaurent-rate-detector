# data_processing.py
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        - Each chunk is processed by one thread that groups reviews by restaurant.
        - Builds an in-memory cache that maps each restaurant slug to all its review indices.
        - Returns the list of restaurants with their reviews + Phase 1 thread traces.

    Phase 2 (merged analysis for a single restaurant):
        - Uses ONLY the subset of reviews that belong to that restaurant (from Phase 1 cache).
        - Threads (chunks of 100 reviews) compute sentiment and local per-year aggregates.
        - The main thread merges those per-year aggregates to produce:
            * ratings_by_year
            * global_rate
            * per-thread traces.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._lock = threading.Lock()

        # Phase 1 cache:
        #   slug -> list of global DataFrame indices
        self._restaurants_indices: Optional[Dict[str, List[int]]] = None
        #   list of per-thread traces
        self._phase1_threads_cache: Optional[List[Dict[str, Any]]] = None

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
                    f"CSV must contain columns: {', '.join(sorted(required))}. "
                    f"Missing: {', '.join(sorted(missing))}"
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
        Each range is a tuple (start, end) with end being exclusive.
        """
        return [(i, min(i + CHUNK_SIZE, n)) for i in range(0, n, CHUNK_SIZE)]

    def _ensure_phase1_cache(self) -> None:
        """
        Build (once) the Phase 1 cache that maps each restaurant slug to
        a list of DataFrame indices for all its reviews, and records
        per-thread traces.

        Subsequent calls will reuse this cache.
        """
        self._ensure_loaded()

        if self._restaurants_indices is not None and self._phase1_threads_cache is not None:
            return

        with self._lock:
            if self._restaurants_indices is not None and self._phase1_threads_cache is not None:
                return

            df = self._df
            if df is None:
                raise RuntimeError("Dataset not loaded.")

            total_reviews = int(len(df))
            if total_reviews == 0:
                self._restaurants_indices = {}
                self._phase1_threads_cache = []
                return

            ranges = self._iter_ranges(total_reviews)
            restaurants_indices: Dict[str, List[int]] = {}
            thread_traces: List[Dict[str, Any]] = []

            def process_chunk(idx: int, start: int, end: int):
                start_t = time.perf_counter()
                chunk = df.iloc[start:end]

                # Map slug -> list of global indices for this chunk
                local_map: Dict[str, List[int]] = {}
                for row in chunk.itertuples():
                    slug = getattr(row, "restaurant_slug", "") or ""
                    if not slug:
                        continue
                    # global index from DataFrame (Index is the original index)
                    global_idx = int(getattr(row, "Index"))
                    local_map.setdefault(slug, []).append(global_idx)

                elapsed = time.perf_counter() - start_t
                stats = {
                    "thread_index": idx,
                    "rows_processed": int(end - start),
                    "unique_restaurants": int(len(local_map)),
                    "duration_ms": round(elapsed * 1000.0, 2),
                }
                return stats, local_map

            max_workers = min(MAX_THREADS, len(ranges)) or 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(process_chunk, idx, start, end): idx
                    for idx, (start, end) in enumerate(ranges)
                }

                for future in as_completed(future_to_idx):
                    stats, local_map = future.result()
                    thread_traces.append(stats)

                    # Merge local_map into the global restaurants_indices
                    for slug, idx_list in local_map.items():
                        restaurants_indices.setdefault(slug, []).extend(idx_list)

            self._restaurants_indices = restaurants_indices
            self._phase1_threads_cache = thread_traces

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def get_all_restaurants_with_stats(self) -> Dict[str, Any]:
        """
        Phase 1:
        - Use multiple threads (one per ~100 rows) to scan the dataset
          and group all reviews by restaurant slug.
        - Cache this mapping in memory so that Phase 2 can re-use it.
        - Return:
            * total_reviews in the dataset
            * list of restaurants with name, slug, review_count AND
              the list of their reviews
            * per-thread traces

        Returns JSON-like dict:
        {
          "total_reviews": int,
          "restaurants": [
             {
               "slug": str,
               "name": str,
               "review_count": int,
               "reviews": [
                  {
                    "index": int,
                    "date": "YYYY-MM-DD" or null,
                    "review_text": str,
                    "yelp_url": str
                  },
                  ...
               ]
             },
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
        self._ensure_phase1_cache()
        df = self._df
        if df is None:
            raise RuntimeError("Dataset not loaded.")

        total_reviews = int(len(df))
        restaurants_indices = self._restaurants_indices or {}
        thread_traces = self._phase1_threads_cache or []

        # Map slug -> pretty display name (we already built this in _ensure_loaded)
        slug_to_name_map = (
            df.drop_duplicates("restaurant_slug")
            .set_index("restaurant_slug")["restaurant_name"]
            .to_dict()
        )

        restaurants_list: List[Dict[str, Any]] = []

        # Sort restaurants by number of reviews (descending)
        for slug, idx_list in sorted(
            restaurants_indices.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            if not slug:
                continue

            reviews_data: List[Dict[str, Any]] = []
            for idx in idx_list:
                row = df.loc[idx]
                date_val = row.get("date")
                if pd.isna(date_val):
                    date_str = None
                else:
                    # isoformat for JSON
                    date_str = (
                        date_val.isoformat()
                        if hasattr(date_val, "isoformat")
                        else str(date_val)
                    )

                reviews_data.append(
                    {
                        "index": int(idx),
                        "date": date_str,
                        "review_text": row.get("review_text"),
                        "yelp_url": row.get("yelp_url"),
                    }
                )

            restaurants_list.append(
                {
                    "slug": slug,
                    "name": slug_to_name_map.get(slug, slug_to_display_name(slug)),
                    "review_count": int(len(idx_list)),
                    "reviews": reviews_data,
                }
            )

        return {
            "total_reviews": total_reviews,
            "restaurants": restaurants_list,
            "threads": thread_traces,
        }

    # ------------------------------------------------------------------
    # Phase 2 (merged analysis for one restaurant)
    # ------------------------------------------------------------------

    def analyze_restaurant_with_stats(self, restaurant_name: str) -> Dict[str, Any]:
        """
        Phase 2 (merged):
            - Re-use the Phase 1 cache instead of rescanning the whole dataset.
            - Find all restaurant slugs whose normalized form matches `restaurant_name`.
            - Only process the reviews belonging to those slugs (subset of the dataset).
            - Use threads (chunks of 100 reviews) to:
                * compute a sentiment-based rating (0–5) for each review
                * accumulate local per-year sums & counts
            - The main thread merges per-year sums & counts to produce:
                * ratings_by_year (last 10 years)
                * global_rate
                * per-thread traces

        Returns a JSON-like dict with:
          - ratings_by_year
          - global_rate (average of yearly ratings)
          - analysis: {
                total_reviews_processed,
                total_threads,
                threads: [
                  {
                    thread_index,
                    rows_processed,
                    rows_with_year,
                    duration_ms,
                    years: { "2019": 120, ... }
                  }
                ]
            }
        """
        self._ensure_loaded()
        self._ensure_phase1_cache()

        df = self._df
        if df is None:
            raise RuntimeError("Dataset not loaded.")

        restaurants_indices = self._restaurants_indices or {}
        total_dataset_reviews = int(len(df))

        # --------------------------------------
        # Determine which slugs match this name
        # --------------------------------------
        candidate_slugs: List[str] = []
        for slug in restaurants_indices.keys():
            if restaurant_matches(slug, restaurant_name):
                candidate_slugs.append(slug)

        # Collect all indices belonging to matching slugs
        candidate_indices: List[int] = []
        for slug in candidate_slugs:
            candidate_indices.extend(restaurants_indices.get(slug, []))

        # Remove duplicates (in case of weird overlaps) and sort
        candidate_indices = sorted(set(candidate_indices))
        total_subset_reviews = len(candidate_indices)

        model = SentimentModel.instance()
        analysis_traces: List[Dict[str, Any]] = []

        # Per-year global aggregates (merged after threads)
        year_sums: Dict[int, float] = {}
        year_counts: Dict[int, int] = {}

        # If nothing matches, return empty results but still include structure
        if total_subset_reviews == 0:
            return {
                "restaurant": restaurant_name,
                "search_query": restaurant_name,
                "dataset_total_reviews": total_dataset_reviews,
                "restaurant_total_reviews": 0,
                "ratings_by_year": {},
                "global_rate": None,
                "analysis": {
                    "total_reviews_processed": 0,
                    "total_threads": 0,
                    "threads": [],
                },
            }

        # --------------------------------------
        # Phase 2: process only the subset (threads)
        # --------------------------------------
        def _iter_subset_ranges(n: int) -> List[tuple]:
            return [(i, min(i + CHUNK_SIZE, n)) for i in range(0, n, CHUNK_SIZE)]

        ranges = _iter_subset_ranges(total_subset_reviews)

        def process_chunk(idx: int, start_pos: int, end_pos: int):
            """
            Process a slice [start_pos, end_pos) of candidate_indices.
            Each review:
              - gets a sentiment-based rating
              - contributes to local per-year sums and counts
            """
            start_t = time.perf_counter()
            rows_processed = int(end_pos - start_pos)
            local_year_sums: Dict[int, float] = {}
            local_year_counts: Dict[int, int] = {}
            rows_with_year = 0

            for pos in range(start_pos, end_pos):
                df_idx = candidate_indices[pos]
                row = df.loc[df_idx]

                text = row.get("review_text", "") or ""
                score = float(model.score_review(text))

                year_val = row.get("year", None)
                if pd.notnull(year_val):
                    year_int = int(year_val)
                    local_year_sums[year_int] = local_year_sums.get(year_int, 0.0) + score
                    local_year_counts[year_int] = local_year_counts.get(year_int, 0) + 1
                    rows_with_year += 1

            elapsed = time.perf_counter() - start_t
            stats = {
                "thread_index": idx,
                "rows_processed": rows_processed,
                "rows_with_year": rows_with_year,
                "duration_ms": round(elapsed * 1000.0, 2),
                "years": {
                    str(y): local_year_counts[y] for y in sorted(local_year_counts.keys())
                },
            }
            return stats, local_year_sums, local_year_counts

        max_workers = min(MAX_THREADS, len(ranges)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_chunk, idx, start_pos, end_pos): idx
                for idx, (start_pos, end_pos) in enumerate(ranges)
            }

            for future in as_completed(future_to_idx):
                stats, local_sums, local_counts = future.result()
                analysis_traces.append(stats)

                # Merge local per-year aggregates into global ones
                for year, s in local_sums.items():
                    year_sums[year] = year_sums.get(year, 0.0) + s
                for year, c in local_counts.items():
                    year_counts[year] = year_counts.get(year, 0) + c

        # ------------------------------------------------------------------
        # Aggregate per-year stats (last 10 years) and compute global rate
        # ------------------------------------------------------------------
        now_year = datetime.utcnow().year
        min_year = now_year - 9  # last 10 years inclusive

        ratings_by_year: Dict[int, float] = {}
        for year, count in year_counts.items():
            if count <= 0:
                continue
            if year < min_year or year > now_year:
                continue
            total_score = year_sums.get(year, 0.0)
            avg = total_score / float(count)
            ratings_by_year[year] = round(avg, 2)

        # Compute global rate as an average of yearly averages
        if ratings_by_year:
            valid_ratings = list(ratings_by_year.values())
            global_rate = round(sum(valid_ratings) / len(valid_ratings), 2)
        else:
            global_rate = None

        # Try to use a nicer restaurant name if we have a matching slug
        if candidate_slugs:
            nice_name = slug_to_display_name(candidate_slugs[0])
        else:
            nice_name = restaurant_name

        ratings_str_keys = {str(y): r for y, r in sorted(ratings_by_year.items())}

        return {
            "restaurant": nice_name,
            "search_query": restaurant_name,
            "dataset_total_reviews": total_dataset_reviews,
            "restaurant_total_reviews": total_subset_reviews,
            "ratings_by_year": ratings_str_keys,
            "global_rate": global_rate,
            "analysis": {
                "total_reviews_processed": total_subset_reviews,
                "total_threads": len(ranges),
                "threads": analysis_traces,
            },
        }
