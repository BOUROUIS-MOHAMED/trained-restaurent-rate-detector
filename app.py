# app.py
from flask import Flask, jsonify, request, render_template

from data_processing import ReviewDataset

app = Flask(__name__)

# Path to your Kaggle CSV
DATASET_PATH = "data/yelp_restaurant_reviews.csv"

# Global dataset manager
dataset = ReviewDataset(csv_path=DATASET_PATH)


@app.route("/")
def index():
    """
    UI page to test the API.
    """
    return render_template("index.html")


@app.route("/api/v1/getAllRestaurents", methods=["GET"])
def get_all_restaurents():
    """
    Phase 1 endpoint.

    GET /api/v1/getAllRestaurents

    Returns JSON:
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
    try:
        payload = dataset.get_all_restaurants_with_stats()
    except FileNotFoundError:
        return (
            jsonify(
                {
                    "error": "Dataset file not found.",
                    "details": f"Expected CSV at {DATASET_PATH}. "
                    "Download the Kaggle dataset and place it there.",
                }
            ),
            500,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "Unexpected error while computing restaurant list.",
                    "details": str(exc),
                }
            ),
            500,
        )

    return jsonify(payload)


@app.route("/api/v1/getrestaurantrate", methods=["GET"])
def get_restaurant_rate():
    """
    Phase 2 + Phase 3 endpoint.

    GET /api/v1/getrestaurantrate?name=hubby

    Returns JSON:
    {
      "restaurant": "...",
      "search_query": "...",
      "dataset_total_reviews": int,
      "restaurant_total_reviews": int,
      "ratings_by_year": { "2019": 4.1, ... },
      "global_rate": 3.95,
      "phase2": {
        "total_reviews_processed": int,
        "total_threads": int,
        "threads": [
          {
            "thread_index": int,
            "rows_processed": int,
            "matched_reviews": int,
            "duration_ms": float
          },
          ...
        ]
      },
      "phase3": {
        "total_year_threads": int,
        "threads": [
          {
            "year": int,
            "rows_processed": int,
            "duration_ms": float
          },
          ...
        ]
      }
    }
    """
    name = request.args.get("name", type=str)
    if not name:
        return jsonify({"error": 'Missing required query parameter "name".'}), 400

    try:
        result = dataset.analyze_restaurant_with_stats(name)
    except FileNotFoundError:
        return (
            jsonify(
                {
                    "error": "Dataset file not found.",
                    "details": f"Expected CSV at {DATASET_PATH}. "
                    "Download the Kaggle dataset and place it there.",
                }
            ),
            500,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "Unexpected error while computing restaurant ratings.",
                    "details": str(exc),
                }
            ),
            500,
        )

    return jsonify(result)


if __name__ == "__main__":
    # You can change host/port if you want exactly http://localhost/...
    # For example: app.run(host="0.0.0.0", port=80)  (requires admin privileges).
    app.run(debug=True)
