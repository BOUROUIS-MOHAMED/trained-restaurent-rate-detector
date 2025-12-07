# restaurant_utils.py
import re
from urllib.parse import urlparse


def extract_restaurant_slug(yelp_url: str) -> str:
    """
    Extract a stable restaurant 'slug' from the Yelp URL.

    Example:
        https://www.yelp.com/biz/milk-bar-las-vegas-las-vegas-3,2,2019-10-10
        -> 'milk-bar-las-vegas-las-vegas-3'

    We:
      * take the path after /biz/
      * drop everything after the first comma (rating/date artifacts)
      * lowercase it
    """
    if not isinstance(yelp_url, str):
        return ""

    parsed = urlparse(yelp_url)
    path = parsed.path or ""

    # /biz/<slug>
    if "/biz/" in path:
        slug = path.split("/biz/", 1)[1]
    else:
        slug = path.lstrip("/")

    # dataset may append ",rating,date" after the slug
    slug = slug.split(",", 1)[0]
    slug = slug.strip("/")
    return slug.lower()


def normalize_restaurant_search(name: str) -> str:
    """
    Normalize user-input restaurant name into a slug-like form.

    "Milk Bar Las Vegas" -> "milk-bar-las-vegas"
    """
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()
    # Replace non-alphanumeric with '-'
    normalized = re.sub(r"[^a-z0-9]+", "-", name)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized


def slug_to_display_name(slug: str) -> str:
    """
    Turn a slug into a human-friendly name.

    "milk-bar-las-vegas-las-vegas-3" -> "Milk Bar Las Vegas Las Vegas 3"
    """
    if not isinstance(slug, str):
        return ""

    text = slug.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title()


def restaurant_matches(slug_from_url: str, search_name: str) -> bool:
    """
    Decide if a given Yelp slug corresponds to the user search.

    We compare both ways:
      - normalized_search in slug
      - slug in normalized_search
    so partial matches like "hubby" can still work.
    """
    slug_from_url = (slug_from_url or "").lower()
    normalized_search = normalize_restaurant_search(search_name)

    if not slug_from_url or not normalized_search:
        return False

    return (normalized_search in slug_from_url) or (slug_from_url in normalized_search)
