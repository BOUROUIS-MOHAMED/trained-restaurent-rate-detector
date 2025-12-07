# restaurant_utils.py
import re
from urllib.parse import urlparse, unquote


def extract_restaurant_slug(yelp_url: str) -> str:
    """
    Extract a stable restaurant 'slug' from the Yelp URL.

    Example:
        https://www.yelp.com/biz/milk-bar-las-vegas-las-vegas-3,2,2019-10-10
        -> 'milk-bar-las-vegas-las-vegas-3'

    For URLs with encoded characters, e.g.:
        /biz/am%C3%A9lies-french-bakery-and-caf%C3%A9-charlotte-11
        -> 'amélies-french-bakery-and-café-charlotte-11'

    Steps:
      * Take the path after /biz/
      * Drop everything after the first comma (rating/date artifacts)
      * URL-decode (%XX -> char)
      * Lowercase it
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

    # Dataset may append ",rating,date" after the slug
    slug = slug.split(",", 1)[0]
    slug = slug.strip("/")

    # Decode %XX sequences like %C3%A9 -> é
    slug = unquote(slug)

    # Normalize to lowercase
    slug = slug.lower()
    return slug


def normalize_restaurant_search(name: str) -> str:
    """
    Normalize a name or slug into a slug-like form without accents.

    - Lowercases
    - Replaces any non [a-z0-9] character (including é, à, etc.) by '-'
    - Collapses multiple '-' into a single one
    - Strips leading/trailing '-'

    Examples:
        "Milk Bar Las Vegas" -> "milk-bar-las-vegas"
        "Amélies French Bakery And Café Charlotte 11"
            -> "am-lies-french-bakery-and-caf-charlotte-11"
        "amélies-french-bakery-and-café-charlotte-11"
            -> "am-lies-french-bakery-and-caf-charlotte-11"
    """
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "-", name)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized


def slug_to_display_name(slug: str) -> str:
    """
    Turn a slug into a human-friendly name.

    - URL-decodes first (so %C3%A9 -> é)
    - Replaces '-' with ' '
    - Collapses spaces
    - Title-cases the result

    Example:
        "am%C3%A9lies-french-bakery-and-caf%C3%A9-charlotte-11"
          -> "Amélies French Bakery And Café Charlotte 11"
    """
    if not isinstance(slug, str):
        return ""

    # Just in case slug still contains %XX sequences
    slug = unquote(slug)

    text = slug.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title()


def restaurant_matches(slug_from_url: str, search_name: str) -> bool:
    """
    Decide if a given Yelp slug corresponds to the user search.

    We normalize BOTH:
      - slug_from_url
      - search_name (what we pass in the API 'name' parameter)

    and then compare the normalized forms by inclusion:
      - normalized_search in normalized_slug
      - OR normalized_slug in normalized_search

    This makes it robust to:
      - accents (é / è / ê etc.)
      - extra words
      - partial matches like "hubby" vs "hubby-burger-los-angeles"
    """
    if not slug_from_url or not search_name:
        return False

    # Normalize both sides
    normalized_slug = normalize_restaurant_search(slug_from_url)
    normalized_search = normalize_restaurant_search(search_name)

    if not normalized_slug or not normalized_search:
        return False

    return (normalized_search in normalized_slug) or (normalized_slug in normalized_search)
