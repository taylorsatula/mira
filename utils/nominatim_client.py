"""
OpenStreetMap geocoding client (Nominatim + Overpass API).

Shared HTTP layer for all OpenStreetMap-based location queries in MIRA
(maps_tool, weather_tool). No API key required.

Public-instance usage policy constraints (https://operations.osmfoundation.org/policies/nominatim/):
- Maximum 1 request/second (enforced here via a process-wide rate limiter)
- Requests must send an identifying User-Agent

Place IDs: MIRA encodes an OpenStreetMap element as "<T><osm_id>" where T is
N (node), W (way), or R (relation) — e.g. "N123456789". This is the format
accepted by the Nominatim /lookup endpoint and by maps_tool's place_details
operation.
"""

import logging
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional

from utils import http_client

logger = logging.getLogger(__name__)

NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
# Public Overpass instances. Reliability varies per instance and over time,
# so nearby's retry loop rotates through the list — but only for the two
# failure modes it handles itself: read timeouts and HTTP-200 responses
# carrying an error "remark" skip to the next instance rather than hammering
# the current one. HTTP-error responses (e.g. 5xx) do NOT rotate: http_client
# retries them in place on the same instance, and once its retries are
# exhausted the error surfaces to the caller without falling back to the
# remaining instances.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

# Nominatim usage policy requires an identifying User-Agent; generic
# library defaults get throttled or blocked.
USER_AGENT = "MIRA-Assistant/1.0 (OpenStreetMap Nominatim client)"

REQUEST_TIMEOUT = 30

# Public Nominatim policy: absolute maximum 1 request/second.
MIN_REQUEST_INTERVAL_SECONDS = 1.0
_rate_lock = threading.Lock()
_last_request_at = 0.0

_PLACE_ID_PATTERN = re.compile(r"^[NWR]\d+$")
_OSM_TYPE_LETTERS = {"node": "N", "way": "W", "relation": "R"}

# Overpass tag keys searched for typed nearby-place queries. "amenity" covers
# most POIs (restaurant, cafe, hospital); the others cover shops, attractions,
# and leisure venues.
_POI_TAG_KEYS = ("amenity", "shop", "tourism", "leisure")

# The public Overpass instances are unreliable under load: beyond HTTP errors
# (handled by http_client), they frequently accept the connection then never
# respond, and report server-side query timeouts as HTTP 200 with an error
# "remark" in the JSON body. Retry both failure modes, rotating instances.
OVERPASS_MAX_ATTEMPTS = 4


def _throttle() -> None:
    """Enforce the Nominatim 1 request/second usage-policy limit."""
    global _last_request_at
    with _rate_lock:
        wait = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _get(url: str, params: Dict[str, Any]) -> Any:
    _throttle()
    response = http_client.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def osm_place_id(osm_type: str, osm_id: Any) -> str:
    """Encode an OSM element as a MIRA place ID, e.g. ("node", 123) -> "N123"."""
    letter = _OSM_TYPE_LETTERS.get(osm_type, "")
    if not letter or osm_id is None:
        return ""
    return f"{letter}{osm_id}"


def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Forward-geocode a free-text query (address, landmark, place name).

    Returns a list of Nominatim result dicts, best match first. Empty list
    means no match was found.
    """
    return _get(
        f"{NOMINATIM_BASE_URL}/search",
        params={
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": limit,
        },
    )


def reverse(lat: float, lng: float) -> Dict[str, Any]:
    """
    Reverse-geocode coordinates to the nearest address.

    Raises:
        ValueError: If no address exists at the given coordinates.
    """
    result = _get(
        f"{NOMINATIM_BASE_URL}/reverse",
        params={"lat": lat, "lon": lng, "format": "jsonv2", "addressdetails": 1},
    )
    if "error" in result:
        raise ValueError(f"No address found at ({lat}, {lng}): {result['error']}")
    return result


def lookup(place_id: str) -> Dict[str, Any]:
    """
    Fetch full details for a MIRA place ID (e.g. "N123456789") including
    extra tags (website, phone, opening_hours) when OpenStreetMap has them.

    Raises:
        ValueError: If the place ID is malformed or unknown.
    """
    if not _PLACE_ID_PATTERN.match(place_id):
        raise ValueError(
            f"Invalid place ID '{place_id}'. Expected OSM format: 'N', 'W', or 'R' "
            "followed by the OSM element ID (e.g. 'N123456789'), as returned by "
            "geocode/find_place results."
        )
    results = _get(
        f"{NOMINATIM_BASE_URL}/lookup",
        params={
            "osm_ids": place_id,
            "format": "jsonv2",
            "addressdetails": 1,
            "extratags": 1,
        },
    )
    if not results:
        raise ValueError(f"No details found for place ID: {place_id}")
    return results[0]


def _escape_overpass_string(value: str) -> str:
    """Escape a value for interpolation into a double-quoted Overpass QL string."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def nearby(
    lat: float,
    lng: float,
    radius: int,
    place_type: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Find points of interest within `radius` meters of a coordinate via the
    Overpass API.

    `place_type` matches POI tag values (e.g. "restaurant", "cafe", "hospital")
    across amenity/shop/tourism/leisure keys. `keyword` matches place names
    case-insensitively. With neither filter, all tagged POIs are returned.
    """
    around = f"(around:{radius},{lat},{lng})"
    clauses: List[str] = []

    if place_type:
        escaped = _escape_overpass_string(place_type)
        for key in _POI_TAG_KEYS:
            clauses.append(f'nwr["{key}"="{escaped}"]{around};')
    elif keyword:
        escaped = _escape_overpass_string(keyword)
        clauses.append(f'nwr["name"~"{escaped}",i]{around};')
    else:
        for key in _POI_TAG_KEYS:
            clauses.append(f'nwr["{key}"]{around};')

    query = f"[out:json][timeout:25];({''.join(clauses)});out center {limit};"

    for attempt in range(OVERPASS_MAX_ATTEMPTS):
        instance = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        _throttle()
        try:
            response = http_client.post(
                instance,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except http_client.TimeoutException as e:
            if attempt == OVERPASS_MAX_ATTEMPTS - 1:
                raise ValueError(
                    f"Overpass query timed out after {OVERPASS_MAX_ATTEMPTS} attempts: {e}"
                )
            delay = 2.0 * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                f"Overpass read timeout on {instance}, attempt {attempt + 1}/{OVERPASS_MAX_ATTEMPTS}, "
                f"retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            continue

        remark = payload.get("remark")
        if remark:
            if attempt == OVERPASS_MAX_ATTEMPTS - 1:
                raise ValueError(
                    f"Overpass query failed after {OVERPASS_MAX_ATTEMPTS} attempts: {remark}"
                )
            delay = 2.0 * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                f"Overpass runtime error on {instance} ({remark}), "
                f"attempt {attempt + 1}/{OVERPASS_MAX_ATTEMPTS}, retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            continue

        return payload.get("elements", [])
