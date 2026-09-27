"""Location context trinket — injects user's location, sunrise/sunset, and 2-hour forecast into system prompt."""
import html
import json
import logging
from typing import Dict, Any

from clients.valkey_client import get_valkey_client
from utils.user_context import get_current_user_id
from .base import EventAwareTrinket

logger = logging.getLogger(__name__)


class LocationTrinket(EventAwareTrinket):
    """
    Reads cached location data from Valkey and renders it into the system prompt.
    Data is populated by the POST /location endpoint on frontend page load.
    """

    variable_name = "location_context"
    cache_policy = True  # Location is stable within a session — don't bust prompt cache

    def generate_content(self, context: Dict[str, Any]) -> str:
        user_id = get_current_user_id()
        cache_key = f"location:{user_id}"

        valkey = get_valkey_client()
        raw = valkey.get(cache_key)

        if not raw:
            return ""

        data = json.loads(raw)
        location_name = data.get("location_name")
        forecast = data.get("forecast")
        sunrise = data.get("sunrise")
        sunset = data.get("sunset")

        if not location_name and not forecast and not sunrise:
            return ""

        lines = ["=== Location Context ==="]

        if location_name:
            lines.append(f"Location: {html.escape(location_name)}")

        if sunrise and sunset:
            lines.append(f"Sunrise: {sunrise} / Sunset: {sunset}")

        uv_index = data.get("uv_index")
        if uv_index is not None:
            if uv_index >= 11:
                lines.append(f"⚠️ UV INDEX: {uv_index} (EXTREME) — Dangerous sun exposure levels")
            else:
                lines.append(f"UV Index: {uv_index} (Very High)")

        if forecast:
            lines.append("Forecast (next 2 hours):")
            for entry in forecast:
                lines.append(
                    f"- {entry['hour']}: {entry['temp']}, {html.escape(entry['condition'])}, "
                    f"{entry['precip_pct']}% chance of rain"
                )

        # Recent daily weather since last conversation (up to 3 days)
        recent_weather = data.get("recent_weather")
        if recent_weather:
            lines.append("Recent weather (since last conversation):")
            for day in recent_weather:
                parts = []
                if "high" in day and "low" in day:
                    parts.append(f"{day['high']}/{day['low']}")
                if "condition" in day:
                    parts.append(html.escape(day["condition"]))
                if "precipitation" in day:
                    parts.append(f"{day['precipitation']} precip")
                lines.append(f"- {html.escape(day['day_label'])}: {', '.join(parts)}")

        return "\n".join(lines)
