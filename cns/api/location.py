"""Location context endpoint — receives coordinates, fetches geocode + forecast, caches in Valkey."""
import json
import logging
from datetime import datetime, date

import pytz

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth.api import get_current_user
from auth.types import SessionData, APITokenContext
from clients.valkey_client import get_valkey_client
from utils import http_client
from utils.database_session_manager import get_shared_session_manager
from utils.timezone_utils import utc_now, strftime_no_pad
from utils.user_context import set_current_user_id, get_user_preferences

from .base import PropagatingHandler

logger = logging.getLogger(__name__)

router = APIRouter()

NOMINATIM_BASE = "https://nominatim.openstreetmap.org/reverse"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_USER_AGENT = "MIRA/1.0 (location-context)"
LOCATION_CACHE_TTL_SECONDS = 86400  # 24 hours

# Simplified WMO weather code mapping (WMO 4677 subset)
WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with light hail",
    99: "Thunderstorm with heavy hail",
}


class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")


def _reverse_geocode(lat: float, lon: float) -> str | None:
    """Reverse geocode coordinates to a human-readable location name via Nominatim."""
    try:
        response = http_client.get(
            NOMINATIM_BASE,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 10},
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {})

        city = address.get("city") or address.get("town") or address.get("village") or address.get("municipality", "")
        state = address.get("state", "")
        country = address.get("country", "")

        parts = [p for p in [city, state, country] if p]
        return ", ".join(parts) if parts else None
    except Exception:
        logger.warning("Nominatim reverse geocode failed", exc_info=True)
        return None


def _get_days_since_last_conversation(user_id: str, timezone: str) -> int:
    """Return number of gap days between last conversation and today (0-3).

    Gap days are the days *between* last_activity_date and today, exclusive of both.
    Today's weather is already covered by the 2-hour forecast; we want the in-between days.
    Returns 0 if user was active yesterday or today.
    """
    session_manager = get_shared_session_manager()
    with session_manager.get_session(user_id) as session:
        result = session.execute_single(
            "SELECT last_activity_date FROM users WHERE id = %(user_id)s",
            {'user_id': user_id},
        )

    if not result or not result.get('last_activity_date'):
        return 0

    last_date: date = result['last_activity_date']
    user_tz = pytz.timezone(timezone)
    today = utc_now().astimezone(user_tz).date()

    # Days between last conversation and today, not counting either endpoint
    # last spoke Friday, today Monday → Sat, Sun = 2 gap days
    gap = (today - last_date).days - 1
    return max(0, min(gap, 3))


def _fetch_forecast(
    lat: float, lon: float, temperature_unit: str, timezone: str, past_days: int = 0,
) -> dict | None:
    """Fetch 2-hour forecast + sunrise/sunset from Open-Meteo.

    When past_days > 0, also fetches daily weather summaries for recent days
    to provide environmental continuity between conversations.
    """
    try:
        daily_vars = "sunrise,sunset,uv_index_max"
        if past_days > 0:
            daily_vars += ",temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum"

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "daily": daily_vars,
            "forecast_hours": 3,
            "forecast_days": 1,
            "temperature_unit": temperature_unit,
            "timezone": timezone,
        }
        if past_days > 0:
            params["past_days"] = past_days

        response = http_client.get(OPEN_METEO_BASE, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        result: dict = {}
        unit_symbol = "°F" if temperature_unit == "fahrenheit" else "°C"

        # Parse hourly forecast
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation_probability", [])
        codes = hourly.get("weather_code", [])

        if len(times) >= 2:
            # Discard first entry (current hour), use next 2
            forecast = []
            for i in range(1, min(3, len(times))):
                hour_dt = datetime.fromisoformat(times[i])
                hour_label = strftime_no_pad(hour_dt, "%-I %p")
                condition = WMO_DESCRIPTIONS.get(codes[i], f"Code {codes[i]}")
                forecast.append({
                    "hour": hour_label,
                    "temp": f"{round(temps[i])}{unit_symbol}",
                    "condition": condition,
                    "precip_pct": precip[i] if i < len(precip) else 0,
                })
            result["forecast"] = forecast

        # Parse daily data — today is the last entry when past_days > 0
        daily = data.get("daily", {})
        daily_dates = daily.get("time", [])
        sunrises = daily.get("sunrise", [])
        sunsets = daily.get("sunset", [])

        # Sunrise/sunset from today (last entry)
        if sunrises and sunsets:
            sunrise_dt = datetime.fromisoformat(sunrises[-1])
            sunset_dt = datetime.fromisoformat(sunsets[-1])
            result["sunrise"] = strftime_no_pad(sunrise_dt, "%-I:%M %p")
            result["sunset"] = strftime_no_pad(sunset_dt, "%-I:%M %p")

        # UV from today (last entry)
        uv_values = daily.get("uv_index_max", [])
        if uv_values and uv_values[-1] is not None and uv_values[-1] >= 8:
            result["uv_index"] = round(uv_values[-1])

        # Recent daily weather history (past days only, exclude today)
        if past_days > 0:
            highs = daily.get("temperature_2m_max", [])
            lows = daily.get("temperature_2m_min", [])
            daily_codes = daily.get("weather_code", [])
            daily_precip = daily.get("precipitation_sum", [])

            recent_weather = []
            # All entries except the last (today) are past days
            for i in range(len(daily_dates) - 1):
                day_dt = date.fromisoformat(daily_dates[i])
                entry: dict = {
                    "date": daily_dates[i],
                    "day_label": strftime_no_pad(day_dt, "%a %b %-d"),
                }
                if i < len(highs) and highs[i] is not None:
                    entry["high"] = f"{round(highs[i])}{unit_symbol}"
                if i < len(lows) and lows[i] is not None:
                    entry["low"] = f"{round(lows[i])}{unit_symbol}"
                if i < len(daily_codes) and daily_codes[i] is not None:
                    entry["condition"] = WMO_DESCRIPTIONS.get(daily_codes[i], f"Code {daily_codes[i]}")
                if i < len(daily_precip) and daily_precip[i] is not None and daily_precip[i] > 0:
                    precip_unit = "in" if temperature_unit == "fahrenheit" else "mm"
                    entry["precipitation"] = f"{daily_precip[i]}{precip_unit}"

                recent_weather.append(entry)

            if recent_weather:
                result["recent_weather"] = recent_weather

        return result if result else None
    except Exception:
        logger.warning("Open-Meteo forecast fetch failed", exc_info=True)
        return None


class LocationHandler(PropagatingHandler):
    def process_request(self, *, user_id: str, latitude: float, longitude: float) -> dict:
        set_current_user_id(user_id)

        prefs = get_user_preferences()
        temperature_unit = prefs.temperature_unit

        past_days = _get_days_since_last_conversation(user_id, prefs.timezone)

        location_name = _reverse_geocode(latitude, longitude)
        weather = _fetch_forecast(latitude, longitude, temperature_unit, prefs.timezone, past_days)

        # Only cache if we got at least some data
        if location_name or weather:
            cache_data = {}
            if location_name:
                cache_data["location_name"] = location_name
            if weather:
                cache_data.update(weather)

            # Valkey write — required infrastructure, failure propagates
            valkey = get_valkey_client()
            cache_key = f"location:{user_id}"
            valkey.set(cache_key, json.dumps(cache_data), ex=LOCATION_CACHE_TTL_SECONDS)

            return {"cached": True, "location_name": location_name}

        return {"cached": False}


@router.post("/location")
def location_endpoint(
    request: LocationRequest,
    current_user: SessionData | APITokenContext = Depends(get_current_user),
):
    """Deliberately sync (not async def) so Starlette runs it in a threadpool
    instead of blocking the event loop during the blocking geocode/forecast
    HTTP round-trips and Postgres/Valkey access."""
    handler = LocationHandler()
    response = handler.handle_request(
        user_id=current_user.user_id,
        latitude=request.latitude,
        longitude=request.longitude,
    )
    return response.to_dict()
