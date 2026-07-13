from datetime import datetime
from zoneinfo import ZoneInfo


def format_local_time(
    iso_timestamp: str | None,
    *,
    timezone: str = "Asia/Shanghai",
) -> str:
    if not iso_timestamp:
        return "(none)"
    try:
        parsed = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return iso_timestamp
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    local = parsed.astimezone(ZoneInfo(timezone))
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")
