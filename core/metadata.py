"""EXIF metadata extraction for Smart Photo Archive."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    text = str(value).strip().strip("\x00")
    return text or None


def _format_date(value: Any) -> str | None:
    text = _clean_text(value)

    if not text:
        return None

    try:
        return datetime.strptime(
            text,
            "%Y:%m:%d %H:%M:%S",
        ).isoformat()
    except ValueError:
        return text


def _format_shutter_speed(value: Any) -> str | None:
    if value is None:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    if seconds <= 0:
        return None

    if seconds >= 1:
        return f"{seconds:g}"

    denominator = round(1 / seconds)
    return f"1/{denominator}"


def _convert_gps_coordinate(
    coordinates: Any,
    reference: Any,
) -> float | None:
    if not coordinates or len(coordinates) != 3:
        return None

    try:
        degrees = float(coordinates[0])
        minutes = float(coordinates[1])
        seconds = float(coordinates[2])

        result = (
            degrees
            + minutes / 60.0
            + seconds / 3600.0
        )

        ref = _clean_text(reference)

        if ref in {"S", "W"}:
            result *= -1

        return result

    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _extract_gps(exif: Any) -> dict[str, Any] | None:
    try:
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPS)
    except (KeyError, TypeError, AttributeError):
        return None

    if not gps_ifd:
        return None

    readable = {
        ExifTags.GPSTAGS.get(tag_id, tag_id): value
        for tag_id, value in gps_ifd.items()
    }

    latitude = _convert_gps_coordinate(
        readable.get("GPSLatitude"),
        readable.get("GPSLatitudeRef"),
    )

    longitude = _convert_gps_coordinate(
        readable.get("GPSLongitude"),
        readable.get("GPSLongitudeRef"),
    )

    altitude = _safe_float(
        readable.get("GPSAltitude")
    )

    if (
        latitude is None
        and longitude is None
        and altitude is None
    ):
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
    }


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value

    return None


def extract_metadata(
    file_path: str | Path,
) -> dict[str, Any]:
    path = Path(file_path).expanduser().resolve()

    metadata: dict[str, Any] = {
        "camera_make": None,
        "camera_model": None,
        "lens_model": None,
        "iso": None,
        "aperture": None,
        "shutter_speed": None,
        "focal_length": None,
        "date_taken": None,
        "orientation": None,
        "gps": None,
        "width": None,
        "height": None,
        "image_format": None,
        "color_mode": None,
    }

    try:
        with Image.open(path) as image:
            metadata["width"] = image.width
            metadata["height"] = image.height
            metadata["image_format"] = image.format
            metadata["color_mode"] = image.mode

            exif = image.getexif()

            if not exif:
                return metadata

            top_exif = {
                ExifTags.TAGS.get(tag_id, tag_id): value
                for tag_id, value in exif.items()
            }

            nested_exif: dict[str, Any] = {}

            try:
                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)

                nested_exif = {
                    ExifTags.TAGS.get(tag_id, tag_id): value
                    for tag_id, value in exif_ifd.items()
                }

            except (KeyError, TypeError, AttributeError):
                pass

            metadata["camera_make"] = _clean_text(
                _first_value(
                    nested_exif.get("Make"),
                    top_exif.get("Make"),
                )
            )

            metadata["camera_model"] = _clean_text(
                _first_value(
                    nested_exif.get("Model"),
                    top_exif.get("Model"),
                )
            )

            metadata["lens_model"] = _clean_text(
                _first_value(
                    nested_exif.get("LensModel"),
                    top_exif.get("LensModel"),
                )
            )

            metadata["iso"] = _safe_int(
                _first_value(
                    nested_exif.get("PhotographicSensitivity"),
                    nested_exif.get("ISOSpeedRatings"),
                    top_exif.get("PhotographicSensitivity"),
                    top_exif.get("ISOSpeedRatings"),
                )
            )

            metadata["aperture"] = _safe_float(
                _first_value(
                    nested_exif.get("FNumber"),
                    top_exif.get("FNumber"),
                )
            )

            metadata["shutter_speed"] = _format_shutter_speed(
                _first_value(
                    nested_exif.get("ExposureTime"),
                    top_exif.get("ExposureTime"),
                )
            )

            metadata["focal_length"] = _safe_float(
                _first_value(
                    nested_exif.get("FocalLength"),
                    top_exif.get("FocalLength"),
                )
            )

            metadata["date_taken"] = _format_date(
                _first_value(
                    nested_exif.get("DateTimeOriginal"),
                    nested_exif.get("DateTimeDigitized"),
                    top_exif.get("DateTimeOriginal"),
                    top_exif.get("DateTimeDigitized"),
                    top_exif.get("DateTime"),
                )
            )

            metadata["orientation"] = _safe_int(
                _first_value(
                    top_exif.get("Orientation"),
                    nested_exif.get("Orientation"),
                )
            )

            metadata["gps"] = _extract_gps(exif)

    except (
        FileNotFoundError,
        PermissionError,
        OSError,
        ValueError,
        TypeError,
    ):
        return metadata

    return metadata


def main() -> None:
    print("=" * 60)
    print("Smart Photo Archive - EXIF Metadata Test")
    print("=" * 60)

    file_path = input(
        "Enter image path: "
    ).strip().strip('"')

    metadata = extract_metadata(file_path)

    print()

    for key, value in metadata.items():
        print(f"{key:16}: {value}")


if __name__ == "__main__":
    main()
