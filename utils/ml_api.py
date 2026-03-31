import time
from pathlib import Path

import httpx

ML_BASE_URL = "https://api.mercadolibre.com"
MAX_RETRIES = 3


def auth_headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def ml_get(path: str, access_token: str) -> dict | list:
    url = f"{ML_BASE_URL}{path}"
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.get(url, headers=auth_headers(access_token), timeout=30)
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code < 500:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"GET {path} falhou após {MAX_RETRIES} tentativas") from last_error


def ml_post(path: str, access_token: str, json_body: dict | None = None, files: dict | None = None) -> dict:
    url = f"{ML_BASE_URL}{path}"
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            if files:
                # multipart — httpx sets Content-Type automatically
                headers = {"Authorization": f"Bearer {access_token}"}
                response = httpx.post(url, headers=headers, files=files, timeout=60)
            else:
                response = httpx.post(
                    url, headers=auth_headers(access_token), json=json_body, timeout=30
                )

            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 204:
                return {}
            if not response.is_success:
                raise RuntimeError(f"ML API error {response.status_code}: {response.text}")
            return response.json()
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code < 500:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"POST {path} falhou após {MAX_RETRIES} tentativas") from last_error


def ml_validate(path: str, access_token: str, json_body: dict) -> list[dict]:
    """Validates an item payload. Returns list of error causes (empty = valid)."""
    url = f"{ML_BASE_URL}{path}"
    response = httpx.post(
        url, headers=auth_headers(access_token), json=json_body, timeout=30
    )

    if response.status_code == 204:
        return []

    if response.status_code == 400:
        data = response.json()
        return [c for c in data.get("cause", []) if c.get("type") == "error"]

    response.raise_for_status()
    return []


def ml_post_description(item_id: str, access_token: str, plain_text: str) -> None:
    """POST description. Falls back to PUT if description already exists (409)."""
    url = f"{ML_BASE_URL}/items/{item_id}/description"
    headers = auth_headers(access_token)
    body = {"plain_text": plain_text}

    response = httpx.post(url, headers=headers, json=body, timeout=30)

    if response.status_code == 409:
        response = httpx.put(
            f"{url}?api_version=2", headers=headers, json=body, timeout=30
        )

    if response.status_code not in (200, 201, 204):
        response.raise_for_status()
