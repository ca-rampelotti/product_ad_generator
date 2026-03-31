import json
import os
import time
from dataclasses import dataclass

import httpx

ML_TOKEN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".ml_tokens.json"
)
ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"
ML_AUTH_URL = "https://auth.mercadolivre.com.br/authorization"


@dataclass
class MLTokens:
    access_token: str
    refresh_token: str
    expires_at: float  # unix timestamp


def build_auth_url(app_id: str, redirect_uri: str) -> str:
    from urllib.parse import urlencode
    params = urlencode({
        "response_type": "code",
        "client_id": app_id,
        "redirect_uri": redirect_uri,
    })
    return f"{ML_AUTH_URL}?{params}"


def load_tokens() -> MLTokens | None:
    # 1. Arquivo local (desenvolvimento)
    if os.path.exists(ML_TOKEN_FILE):
        try:
            with open(ML_TOKEN_FILE) as f:
                data = json.load(f)
            return MLTokens(**data)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # 2. Streamlit secrets (produção — token chumbado manualmente)
    try:
        import streamlit as st
        token = st.secrets.get("ML_ACCESS_TOKEN", "")
        if token:
            return MLTokens(
                access_token=token,
                refresh_token=st.secrets.get("ML_REFRESH_TOKEN", ""),
                expires_at=float(st.secrets.get("ML_TOKEN_EXPIRES_AT", 9999999999)),
            )
    except Exception:
        pass

    return None


def save_tokens(tokens: MLTokens) -> None:
    with open(ML_TOKEN_FILE, "w") as f:
        json.dump(
            {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "expires_at": tokens.expires_at,
            },
            f,
        )


def revoke_tokens() -> None:
    if os.path.exists(ML_TOKEN_FILE):
        os.remove(ML_TOKEN_FILE)


def authorize_with_code(
    code: str, app_id: str, secret_key: str, redirect_uri: str
) -> MLTokens:
    """Exchange authorization code for tokens and persist them."""
    tokens = _exchange_code(code, app_id, secret_key, redirect_uri)
    save_tokens(tokens)
    return tokens


def get_valid_token(app_id: str, secret_key: str) -> str | None:
    """Returns a valid access_token, refreshing automatically if expired.

    Returns None if no tokens are saved (user not connected).
    """
    tokens = load_tokens()
    if tokens is None:
        return None

    if time.time() >= tokens.expires_at:
        if not tokens.refresh_token:
            # Sem refresh_token (offline_access não ativado) — token expirou, precisa reconectar
            return None
        tokens = _refresh_tokens(tokens, app_id, secret_key)
        save_tokens(tokens)

    return tokens.access_token


def _exchange_code(
    code: str, app_id: str, secret_key: str, redirect_uri: str
) -> MLTokens:
    response = httpx.post(
        ML_TOKEN_URL,
        headers={
            "accept": "application/json",
            "content-type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "client_id": app_id,
            "client_secret": secret_key,
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if not response.is_success:
        raise RuntimeError(f"ML OAuth error {response.status_code}: {response.text}")
    data = response.json()
    if "refresh_token" not in data:
        print("AVISO: refresh_token não retornado. Ative o escopo 'offline_access' no painel do app ML.")
    return MLTokens(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        expires_at=time.time() + data["expires_in"] - 300,  # 5min safety buffer
    )


def _refresh_tokens(tokens: MLTokens, app_id: str, secret_key: str) -> MLTokens:
    response = httpx.post(
        ML_TOKEN_URL,
        headers={
            "accept": "application/json",
            "content-type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "client_id": app_id,
            "client_secret": secret_key,
            "refresh_token": tokens.refresh_token,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return MLTokens(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=time.time() + data["expires_in"] - 300,
    )
