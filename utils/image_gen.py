import base64
import io
import os
import time

from PIL import Image
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, GEMINI_IMAGE_MODEL

MAX_FILE_BYTES = 1_900_000  # 1.9 MB — abaixo do limite de 2 MB da Shopee

SLOT_LABELS = {
    "capa": "Capa",
    "detalhe": "Detalhe",
    "lifestyle": "Lifestyle",
    "objecao": "Quebra de Objeção",
}

SLOT_ZIP_NAMES = {
    "capa": "01_capa.png",
    "detalhe": "02_detalhe.png",
    "lifestyle": "03_lifestyle.png",
    "objecao": "04_objecao.png",
}


def _detect_media_type(b64_data: str) -> str:
    prefix = b64_data[:12]
    if prefix.startswith("/9j"):
        return "image/jpeg"
    if prefix.startswith("iVBOR"):
        return "image/png"
    if prefix.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _compress_to_limit(path: str) -> None:
    """Re-salva a imagem como JPEG reduzindo qualidade até ficar abaixo de MAX_FILE_BYTES."""
    if os.path.getsize(path) <= MAX_FILE_BYTES:
        return

    img = Image.open(path).convert("RGB")
    for quality in range(85, 29, -10):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= MAX_FILE_BYTES:
            with open(path, "wb") as f:
                f.write(buf.getvalue())
            return

    # último recurso: qualidade 30
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30, optimize=True)
    with open(path, "wb") as f:
        f.write(buf.getvalue())


def generate_image(prompt: str, reference_images_base64: list[str] | None, output_path: str) -> str | None:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY não configurada no .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    contents: list = []
    for ref_b64 in (reference_images_base64 or []):
        contents.append(
            types.Part.from_bytes(
                data=base64.b64decode(ref_b64),
                mime_type=_detect_media_type(ref_b64),
            )
        )
    contents.append(types.Part.from_text(text=prompt))

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=contents,
                config=config,
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)
                    _compress_to_limit(output_path)
                    return output_path
            raise RuntimeError("Gemini retornou resposta sem imagem (possível recusa por política de conteúdo)")
        except Exception as e:
            last_error = e
            if attempt < 1:
                time.sleep(3)

    raise RuntimeError(
        f"Geração de imagem falhou após 2 tentativas — {type(last_error).__name__}: {last_error}"
    ) from last_error
