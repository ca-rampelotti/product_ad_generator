import re

from config import MAX_TOKENS
from utils.llm import call_llm

_EXTRACTION_PROMPT = """Extract all data from this size/measurements table.
Return ONLY in this XML format, keeping exact values and units from the image:
<tabela>
  <colunas>Column1,Column2,Column3</colunas>
  <linha>value1,value2,value3</linha>
  <linha>value1,value2,value3</linha>
</tabela>"""


def _detect_media_type(b64_data: str) -> str:
    prefix = b64_data[:12]
    if prefix.startswith("/9j"):
        return "image/jpeg"
    if prefix.startswith("iVBOR"):
        return "image/png"
    if prefix.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def extract_size_table(image_base64: str) -> dict | None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _detect_media_type(image_base64),
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": _EXTRACTION_PROMPT},
            ],
        }
    ]

    try:
        raw = call_llm(messages=messages, model="", temperature=0, max_tokens=MAX_TOKENS)
    except Exception as e:
        print(f"[size_table_extractor] LLM error: {e}")
        return None

    return _parse_table_xml(raw)


def _parse_table_xml(raw: str) -> dict | None:
    try:
        colunas_match = re.search(r"<colunas>(.*?)</colunas>", raw, re.DOTALL)
        linhas_matches = re.findall(r"<linha>(.*?)</linha>", raw, re.DOTALL)

        if not colunas_match or not linhas_matches:
            return None

        colunas = [c.strip() for c in colunas_match.group(1).split(",")]
        linhas = [[v.strip() for v in linha.split(",")] for linha in linhas_matches]
        return {"colunas": colunas, "linhas": linhas}
    except Exception:
        return None
