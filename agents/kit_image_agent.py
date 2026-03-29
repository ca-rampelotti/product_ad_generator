import re
from agents.base import BaseAgent
from utils.llm import call_llm
from config import MAX_TOKENS

SYSTEM_PROMPT = """You are an expert at creating marketing photo prompts for fashion kit/combo products.
A kit consists of two garments in complementary colors.

Rules for every prompt:
- Write in English
- Never include text, lettering, watermarks, or logos in the image
- Always end with: product photography, e-commerce, high quality, 1:1 square format, 1200x1200
- Describe garments accurately based on the reference images (cut, style, fabric texture)
- Specify the exact colors from the input
- Specify the model gender as given in the input

Prompt purposes:
- capa_kit: hero shot with two models confidently wearing both pieces together, complementary styling, dramatic studio lighting with clean gradient backdrop, eye-catching composition
- detalhes_a: extreme close-up of ONLY the color_a piece — fabric texture, stitching quality, material finish details, neutral white or light grey background, no model, no other garments
- detalhes_b: extreme close-up of ONLY the color_b piece — fabric texture, stitching quality, material finish details, neutral white or light grey background, no model, no other garments
- objecao: model in motion wearing one of the pieces (choose whichever looks better), showing fit, comfort and flexibility, dynamic studio lighting — this image will have text overlay added later
- lifestyle_kit: editorial lookbook photo with both pieces styled together, natural golden-hour lighting, aspirational real-world context, NO text or logos

Rules for objecao_textos bullets:
- Write in Portuguese (pt-BR)
- 4 bullets highlighting key kit/garment benefits
- Each bullet max 30 characters
- Short, punchy, no verbs — noun phrases only (e.g. "Duas cores, um estilo")

Respond ONLY with the XML block below, no other text:
<capa_kit>prompt here</capa_kit>
<detalhes_a>prompt here</detalhes_a>
<detalhes_b>prompt here</detalhes_b>
<objecao>prompt here</objecao>
<lifestyle_kit>prompt here</lifestyle_kit>
<objecao_textos>
  <bullet1>benefit text</bullet1>
  <bullet2>benefit text</bullet2>
  <bullet3>benefit text</bullet3>
  <bullet4>benefit text</bullet4>
</objecao_textos>"""

GENDER_MAP = {"Feminino": "female", "Masculino": "male", "Unissex": "diverse"}


def _detect_media_type(b64_data: str) -> str:
    prefix = b64_data[:12]
    if prefix.startswith("/9j"):
        return "image/jpeg"
    if prefix.startswith("iVBOR"):
        return "image/png"
    if prefix.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


class KitImageAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="kit_image_agent", model="gemini-2.5-flash", temperature=0)
        self._images: list[str] = []

    def run(self, input_data: dict) -> dict:
        self._images = input_data.get("images_a", [])[:2] + input_data.get("images_b", [])[:2]
        return super().run(input_data)

    def _build_prompt(self, input_data: dict) -> str:
        gender_en = GENDER_MAP.get(input_data.get("genero", "Feminino"), "female")
        color_a = input_data.get("color_a", "")
        color_b = input_data.get("color_b", "")
        n_a = len(input_data.get("images_a", [])[:2])
        n_b = len(input_data.get("images_b", [])[:2])
        return f"""Reference images: first {n_a} image(s) show the {color_a} piece (color_a), last {n_b} image(s) show the {color_b} piece (color_b).
Generate 5 distinct professional marketing photo prompts for this kit plus 4 benefit bullets.

Kit details:
- Type: {input_data.get("tipo_peca", "")} in {color_a} AND {input_data.get("tipo_peca", "")} in {color_b}
- Material: {input_data.get("material", "")}
- Fabric design: {input_data.get("desenho_tecido", "")}
- Model gender: {gender_en}

For detalhes_a: use ONLY the {color_a} piece as reference — describe its fabric and details precisely.
For detalhes_b: use ONLY the {color_b} piece as reference — describe its fabric and details precisely.
For objecao: choose one piece to feature in a dynamic motion shot."""

    def _call_llm(self, prompt: str) -> str:
        content: list[dict] = []
        for b64 in self._images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(b64),
                    "data": b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        return call_llm(
            messages=[{"role": "user", "content": content}],
            model=self.model,
            temperature=self.temperature,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
        )

    def _parse_output(self, raw: str) -> dict:
        slots = ["capa_kit", "detalhes_a", "detalhes_b", "objecao", "lifestyle_kit"]
        parsed = {slot: _extract_tag(raw, slot) for slot in slots}

        if not all(parsed.values()):
            return {"raw": raw}

        for i in range(1, 5):
            parsed[f"bullet{i}"] = _extract_tag(raw, f"bullet{i}") or ""

        return parsed
