import re
from agents.base import BaseAgent
from utils.llm import call_llm
from config import MAX_TOKENS

SYSTEM_PROMPT_TEMPLATE = """You are an expert at creating professional product photography prompts for AI image generation.
You analyze garment images and generate detailed, optimized prompts for marketing photos for e-commerce listings on Mercado Livre.

Rules for every image prompt:
- Write in English
- Never include text, lettering, watermarks, or logos in the image
- Always end with: product photography, e-commerce, high quality, 1:1 square format, 1200x1200
- Describe the garment accurately based on the reference images (cut, style, fabric texture)
- NEVER name or describe the garment color in the prompt — the reference images define the exact color and shade. Do not translate or interpret the color name. Let the model reproduce the exact color from the images.
- Specify lighting, background, and composition clearly
- Specify the model gender as given in the input

Prompt-specific rules:
- capa: {background_rule}, model wearing the garment, {pose_static}, fashion magazine energy
- detalhe: extreme close-up of fabric texture, stitching and finish details, neutral light grey background
- lifestyle: model using the garment in {scenario_desc}, natural lighting, {pose_static}
- objecao: model {pose_motion} showing fit, comfort and flexibility of the garment, dynamic studio lighting

Rules for objecao_textos bullets:
- Write in Portuguese (pt-BR)
- 4 bullets highlighting key garment benefits
- Each bullet max 30 characters
- Short, punchy, no verbs — noun phrases only (e.g. "Secagem ultra-rápida", "Toque macio")

Respond ONLY with the XML block below, no other text:
<prompts>
  <capa>prompt here</capa>
  <detalhe>prompt here</detalhe>
  <lifestyle>prompt here</lifestyle>
  <objecao>prompt here</objecao>
</prompts>
<objecao_textos>
  <bullet1>benefit text</bullet1>
  <bullet2>benefit text</bullet2>
  <bullet3>benefit text</bullet3>
  <bullet4>benefit text</bullet4>
</objecao_textos>"""

GENDER_MAP = {"Feminino": "female", "Masculino": "male", "Unissex": "diverse"}

BG_MAP = {
    "Branco puro": "MANDATORY pure white background (#FFFFFF), clean studio lighting",
    "Cinza claro": "neutral light grey background, soft studio lighting",
    "Gradiente suave": "subtle gradient background from white to light grey, professional studio lighting",
}

SCENARIO_MAP = {
    "Urbano": "an urban city street setting",
    "Praia": "a tropical beach setting",
    "Academia": "a modern gym or fitness studio",
    "Casa/Cozy": "a cozy home interior",
    "Natureza": "a lush natural outdoor setting",
    "Estúdio": "a clean professional studio with creative lighting",
}

POSE_MAP = {
    "Editorial/Fashion": ("striking editorial pose", "in dynamic editorial motion"),
    "Natural/Casual": ("relaxed natural pose", "in casual natural movement"),
    "Esportivo/Movimento": ("athletic confident pose", "in energetic athletic motion"),
}


def _build_system_prompt(input_data: dict) -> str:
    bg = BG_MAP.get(input_data.get("img_fundo", ""), BG_MAP["Branco puro"])
    scenario = SCENARIO_MAP.get(input_data.get("img_cenario", ""), "a real-world aspirational context")
    pose_static, pose_motion = POSE_MAP.get(input_data.get("img_pose", ""), POSE_MAP["Editorial/Fashion"])
    return SYSTEM_PROMPT_TEMPLATE.format(
        background_rule=bg,
        scenario_desc=scenario,
        pose_static=pose_static,
        pose_motion=pose_motion,
    )


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _detect_media_type(b64_data: str) -> str:
    prefix = b64_data[:12]
    if prefix.startswith("/9j"):
        return "image/jpeg"
    if prefix.startswith("iVBOR"):
        return "image/png"
    if prefix.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


class ImageGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="image_generator", model="gemini-2.5-flash", temperature=0)
        self._images: list[str] = []
        self._input_data: dict = {}

    def run(self, input_data: dict) -> dict:
        self._images = input_data.get("imagens", [])
        self._input_data = input_data
        return super().run(input_data)

    def _build_prompt(self, input_data: dict) -> str:
        gender_en = GENDER_MAP.get(input_data.get("genero", "Feminino"), "female")
        descricao_snippet = (input_data.get("descricao", "") or "")[:200]
        return f"""Analyze the reference images provided and generate 4 distinct professional marketing photo prompts plus 4 benefit bullets.

Product details:
- Type: {input_data.get("tipo_peca", "")}
- Material: {input_data.get("material", "")}
- Fabric design: {input_data.get("desenho_tecido", "")}
- Model gender: {gender_en}
- Listing title: {input_data.get("titulo", "")}
- Description excerpt: {descricao_snippet}

The garment color and shade must match exactly what is shown in the reference images. Do not describe or name the color — reproduce it as-is from the images."""

    def _call_llm(self, prompt: str) -> str:
        content_parts: list[dict] = []
        for b64 in self._images[:3]:
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_media_type(b64),
                    "data": b64,
                },
            })
        content_parts.append({"type": "text", "text": prompt})

        return call_llm(
            messages=[{"role": "user", "content": content_parts}],
            model=self.model,
            temperature=self.temperature,
            max_tokens=MAX_TOKENS,
            system=_build_system_prompt(self._input_data),
        )

    def _parse_output(self, raw: str) -> dict:
        slots = ["capa", "detalhe", "lifestyle", "objecao"]
        parsed = {slot: _extract_tag(raw, slot) for slot in slots}

        if not all(parsed.values()):
            return {"raw": raw}

        for i in range(1, 5):
            parsed[f"bullet{i}"] = _extract_tag(raw, f"bullet{i}") or ""

        return parsed
