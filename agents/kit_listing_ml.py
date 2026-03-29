import re
from agents.base import BaseAgent
from agents.listing_generator_ml import SYSTEM_PROMPT, _detect_media_type, _extract_tag
from utils.llm import call_llm
from config import MAX_TOKENS


class KitListingMLAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="kit_listing_ml",
            model="claude-sonnet-4-20250514",
            temperature=0,
        )
        self._images: list[str] = []

    def run(self, input_data: dict) -> dict:
        self._images = input_data.get("imagens", [])
        return super().run(input_data)

    def _build_prompt(self, input_data: dict) -> str:
        color_a = input_data.get("color_a", "")
        color_b = input_data.get("color_b", "")
        tamanhos = ", ".join(input_data.get("tamanhos", [])) or "não informado"
        imagens_aviso = (
            "Analise as imagens enviadas para entender o produto visualmente."
            if self._images
            else "Nenhuma imagem foi enviada. Gere o conteúdo com base nas informações textuais."
        )

        return f"""{imagens_aviso}

Este é um KIT composto por duas peças: {input_data.get("tipo_peca", "")} na cor {color_a} E {input_data.get("tipo_peca", "")} na cor {color_b}.

Gere um anúncio completo para o Mercado Livre com os dados abaixo:

- Tipo de peça: Kit de {input_data.get("tipo_peca", "")}
- Cores do kit: {color_a} + {color_b}
- Gênero: {input_data.get("genero", "")}
- Material: {input_data.get("material", "")}
- Desenho do tecido: {input_data.get("desenho_tecido", "")}
- Tamanhos disponíveis: {tamanhos}
- SKU base: {input_data.get("sku_base", "")} — use o formato [SKU_BASE]-KIT-[COR_A]-[COR_B]-TAMANHO (sem acentos, CAPS LOCK)
- Preço original: R$ {input_data.get("preco_original", "")}
- Preço com desconto: R$ {input_data.get("preco_desconto", "")}

Adapte o título, descrição e palavras-chave para refletir que é um kit com duas cores.
Siga rigorosamente as regras do system prompt e responda apenas com as tags XML."""

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
        titulo = _extract_tag(raw, "titulo")
        palavras_chave = _extract_tag(raw, "palavras_chave")
        modelo = _extract_tag(raw, "modelo")
        esportes_raw = _extract_tag(raw, "esportes")
        descricao = _extract_tag(raw, "descricao")
        skus_raw = _extract_tag(raw, "skus")
        tipo_manga = _extract_tag(raw, "tipo_manga")
        tipo_gola = _extract_tag(raw, "tipo_gola")
        estilos_raw = _extract_tag(raw, "estilos")
        caimento = _extract_tag(raw, "caimento")
        temporada = _extract_tag(raw, "temporada")

        if not all([titulo, palavras_chave, modelo, esportes_raw, descricao, skus_raw, tipo_manga, tipo_gola]):
            return {"raw": raw}

        esportes = [e.strip() for e in esportes_raw.split(",") if e.strip()]
        skus = [s.strip() for s in skus_raw.replace("\\n", "\n").split("\n") if s.strip()]
        estilos = [e.strip() for e in estilos_raw.split(",") if e.strip()] if estilos_raw else []

        return {
            "titulo": titulo,
            "palavras_chave": palavras_chave,
            "modelo_keywords": modelo,
            "esportes_recomendados": esportes,
            "descricao": descricao.replace("\\n", "\n"),
            "skus": skus,
            "tipo_manga": tipo_manga,
            "tipo_gola": tipo_gola,
            "estilos": estilos,
            "caimento": caimento or "",
            "temporada": temporada or "",
        }
