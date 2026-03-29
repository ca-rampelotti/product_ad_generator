import re
from agents.base import BaseAgent
from agents.listing_generator_shopee import SYSTEM_PROMPT, _detect_media_type, _extract_tag
from utils.llm import call_llm
from config import MAX_TOKENS


class KitListingShopeeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="kit_listing_shopee",
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

Gere um anúncio completo para a Shopee com os dados abaixo:

- Tipo de peça: Kit de {input_data.get("tipo_peca", "")}
- Cores do kit: {color_a} + {color_b}
- Gênero: {input_data.get("genero", "")}
- Material: {input_data.get("material", "")}
- Desenho do tecido: {input_data.get("desenho_tecido", "")}
- Tamanhos disponíveis: {tamanhos}
- SKU base: {input_data.get("sku_base", "")} — use o formato [SKU_BASE]-KIT-[COR_A]-[COR_B]-TAMANHO (sem acentos, CAPS LOCK)
- Preço original: R$ {input_data.get("preco_original", "")}
- Preço com desconto: R$ {input_data.get("preco_desconto", "")}

Adapte nome, categoria e descrição para refletir que é um kit com duas cores.
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
        nome = _extract_tag(raw, "nome")
        categoria = _extract_tag(raw, "categoria")
        descricao = _extract_tag(raw, "descricao")
        skus_raw = _extract_tag(raw, "skus")
        estacoes = _extract_tag(raw, "estacoes")
        estilo_raw = _extract_tag(raw, "estilo")
        comprimento_superior = _extract_tag(raw, "comprimento_superior")
        modelo_raw = _extract_tag(raw, "modelo")

        if not all([nome, categoria, descricao, skus_raw, estacoes, estilo_raw, comprimento_superior, modelo_raw]):
            return {"raw": raw}

        return {
            "nome": nome,
            "categoria": categoria,
            "descricao": descricao.replace("\\n", "\n"),
            "skus": [s.strip() for s in skus_raw.replace("\\n", "\n").split("\n") if s.strip()],
            "estacoes": estacoes,
            "estilo": [e.strip() for e in estilo_raw.split(",") if e.strip()],
            "comprimento_superior": comprimento_superior,
            "modelo": [m.strip() for m in modelo_raw.split(",") if m.strip()],
        }
