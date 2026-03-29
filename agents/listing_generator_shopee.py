import re
from agents.base import BaseAgent
from utils.llm import call_llm
from config import MAX_TOKENS

CATEGORIAS_SHOPEE = """
Roupas Femininas > Blusas > Body
Roupas Femininas > Blusas > Camisas e Blusas
Roupas Femininas > Blusas > Camisas Polo
Roupas Femininas > Blusas > Camisetas
Roupas Femininas > Blusas > Regatas
Roupas Femininas > Blusas > Top faixa e Croppeds
Roupas Femininas > Blusas > Outros
Roupas Femininas > Calças e Leggings > Calças
Roupas Femininas > Calças e Leggings > Leggings e Treggings
Roupas Femininas > Calças e Leggings > Outros
Roupas Femininas > Shorts > Short saia
Roupas Femininas > Shorts > Shorts
Roupas Femininas > Shorts > Outros
Roupas Femininas > Saias
Roupas Femininas > Jeans
Roupas Femininas > Vestidos
Roupas Femininas > Vestidos de Casamento
Roupas Femininas > Macacões, Macaquinhos e Jardineiras
Roupas Femininas > Jaquetas, Casacos e Coletes > Blazers
Roupas Femininas > Jaquetas, Casacos e Coletes > Capas
Roupas Femininas > Jaquetas, Casacos e Coletes > Coletes
Roupas Femininas > Jaquetas, Casacos e Coletes > Jaquetas
Roupas Femininas > Jaquetas, Casacos e Coletes > Jaquetas e Casacos de Inverno
Roupas Femininas > Jaquetas, Casacos e Coletes > Outros
Roupas Femininas > Agasalhos e Cardigans
Roupas Femininas > Moletons e Suéteres
Roupas Femininas > Sets (Conjuntos)
Roupas Femininas > Lingerie e Roupa Íntima > Calças de segurança
Roupas Femininas > Lingerie e Roupa Íntima > Calcinhas
Roupas Femininas > Lingerie e Roupa Íntima > Cinta modeladora
Roupas Femininas > Lingerie e Roupa Íntima > Conjuntos
Roupas Femininas > Lingerie e Roupa Íntima > Lingerie sexy
Roupas Femininas > Lingerie e Roupa Íntima > Roupa íntima térmica
Roupas Femininas > Lingerie e Roupa Íntima > Sutiãs
Roupas Femininas > Lingerie e Roupa Íntima > Outros
Roupas Femininas > Traje para Dormir e Pijamas
Roupas Femininas > Meias > Meia calça
Roupas Femininas > Meias > Meias
Roupas Femininas > Meias > Outros
Roupas Femininas > Roupas de Maternidade > Blusas de maternidade
Roupas Femininas > Roupas de Maternidade > Calças de maternidade
Roupas Femininas > Roupas de Maternidade > Conjuntos de maternidade
Roupas Femininas > Roupas de Maternidade > Roupas de amamentação
Roupas Femininas > Roupas de Maternidade > Sutiã de amamentação
Roupas Femininas > Roupas de Maternidade > Vestido de maternidade
Roupas Femininas > Roupas de Maternidade > Outros
Roupas Femininas > Roupas Tradicionais
Roupas Femininas > Fantasias e Cosplay
Roupas Femininas > Tecidos
Roupas Femininas > Outros
"""

SYSTEM_PROMPT = f"""Você é um especialista em criar anúncios de roupa otimizados para a Shopee Brasil.

CATEGORIAS DISPONÍVEIS:
{CATEGORIAS_SHOPEE.strip()}

Regras para NOME DO PRODUTO:
- Máximo 120 caracteres
- Inclua tipo de peça, material, público-alvo, usos e diferenciais
- Use termos que compradores reais pesquisam na Shopee
- Sem nome de marca

Regras para CATEGORIA:
- Escolha o caminho mais específico possível da lista acima
- Formato exato: "Roupas Femininas > Subcategoria > Sub-subcategoria"
- Se não houver sub-subcategoria, pare no nível disponível

Regras para SKUs:
- Gere um SKU para cada combinação de cor e tamanho fornecidos
- Formato exato: [SKU_BASE]-COR-TAMANHO (tudo em CAPS LOCK, sem acentos)
- Um SKU por linha, sem numeração nem marcadores

Regras para ESTAÇÕES:
- Escolha EXATAMENTE uma: Verão, Outono, Inverno ou Primavera
- Baseie na peça e no material

Regras para ESTILO:
- Até 5 estilos separados por vírgula
- Seja criativo com termos que usuários pesquisariam
- Ex: Casual, Esportivo, Fitness, Minimalista, Streetwear, Moderno, Boho, Elegante

Regras para COMPRIMENTO DA PARTE SUPERIOR:
- Baseado nas imagens: Curto, Médio ou Longo

Regras para MODELO (search terms):
- Até 5 termos ou frases que um comprador pesquisaria
- Separados por vírgula, criativos e específicos ao produto
- Ex: cropped academia, blusa fitness feminina, top dry fit

Regras para DESCRIÇÃO:
- Plain text sem HTML e sem markdown
- Quebras de linha com \\n
- Tom persuasivo e direto
- Siga EXATAMENTE esta estrutura:

1. LINHA DE TÍTULO: [TIPO DA PEÇA EM MAIÚSCULAS] | [MATERIAL] | [USOS PRINCIPAIS]

2. Parágrafo de abertura (2-3 linhas): para quem é, material, benefício principal e ocasiões de uso.

3. Seção "POR QUE VOCE VAI AMAR":
   - 6 a 8 benefícios com traço (-)
   - Cada tópico: NOME EM MAIÚSCULAS: explicação curta

4. Seção "ESPECIFICAÇÕES":
   - Gênero, Tipo, Material, Tamanhos, Origem, Lavagem

5. Seção "DÚVIDAS FREQUENTES":
   - 4 a 5 perguntas e respostas. Formato: "- Pergunta? Resposta."

6. Seção "O QUE VOCÊ RECEBE":
   - Uma linha descrevendo o que o comprador recebe

Responda EXCLUSIVAMENTE com as tags XML abaixo, sem nenhum texto fora delas:
<nome>máx 120 caracteres, sem marca</nome>
<categoria>Roupas Femininas > X > Y</categoria>
<descricao>texto persuasivo em plain text, use quebras de linha reais entre seções</descricao>
<skus>SKU-COR-TAMANHO
SKU-COR-TAMANHO
...</skus>
<estacoes>Verão / Outono / Inverno / Primavera — escolha apenas uma</estacoes>
<estilo>estilo1, estilo2, estilo3, estilo4, estilo5</estilo>
<comprimento_superior>Curto / Médio / Longo — baseado nas imagens</comprimento_superior>
<modelo>termo1, termo2, termo3, termo4, termo5</modelo>"""


def _detect_media_type(b64_data: str) -> str:
    prefix = b64_data[:12]
    if prefix.startswith("/9j"):
        return "image/jpeg"
    if prefix.startswith("iVBOR"):
        return "image/png"
    if prefix.startswith("R0lGO"):
        return "image/gif"
    if prefix.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


class ListingGeneratorShopeeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="listing_generator_shopee",
            model="claude-sonnet-4-20250514",
            temperature=0,
        )
        self._images: list[str] = []

    def run(self, input_data: dict) -> dict:
        self._images = input_data.get("imagens", [])
        return super().run(input_data)

    def _build_prompt(self, input_data: dict) -> str:
        tamanhos = ", ".join(input_data.get("tamanhos", [])) or "não informado"
        imagens_aviso = (
            "Analise as imagens enviadas para entender o produto visualmente."
            if self._images
            else "Nenhuma imagem foi enviada. Gere o conteúdo com base apenas nas informações textuais."
        )

        return f"""{imagens_aviso}

Gere um anúncio completo para a Shopee com os dados abaixo:

- Tipo de peça: {input_data.get("tipo_peca", "")}
- Gênero: {input_data.get("genero", "")}
- Material: {input_data.get("material", "")}
- Desenho do tecido: {input_data.get("desenho_tecido", "")}
- Cores disponíveis: {input_data.get("cores", "")}
- Tamanhos disponíveis: {tamanhos}
- SKU base: {input_data.get("sku_base", "")}
- Preço original: R$ {input_data.get("preco_original", "")}
- Preço com desconto: R$ {input_data.get("preco_desconto", "")}

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

        skus = [s.strip() for s in skus_raw.replace("\\n", "\n").split("\n") if s.strip()]
        estilo = [e.strip() for e in estilo_raw.split(",") if e.strip()]
        modelo = [m.strip() for m in modelo_raw.split(",") if m.strip()]

        return {
            "nome": nome,
            "categoria": categoria,
            "descricao": descricao.replace("\\n", "\n"),
            "skus": skus,
            "estacoes": estacoes,
            "estilo": estilo,
            "comprimento_superior": comprimento_superior,
            "modelo": modelo,
        }
