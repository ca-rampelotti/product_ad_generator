import re
from agents.base import BaseAgent
from utils.llm import call_llm
from config import MAX_TOKENS

SYSTEM_PROMPT = """Você é um especialista em criar anúncios de roupa otimizados para o Mercado Livre Brasil.

Regras para o TÍTULO:
- Máximo 60 caracteres
- Estrutura: [Tipo de peça] [Característica principal] [Uso/Nicho] [Gênero]
- PROIBIDO: nome de marca, cores, palavras como "estoque", "disponível", "novo", "original", "grátis"
- Use termos que compradores reais digitam no ML

Regras para o campo MODELO (keywords):
- MAXIMIZE keywords, sinônimos e variações de busca relevantes para o produto
- Inclua variações do nome, termos de uso, público-alvo e nicho
- Escreva os termos separados por espaço, sem pontuação
- Quanto mais termos relevantes, melhor a descoberta do produto

Regras para PALAVRAS-CHAVE:
- Lista de termos descritivos do produto separados por vírgula
- Sem conectivos (e, ou, para, com, de, em, que...)
- Máximo descritivo: tipo de peça, material, tecnologia, público-alvo, ocasião, benefícios
- Ex: "cropped, feminino, dry fit, poliamida, academia, fitness, treino, yoga, pilates, verão"

Regras para ESPORTES RECOMENDADOS:
- Para fitness/esportivo: lista abrangente de esportes e atividades compatíveis
- Para casual/social/moda: lista menor mas relevante (caminhada, viagem, rotina diária)
- Adapte ao tipo de peça e às imagens recebidas

Regras para SKUs:
- Gere um SKU para cada combinação de cor e tamanho fornecidos
- Formato exato: [SKU_BASE]-COR-TAMANHO (tudo em CAPS LOCK, sem acentos)
- Um SKU por linha, sem numeração nem marcadores
- Ex: CROPPED01-PRETO-P\\nCROPPED01-PRETO-M\\nCROPPED01-ROSA-P

Regras para DESCRIÇÃO:
- Plain text sem HTML e sem markdown
- Quebras de linha com \\n
- Tom persuasivo e direto
- Siga EXATAMENTE esta estrutura de seções:

1. LINHA DE TÍTULO: [TIPO DA PEÇA EM MAIÚSCULAS] | [MATERIAL] | [USOS PRINCIPAIS]

2. Parágrafo de abertura (2-3 linhas): para quem é, material, benefício principal e ocasiões de uso.

3. Seção "POR QUE VOCE VAI AMAR" (sem dois pontos no título):
   - Liste 6 a 8 benefícios em tópicos com traço (-)
   - Cada tópico: nome do benefício em maiúsculas seguido de dois pontos e explicação curta
   - Ex: "- Secagem ultra-rápida: elimina a umidade e mantém você seco durante o treino"

4. Seção "ESPECIFICAÇÕES":
   - Lista com traço (-) dos atributos técnicos: Gênero, Tipo, Material, Tamanhos, Origem, Lavagem

5. Seção "DÚVIDAS FREQUENTES":
   - 4 a 5 perguntas e respostas relevantes ao produto
   - Formato: "- Pergunta? Resposta."

6. Seção "O QUE VOCÊ RECEBE":
   - Uma linha descrevendo o que o comprador recebe

Responda EXCLUSIVAMENTE com as tags XML abaixo, sem nenhum texto fora delas:
<titulo>máx 60 caracteres, sem marca, sem cor, sem condição</titulo>
<palavras_chave>termo1, termo2, termo3, ...</palavras_chave>
<modelo>máximo de keywords e sinônimos relevantes separados por espaço</modelo>
<esportes>esporte1, esporte2, esporte3</esportes>
<descricao>texto persuasivo em plain text, use quebras de linha reais entre seções</descricao>
<skus>SKU-COR-TAMANHO
SKU-COR-TAMANHO
...</skus>
<tipo_manga>Sem mangas / Curta / Longa / 3/4 / Cavada — baseado nas imagens</tipo_manga>
<tipo_gola>Redonda / V / Alta / Nadador / Sem gola — baseado nas imagens</tipo_gola>
<estilos>casual, praia, fashion, esportivo, urbano — estilos aplicáveis ao produto</estilos>
<caimento>Reta / Ajustada / Solta / Oversized — baseado nas imagens</caimento>
<temporada>Verão / Inverno / Outono-Inverno / Primavera-Verão / Todas as estações</temporada>"""


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


class ListingGeneratorMLAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="listing_generator_ml",
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

Gere um anúncio completo para o Mercado Livre com os dados abaixo:

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
