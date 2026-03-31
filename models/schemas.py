from pydantic import BaseModel


class ListingOutputML(BaseModel):
    titulo: str
    palavras_chave: str
    modelo_keywords: str
    modelo: str
    esportes_recomendados: list[str]
    descricao: str
    skus: list[str]
    tipo_manga: str
    tipo_gola: str
    estilos: list[str]
    caimento: str
    temporada: str


class ListingOutputShopee(BaseModel):
    nome: str
    categoria: str
    descricao: str
    skus: list[str]
    estacoes: str
    estilo: list[str]
    comprimento_superior: str
    modelo: list[str]
