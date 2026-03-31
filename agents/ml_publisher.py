import unicodedata
from pathlib import Path
from urllib.parse import quote

from agents.base import BaseAgent
from utils.ml_api import ml_get, ml_post, ml_validate, ml_post_description

DEFAULT_LISTING_TYPE = "gold_special"
DEFAULT_ESTOQUE = 10

# Medidas de quadril padrão por tamanho (cm) — usadas na criação do size grid
HIP_MEASURES = {"PP": 82, "P": 86, "M": 90, "G": 94, "GG": 98, "XGG": 102}


class MLPublisherAgent(BaseAgent):
    """Publica um anúncio completo no Mercado Livre via API (7 steps)."""

    def __init__(self):
        super().__init__(name="ml_publisher", model="", temperature=0)

    def run(self, input_data: dict) -> dict:
        """
        input_data:
            access_token: str
            titulo: str
            descricao: str
            preco: float
            tamanhos: list[str]
            cores: list[str]
            skus: list[str]
            image_paths_by_color: dict[str, list[str]]  — {color: [path, ...]}
            listing_type: str  (opcional, default gold_special)
            estoque_por_variacao: int | dict[cor][tamanho]  (opcional, default 10)
        """
        access_token = input_data["access_token"]
        titulo = input_data["titulo"]
        descricao = input_data["descricao"]
        preco = float(input_data["preco"])
        tamanhos: list[str] = input_data.get("tamanhos", [])
        cores: list[str] = input_data.get("cores", [])
        skus: list[str] = input_data.get("skus", [])
        image_paths_by_color: dict[str, list[str]] = input_data.get("image_paths_by_color", {})
        listing_type = input_data.get("listing_type", DEFAULT_LISTING_TYPE)
        estoque_input = input_data.get("estoque_por_variacao", DEFAULT_ESTOQUE)
        estoque_por_variacao: int | dict = (
            estoque_input if isinstance(estoque_input, dict) else int(estoque_input)
        )

        # Step 1: predict category + domain
        category_id, domain_id = self._predict_category(titulo, access_token)

        # Step 2: fetch or create size grid for this domain
        size_grid_id, size_to_row = self._get_size_grid(domain_id, tamanhos, access_token)

        # Step 3: upload all images, grouped by color
        picture_ids_by_color: dict[str, list[str]] = {}
        all_picture_ids: list[str] = []
        for color, paths in image_paths_by_color.items():
            ids = self._upload_images(paths, access_token)
            picture_ids_by_color[color] = ids
            all_picture_ids.extend(ids)

        # Step 4 + 5: build, validate, create
        item_json = _build_item_json(
            titulo=titulo,
            category_id=category_id,
                preco=preco,
            listing_type=listing_type,
            all_picture_ids=all_picture_ids,
            tamanhos=tamanhos,
            cores=cores,
            skus=skus,
            estoque_por_variacao=estoque_por_variacao,
            picture_ids_by_color=picture_ids_by_color,
            size_grid_id=size_grid_id,
            size_to_row=size_to_row,
            modelo=input_data.get("modelo", ""),
        )

        errors = ml_validate("/items/validate", access_token, item_json)
        if errors:
            msgs = "; ".join(e.get("message", str(e)) for e in errors)
            raise ValueError(f"Validação ML falhou: {msgs}")

        item = ml_post("/items", access_token, json_body=item_json)
        item_id = item["id"]

        # Step 6: description
        ml_post_description(item_id, access_token, descricao)

        # Step 7: verify
        final = ml_get(f"/items/{item_id}", access_token)

        return {
            "item_id": item_id,
            "permalink": final.get("permalink", ""),
            "status": final.get("status", ""),
        }

    def _predict_category(self, titulo: str, access_token: str) -> tuple[str, str]:
        result = ml_get(
            f"/sites/MLB/domain_discovery/search?q={quote(titulo)}&limit=3",
            access_token,
        )
        if result and isinstance(result, list):
            domain_id = result[0].get("domain_id", "").replace("MLB-", "")
            return result[0]["category_id"], domain_id
        raise RuntimeError("Não foi possível predizer a categoria do produto")

    def _get_size_grid(
        self, domain_id: str, tamanhos: list[str], access_token: str
    ) -> tuple[str | None, dict[str, str]]:
        """Cria um novo size grid para o domínio com os tamanhos do anúncio."""
        if not tamanhos:
            return None, {}
        return _create_size_grid(domain_id, tamanhos, access_token)

    def _upload_images(self, paths: list[str], access_token: str) -> list[str]:
        ids = []
        for path in paths:
            if not Path(path).exists():
                continue
            with open(path, "rb") as f:
                data = f.read()
            response = ml_post(
                "/pictures/items/upload",
                access_token,
                files={"file": (Path(path).name, data, "image/png")},
            )
            if "id" in response:
                ids.append(response["id"])
        return ids

    # BaseAgent interface — não utilizado (sem LLM neste agente)
    def _build_prompt(self, input_data: dict) -> str:
        raise NotImplementedError

    def _parse_output(self, raw: str) -> dict:
        raise NotImplementedError

    def _call_llm(self, prompt: str) -> str:
        raise NotImplementedError


# ─── helpers ────────────────────────────────────────────────────────────────


def _build_size_to_row_map(grid: dict, tamanhos: list[str]) -> dict[str, str]:
    """Mapeia tamanho → row_id a partir de um size grid já carregado."""
    mapping: dict[str, str] = {}
    for row in grid.get("rows", []):
        for attr in row.get("attributes", []):
            if attr.get("id") == "SIZE":
                values = attr.get("values", [])
                if values:
                    size_name = values[0].get("name", "")
                    if size_name in tamanhos:
                        mapping[size_name] = row["id"]
    return mapping


def _create_size_grid(
    domain_id: str, tamanhos: list[str], access_token: str
) -> tuple[str, dict[str, str]]:
    """Cria um size grid no ML para o domínio e tamanhos informados."""
    rows = []
    for size in tamanhos:
        hip = HIP_MEASURES.get(size, 90)
        rows.append({
            "attributes": [
                {"id": "SIZE", "values": [{"name": size}]},
                {"id": "FILTRABLE_SIZE", "values": [{"name": size}]},
                {"id": "GARMENT_HIP_WIDTH_FROM", "values": [
                    {"name": f"{hip} cm", "struct": {"number": hip, "unit": "cm"}}
                ]},
            ]
        })

    payload = {
        "site_id": "MLB",
        "domain_id": domain_id,
        "names": {"MLB": f"Grade {domain_id} {'-'.join(tamanhos)}"},
        "measure_type": "CLOTHING_MEASURE",
        "main_attribute_id": "SIZE",
        "attributes": [{"id": "GENDER", "values": [{"id": "339665", "name": "Feminino"}]}],
        "rows": rows,
    }

    grid = ml_post("/catalog/charts", access_token, json_body=payload)
    grid_id = grid["id"]
    size_to_row = {
        row["id"].split(":")[-1]: f'{grid_id}:{i + 1}'
        for i, row in enumerate(grid.get("rows", []))
    }
    # Montar mapeamento tamanho → row_id
    size_to_row = _build_size_to_row_map(grid, tamanhos)
    return grid_id, size_to_row


def _build_item_json(
    titulo: str,
    category_id: str,
    preco: float,
    listing_type: str,
    all_picture_ids: list[str],
    tamanhos: list[str],
    cores: list[str],
    skus: list[str],
    estoque_por_variacao: int | dict,
    picture_ids_by_color: dict[str, list[str]],
    size_grid_id: str | None,
    size_to_row: dict[str, str],
    modelo: str = "",
) -> dict:
    attributes = [
        {"id": "ITEM_CONDITION", "value_id": "2230284", "value_name": "Novo"},
        {"id": "GENDER", "value_id": "339665", "value_name": "Feminino"},
        {"id": "BRAND", "value_name": "Genérica"},
    ]
    if modelo:
        attributes.append({"id": "MODEL", "value_name": modelo})
    if size_grid_id:
        attributes.append({"id": "SIZE_GRID_ID", "value_name": size_grid_id})

    item: dict = {
        "title": titulo[:60],
        "category_id": category_id,
        "price": preco,
        "currency_id": "BRL",
        "buying_mode": "buy_it_now",
        "listing_type_id": listing_type,
        "condition": "new",
        "channels": ["marketplace", "mshops"],
        "pictures": [{"id": pid} for pid in all_picture_ids],
        "attributes": attributes,
        "sale_terms": [
            {"id": "WARRANTY_TYPE", "value_name": "Garantia do vendedor"},
            {"id": "WARRANTY_TIME", "value_name": "90 dias"},
        ],
    }

    if tamanhos and cores:
        item["variations"] = _build_variations(
            cores=cores,
            tamanhos=tamanhos,
            skus=skus,
            preco=preco,
            estoque_por_variacao=estoque_por_variacao,
            picture_ids_by_color=picture_ids_by_color,
            size_to_row=size_to_row,
        )
    else:
        item["available_quantity"] = estoque_por_variacao

    return item


def _build_variations(
    cores: list[str],
    tamanhos: list[str],
    skus: list[str],
    preco: float,
    estoque_por_variacao: int | dict,
    picture_ids_by_color: dict[str, list[str]],
    size_to_row: dict[str, str],
) -> list[dict]:
    variations = []
    for color in cores:
        color_key = color.strip()
        pic_ids = picture_ids_by_color.get(color_key, [])
        for size in tamanhos:
            if isinstance(estoque_por_variacao, dict):
                qty = int(estoque_por_variacao.get(color_key, {}).get(size, DEFAULT_ESTOQUE))
            else:
                qty = estoque_por_variacao

            variation: dict = {
                "attribute_combinations": [
                    {"id": "COLOR", "value_name": color_key.capitalize()},
                    {"id": "SIZE", "value_name": size},
                ],
                "available_quantity": qty,
                "price": preco,
                "seller_custom_field": _find_sku(skus, color_key, size),
            }
            if pic_ids:
                variation["picture_ids"] = pic_ids
            if size in size_to_row:
                variation["attributes"] = [
                    {"id": "SIZE_GRID_ROW_ID", "value_name": size_to_row[size]}
                ]
            variations.append(variation)
    return variations


def _find_sku(skus: list[str], color: str, size: str) -> str:
    color_norm = _strip_accents(color).upper()
    size_norm = size.upper()
    for sku in skus:
        sku_norm = _strip_accents(sku).upper()
        if color_norm in sku_norm and size_norm in sku_norm:
            return sku
    return f"{_strip_accents(color).upper()}-{size_norm}"


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))
