"""
Teste de integração com a API do Mercado Livre.
Publica um anúncio de teste simples (sem imagens geradas) e remove em seguida.

Uso:
    .venv/bin/python scripts/test_ml_publish.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ml_auth import get_valid_token, load_tokens
from utils.ml_api import ml_get, ml_post, ml_validate, ml_post_description
from config import ML_APP_ID, ML_SECRET_KEY

import httpx

ML_BASE_URL = "https://api.mercadolibre.com"


def _fetch_size_grid_id(token: str) -> str | None:
    """Busca o SIZE_GRID_ID de um anúncio existente do seller."""
    data = ml_get("/users/3161803795/items/search?limit=20", token)
    for item_id in data.get("results", []):
        item = ml_get(f"/items/{item_id}", token)
        for attr in item.get("attributes", []):
            if attr.get("id") == "SIZE_GRID_ID" and attr.get("value_name"):
                return attr["value_name"]
    return None


def delete_item(item_id: str, token: str) -> None:
    httpx.put(
        f"{ML_BASE_URL}/items/{item_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"status": "closed"},
        timeout=15,
    )
    print(f"Anúncio {item_id} fechado (removido do ar).")


def main() -> None:
    if not ML_APP_ID or not ML_SECRET_KEY:
        print("ERRO: ML_APP_ID e ML_SECRET_KEY não configurados no .env")
        sys.exit(1)

    # Autorizar via --code se passado
    if "--code" in sys.argv:
        idx = sys.argv.index("--code")
        raw = sys.argv[idx + 1]
        from urllib.parse import urlparse, parse_qs
        from utils.ml_auth import authorize_with_code
        from config import ML_REDIRECT_URI
        code = parse_qs(urlparse(raw).query).get("code", [raw])[0]
        print(f"Trocando code por token...")
        authorize_with_code(code, ML_APP_ID, ML_SECRET_KEY, ML_REDIRECT_URI)
        print("Conta conectada com sucesso!")

    # 1. Verificar token
    token = get_valid_token(ML_APP_ID, ML_SECRET_KEY)
    if not token:
        from utils.ml_auth import build_auth_url
        from config import ML_REDIRECT_URI
        auth_url = build_auth_url(ML_APP_ID, ML_REDIRECT_URI)
        print("Conta não conectada. Acesse a URL abaixo, autorize e rode novamente com --code:\n")
        print(f"  {auth_url}\n")
        print("Exemplo:")
        print("  .venv/bin/python scripts/test_ml_publish.py --code 'https://httpbin.org/get?code=TG-...'")
        sys.exit(1)

    print(f"Token OK: {token[:20]}...")

    # Categoria: Leggings. SIZE_GRID_ID criado via API (Grade Legging Feminina PP-GG)
    category_id = "MLB278018"
    size_grid_id = "5295163"
    # Mapeamento tamanho → row_id da grade
    size_to_row = {"PP": "5295163:1", "P": "5295163:2", "M": "5295163:3", "G": "5295163:4", "GG": "5295163:5"}
    print(f"\n[Step 1] Categoria: Leggings ({category_id}) | SIZE_GRID_ID: {size_grid_id}")

    # Step 3: Upload de imagem (reutilizando imagem existente da conta)
    print("\n[Step 3] Fazendo upload da imagem de teste...")
    import httpx as _httpx
    img_bytes = _httpx.get("http://http2.mlstatic.com/D_866110-MLB107701764026_032026-O.jpg").content
    pic_response = ml_post(
        "/pictures/items/upload", token,
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    picture_id = pic_response["id"]
    print(f"  Picture ID: {picture_id}")

    item_json = {
        "title": "Legging Feminina Fitness Teste API",
        "category_id": category_id,
        "price": 49.90,
        "currency_id": "BRL",
        "buying_mode": "buy_it_now",
        "listing_type_id": "gold_special",
        "pictures": [{"id": picture_id}],
        "condition": "new",
        "attributes": [
            {"id": "ITEM_CONDITION", "value_id": "2230284", "value_name": "Novo"},
            {"id": "GENDER", "value_id": "339665", "value_name": "Feminino"},
            {"id": "BRAND", "value_name": "Genérica"},
            {"id": "COLOR", "value_name": "Preto"},
            {"id": "MODEL", "value_name": "Legging"},
            {"id": "SIZE_GRID_ID", "value_name": size_grid_id},
        ],
        "variations": [
            {
                "attribute_combinations": [{"id": "SIZE", "value_name": size}],
                "attributes": [{"id": "SIZE_GRID_ROW_ID", "value_name": row_id}],
                "available_quantity": 5,
                "price": 49.90,
                "picture_ids": [picture_id],
            }
            for size, row_id in size_to_row.items()
        ],
        "sale_terms": [
            {"id": "WARRANTY_TYPE", "value_name": "Garantia do vendedor"},
            {"id": "WARRANTY_TIME", "value_name": "90 dias"},
        ],
    }

    # 4. Validar (warnings de imagem são esperados no teste)
    print("\n[Step 4] Validando item...")
    errors = ml_validate("/items/validate", token, item_json)
    blocking = [e for e in errors if "picture" not in e.get("message", "").lower()]
    if blocking:
        print("  Erros de validação:")
        for e in blocking:
            print(f"    - {e.get('message')}")
        print("\nAbortando — corrija os erros antes de publicar.")
        sys.exit(1)
    print("  Validação OK (erros de imagem ignorados no teste)")

    # 5. Criar item
    print("\n[Step 5] Criando anúncio...")
    item = ml_post("/items", token, json_body=item_json)
    item_id = item["id"]
    permalink = item.get("permalink", "")
    status = item.get("status", "")
    print(f"  Item criado: {item_id}")
    print(f"  Status: {status}")
    print(f"  Link: {permalink}")

    # 6. Descrição
    print("\n[Step 6] Adicionando descrição...")
    ml_post_description(item_id, token, "Blusa de teste via API. Será removida em seguida.")
    print("  Descrição adicionada.")

    # 7. Verificar
    print("\n[Step 7] Verificando item...")
    final = ml_get(f"/items/{item_id}", token)
    print(f"  Status final: {final['status']}")
    print(f"  Imagens: {len(final.get('pictures', []))}")

    # 8. Remover anúncio de teste
    print("\n[Cleanup] Removendo anúncio de teste...")
    delete_item(item_id, token)

    print("\n✓ Integração funcionando corretamente!")


if __name__ == "__main__":
    main()
