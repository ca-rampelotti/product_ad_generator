from agents.listing_generator_ml import ListingGeneratorMLAgent
from agents.listing_generator_shopee import ListingGeneratorShopeeAgent
from models.schemas import ListingOutputML, ListingOutputShopee


class ListingPipeline:
    def run(
        self,
        marketplaces: list[str],
        imagens: list[str],
        tipo_peca: str,
        genero: str,
        material: str,
        desenho_tecido: str,
        cores: str,
        tamanhos: list[str],
        sku_base: str,
        preco_original: float,
        preco_desconto: float,
    ) -> dict:
        input_data = {
            "imagens": imagens,
            "tipo_peca": tipo_peca,
            "genero": genero,
            "material": material,
            "desenho_tecido": desenho_tecido,
            "cores": cores,
            "tamanhos": tamanhos,
            "sku_base": sku_base,
            "preco_original": preco_original,
            "preco_desconto": preco_desconto,
        }

        results = {}

        if "Mercado Livre" in marketplaces:
            result = ListingGeneratorMLAgent().run(input_data)
            results["ml"] = result if "raw" in result else ListingOutputML(**result)

        if "Shopee" in marketplaces:
            result = ListingGeneratorShopeeAgent().run(input_data)
            results["shopee"] = result if "raw" in result else ListingOutputShopee(**result)

        return results
