import os

from agents.kit_image_agent import KitImageAgent
from agents.kit_listing_ml import KitListingMLAgent
from agents.kit_listing_shopee import KitListingShopeeAgent
from models.schemas import ListingOutputML, ListingOutputShopee
from utils.image_gen import generate_image
from utils.image_overlay import overlay_objection_text

KIT_SLOT_LABELS = {
    "capa_kit": "Capa do Kit",
    "detalhes_a": "Detalhes",    # label shown with color_a name in UI
    "detalhes_b": "Detalhes",    # label shown with color_b name in UI
    "objecao": "Quebra de Objeção",
    "lifestyle_kit": "Lifestyle",
}

# Slots that use only one color's reference images
_REFS_FOR_SLOT = {
    "detalhes_a": "a",
    "detalhes_b": "b",
}


class KitPipeline:
    def run_combination(
        self,
        images_a: list[str],
        images_b: list[str],
        color_a: str,
        color_b: str,
        product_data: dict,
        marketplaces: list[str],
        output_dir: str,
    ) -> dict:
        combo_key = f"{color_a}_{color_b}".replace(" ", "_")
        all_images = images_a[:2] + images_b[:2]

        images = self._generate_images(
            images_a, images_b, color_a, color_b, product_data, output_dir, combo_key
        )
        listings = self._generate_listings(
            all_images, color_a, color_b, product_data, marketplaces
        )

        return {"images": images, "listings": listings, "color_a": color_a, "color_b": color_b}

    def _generate_images(
        self,
        images_a: list[str],
        images_b: list[str],
        color_a: str,
        color_b: str,
        product_data: dict,
        output_dir: str,
        combo_key: str,
    ) -> dict[str, str | None]:
        agent = KitImageAgent()
        prompts = agent.run({
            **product_data,
            "images_a": images_a,
            "images_b": images_b,
            "color_a": color_a,
            "color_b": color_b,
        })

        if "raw" in prompts:
            return {slot: None for slot in KIT_SLOT_LABELS}

        all_refs = images_a[:2] + images_b[:2]
        results: dict[str, str | None] = {}

        for slot in KIT_SLOT_LABELS:
            output_path = os.path.join(output_dir, f"{combo_key}_{slot}.png")

            ref_set = _REFS_FOR_SLOT.get(slot)
            if ref_set == "a":
                refs = images_a[:2]
            elif ref_set == "b":
                refs = images_b[:2]
            else:
                refs = all_refs

            try:
                path = generate_image(prompts[slot], refs, output_path)
                if slot == "objecao" and path:
                    bullets = [prompts.get(f"bullet{k}", "") for k in range(1, 5)]
                    bullets = [b for b in bullets if b]
                    if bullets:
                        path = overlay_objection_text(path, bullets, output_path)
                results[slot] = path
            except Exception as e:
                print(f"[kit_pipeline] Image {slot} failed: {e}")
                results[slot] = None

        return results

    def _generate_listings(
        self,
        all_images: list[str],
        color_a: str,
        color_b: str,
        product_data: dict,
        marketplaces: list[str],
    ) -> dict:
        kit_input = {
            **product_data,
            "imagens": all_images,
            "color_a": color_a,
            "color_b": color_b,
        }

        listings = {}

        if "Mercado Livre" in marketplaces:
            result = KitListingMLAgent().run(kit_input)
            listings["ml"] = result if "raw" in result else ListingOutputML(**result)

        if "Shopee" in marketplaces:
            result = KitListingShopeeAgent().run(kit_input)
            listings["shopee"] = result if "raw" in result else ListingOutputShopee(**result)

        return listings
