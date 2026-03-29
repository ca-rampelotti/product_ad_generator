import base64
import io
import os
import shutil
import sys
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from jinja2 import Environment, FileSystemLoader

# Injeta o segredo do Streamlit Cloud no ambiente antes de importar os módulos do
# projeto, para que config.py leia GOOGLE_API_KEY corretamente via os.getenv.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass  # fallback: config.py lê do .env local via python-dotenv

from agents.image_generator import ImageGeneratorAgent
from pipelines.kit_pipeline import KitPipeline, KIT_SLOT_LABELS
from pipelines.listing_pipeline import ListingPipeline
from utils.image_gen import generate_image, SLOT_LABELS, SLOT_ZIP_NAMES
from utils.image_overlay import overlay_objection_text
from utils.image_renderer import render_html_to_image
from utils.size_table_extractor import extract_size_table

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")

FISCAL_INFO = {
    "NCM": "61091000",
    "Origem": "0",
    "CFOP": "5101",
    "CSOSN": "102",
    "CFOP diferentes estados": "6107",
    "PIS e COFINS": "99",
}

PACKAGE_INFO = {
    "Peso": "0,1 kg",
    "Comprimento": "30 cm",
    "Largura": "20 cm",
    "Altura": "1 cm",
}

CORES_OPCOES = [
    "branco", "preto", "vermelho", "caramelo", "marrom",
    "rosa", "amarelo", "verde", "azul", "bordo",
]

st.set_page_config(page_title="Gerador de Anúncios", layout="wide")

# ── Autenticação ──
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("Gerador de Anúncios")
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        valid_user = st.secrets.get("LOGIN_USER", "admin")
        valid_pass = st.secrets.get("LOGIN_PASSWORD", "admin")
        if username == valid_user and password == valid_pass:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")
    st.stop()

st.title("Gerador de Anúncios")

# ── Session state ──
for key, default in [
    ("inputs", {}),
    ("imagens_por_cor", {}),
    ("marketplaces", []),
    ("gen_running", False),
    ("listing", {}),
    ("listing_error", ""),
    ("size_table_data", None),
    ("size_table_image", None),
    ("image_prompts", {}),
    ("generated_images", {}),
    ("image_errors", {}),
    ("image_bullets", {}),
    ("kit_results", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _copy_js(text: str) -> None:
    """Inject JS to copy text to clipboard."""
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(
        f"""<script>
        navigator.clipboard.writeText(`{escaped}`);
        </script>
        <p style="font-size:13px; color:#888; margin:0;">Copiado!</p>
        """,
        height=25,
    )


def _copyable_field(label: str, value: str, key: str) -> None:
    """Campo read-only com label, valor e botão copiar."""
    c_label, c_value, c_copy = st.columns([1.2, 3, 0.4])
    c_label.markdown(f"**{label}**")
    c_value.code(value, language=None)
    if c_copy.button("📋", key=f"cp_{key}"):
        _copy_js(value)


def _editable_field(label: str, value: str, key: str) -> None:
    """Campo editável com botão copiar (copia o valor editado)."""
    c_field, c_copy = st.columns([4, 0.4])
    c_field.text_input(label, value=value, key=f"edit_{key}")
    if c_copy.button("📋", key=f"cp_{key}"):
        _copy_js(st.session_state.get(f"edit_{key}", value))


def _editable_text_area(label: str, value: str, key: str) -> None:
    """Campo de texto longo editável com botão copiar."""
    col_title, col_copy = st.columns([4, 0.4])
    col_title.markdown(f"**{label}**")
    if col_copy.button("📋", key=f"cp_{key}"):
        _copy_js(st.session_state.get(f"ta_{key}", value))
    st.text_area(
        label, value=value, height=300,
        label_visibility="collapsed", key=f"ta_{key}",
    )


def _section_header(title: str) -> None:
    """Cabeçalho de seção visual."""
    st.markdown(f"#### {title}")


def _strip_accents(text: str) -> str:
    """Remove acentos para matching de SKUs."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _filter_skus_by_color(skus: list[str], color: str) -> list[str]:
    """Filtra SKUs que contêm a cor no nome."""
    color_upper = _strip_accents(color).upper()
    return [s for s in skus if color_upper in _strip_accents(s).upper()]


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY: MARKETPLACE LISTINGS
# ═══════════════════════════════════════════════════════════════════════════════

def _show_ml_listing(ml, inputs: dict, prefix: str = "ml") -> None:
    if isinstance(ml, dict) and "raw" in ml:
        st.error("Falha ao gerar anúncio ML.")
        st.text(ml["raw"])
        return

    _section_header("Informações Principais")
    _editable_field("Título", ml.titulo, f"{prefix}_titulo")
    _editable_text_area("Descrição", ml.descricao, f"{prefix}_descricao")

    _section_header("Descoberta e Busca")
    _editable_field("Palavras-chave", ml.palavras_chave, f"{prefix}_palavras")
    _editable_field("Modelo (keywords)", ml.modelo_keywords, f"{prefix}_modelo")
    _editable_field("Esportes Recomendados", ", ".join(ml.esportes_recomendados), f"{prefix}_esportes")

    _section_header("Atributos do Produto")
    attrs = {
        "Tipo de manga": ml.tipo_manga,
        "Tipo de gola": ml.tipo_gola,
        "Estilos": ", ".join(ml.estilos) if ml.estilos else "—",
        "Forma de caimento": ml.caimento or "—",
        "Temporada de lançamento": ml.temporada or "—",
        "Marca": "Genérica",
        "Com materiais reciclados": "Não",
        "É kit": "Não",
        "Fonte do produto": "Nacional",
        "Idade": "Adultos",
        "Condição do item": "Novo",
        "É marca emergente": "Não",
    }
    for campo, valor in attrs.items():
        safe_key = campo.lower().replace(" ", "_")
        _copyable_field(campo, valor, f"{prefix}_{safe_key}")

    _section_header("Fiscal e Envio")
    fiscal_text = "\n".join(f"{k}: {v}" for k, v in {**FISCAL_INFO, **PACKAGE_INFO}.items())
    col_fiscal, col_copy = st.columns([4, 0.4])
    col_fiscal.code(fiscal_text, language=None)
    if col_copy.button("📋", key=f"cp_{prefix}_fiscal"):
        _copy_js(fiscal_text)


def _show_shopee_listing(shopee, inputs: dict, prefix: str = "shopee") -> None:
    if isinstance(shopee, dict) and "raw" in shopee:
        st.error("Falha ao gerar anúncio Shopee.")
        st.text(shopee["raw"])
        return

    _section_header("Informações Principais")
    _editable_field("Nome do Produto", shopee.nome, f"{prefix}_nome")
    _editable_field("Categoria", shopee.categoria, f"{prefix}_categoria")
    _editable_text_area("Descrição", shopee.descricao, f"{prefix}_descricao")

    _section_header("Atributos do Produto")
    attrs = {
        "Estações do ano": shopee.estacoes,
        "Comprimento parte superior": shopee.comprimento_superior,
        "Estilo": ", ".join(shopee.estilo),
        "Modelo": ", ".join(shopee.modelo),
        "Marca": "Sem marca",
        "País de origem": "Brasil",
        "Material": inputs.get("material", ""),
        "Estampa": inputs.get("desenho_tecido", ""),
        "Condição": "Novo",
        "Pequeno": "Não",
        "Plus size": "Não",
        "Produto personalizado": "Não",
        "Quantidade": "1",
        "Tamanho do pacote": "0,1 kg",
    }
    for campo, valor in attrs.items():
        safe_key = campo.lower().replace(" ", "_")
        _copyable_field(campo, valor, f"{prefix}_{safe_key}")

    _section_header("Fiscal e Envio")
    fiscal_text = "\n".join(f"{k}: {v}" for k, v in {**FISCAL_INFO, **PACKAGE_INFO}.items())
    col_fiscal, col_copy = st.columns([4, 0.4])
    col_fiscal.code(fiscal_text, language=None)
    if col_copy.button("📋", key=f"cp_{prefix}_fiscal"):
        _copy_js(fiscal_text)


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY: VARIAÇÕES (Imagens + SKUs + Preço)
# ═══════════════════════════════════════════════════════════════════════════════

def _show_variations(listing: dict, inputs: dict) -> None:
    """Exibe a tab de variações: preço, tamanhos, imagens por cor, SKUs filtrados."""
    active = listing.get("ml") or listing.get("shopee")
    if not active:
        return

    # Preço e tamanhos no topo
    _section_header("Preço e Tamanhos")
    orig = inputs.get("preco_original", 0)
    desc = inputs.get("preco_desconto", 0)
    c1, c2, c3 = st.columns(3)
    with c1:
        _copyable_field("Preço original", f"R$ {orig:.2f}", "var_preco_orig")
    with c2:
        _copyable_field("Preço com desconto", f"R$ {desc:.2f}", "var_preco_desc")
    with c3:
        tamanhos = inputs.get("tamanhos", [])
        _copyable_field("Tamanhos", ", ".join(tamanhos) if tamanhos else "—", "var_tamanhos")

    # Todas as SKUs
    skus = getattr(active, "skus", [])

    # Imagens por cor
    cores_str = inputs.get("cores", "")
    cores_list = [c.strip() for c in cores_str.split(",") if c.strip()] or [""]

    _section_header("Variações por Cor")

    for color in cores_list:
        color_key = color or "produto"
        with st.container():
            st.markdown(f"##### {color_key.capitalize()}")

            # Imagens
            color_images = st.session_state.generated_images.get(color_key, {})
            color_errors = st.session_state.image_errors.get(color_key, {})

            if color_images:
                img_cols = st.columns(len(SLOT_LABELS))
                for col, (slot, label) in zip(img_cols, SLOT_LABELS.items()):
                    with col:
                        st.caption(label)
                        path = color_images.get(slot)
                        if path and os.path.exists(path):
                            st.image(path, use_container_width=True)
                            with open(path, "rb") as f:
                                st.download_button(
                                    "📥", data=f.read(),
                                    file_name=f"{color_key}_{slot}.png",
                                    mime="image/png",
                                    key=f"dl_{color_key}_{slot}",
                                )
                            prompts = st.session_state.image_prompts.get(color_key, {})
                            if prompts and st.button("Regenerar", key=f"regen_{color_key}_{slot}"):
                                with st.spinner(f"Regenerando {label}..."):
                                    safe_c = color_key.replace(" ", "_")
                                    out = os.path.join(OUTPUT_DIR, f"{safe_c}_{slot}.png")
                                    try:
                                        new_path = generate_image(
                                            prompts[slot],
                                            st.session_state.imagens_por_cor.get(color_key, []),
                                            out,
                                        )
                                        if slot == "objecao" and new_path:
                                            base_out = out.replace(".png", "_base.png")
                                            shutil.copy2(new_path, base_out)
                                            cur_bullets = st.session_state.image_bullets.get(color_key, [])
                                            if cur_bullets:
                                                new_path = overlay_objection_text(new_path, cur_bullets, out)
                                        st.session_state.generated_images[color_key][slot] = new_path
                                        st.session_state.image_errors[color_key][slot] = None
                                    except Exception as e:
                                        st.session_state.generated_images[color_key][slot] = None
                                        st.session_state.image_errors[color_key][slot] = str(e)
                                st.rerun()
                        else:
                            err = color_errors.get(slot, "")
                            st.warning(err or "Falhou")

                # ── Bullets editáveis (objeção) ──
                objecao_path = color_images.get("objecao")
                cur_bullets = st.session_state.image_bullets.get(color_key, [])
                if objecao_path and os.path.exists(objecao_path) and cur_bullets:
                    safe_c = color_key.replace(" ", "_")
                    base_path = os.path.join(OUTPUT_DIR, f"{safe_c}_objecao_base.png")
                    if os.path.exists(base_path):
                        with st.expander("Editar textos da objeção", expanded=False):
                            padded = cur_bullets + [""] * (4 - len(cur_bullets))
                            b1 = st.text_input("Bullet 1", value=padded[0], key=f"b1_{color_key}", max_chars=30)
                            b2 = st.text_input("Bullet 2", value=padded[1], key=f"b2_{color_key}", max_chars=30)
                            b3 = st.text_input("Bullet 3", value=padded[2], key=f"b3_{color_key}", max_chars=30)
                            b4 = st.text_input("Bullet 4", value=padded[3], key=f"b4_{color_key}", max_chars=30)
                            if st.button("Aplicar texto", key=f"apply_bullets_{color_key}"):
                                new_bullets = [b for b in [b1, b2, b3, b4] if b.strip()]
                                out_path = os.path.join(OUTPUT_DIR, f"{safe_c}_objecao.png")
                                overlay_objection_text(base_path, new_bullets, out_path)
                                st.session_state.generated_images[color_key]["objecao"] = out_path
                                st.session_state.image_bullets[color_key] = new_bullets
                                st.rerun()

            # SKUs filtrados por cor
            color_skus = _filter_skus_by_color(skus, color_key)
            if color_skus:
                skus_text = "\n".join(color_skus)
                col_sku, col_copy = st.columns([4, 0.4])
                col_sku.code(skus_text, language=None)
                if col_copy.button("📋", key=f"cp_skus_{color_key}"):
                    _copy_js(skus_text)
            elif skus:
                # Fallback: mostrar todos os SKUs se filtro não encontrou
                skus_text = "\n".join(skus)
                col_sku, col_copy = st.columns([4, 0.4])
                col_sku.code(skus_text, language=None)
                if col_copy.button("📋", key=f"cp_skus_all_{color_key}"):
                    _copy_js(skus_text)

            st.divider()

    # Tabela de tamanhos
    if st.session_state.size_table_image and os.path.exists(st.session_state.size_table_image):
        st.markdown("##### Tabela de Tamanhos")
        st.image(st.session_state.size_table_image, use_container_width=False, width=600)
        with open(st.session_state.size_table_image, "rb") as f:
            st.download_button(
                "📥 Baixar Tabela",
                data=f.read(),
                file_name="tabela_tamanhos.png",
                mime="image/png",
                key="dl_tabela_var",
            )

    # Revisão da tabela extraída
    if st.session_state.size_table_data:
        with st.expander("Revisar dados da tabela de medidas"):
            table_data = st.session_state.size_table_data
            df = pd.DataFrame(table_data["linhas"], columns=table_data["colunas"])
            edited_df = st.data_editor(df, use_container_width=True, key="tabela_editor")
            if st.button("Salvar e regerar imagem", key="btn_salvar_tabela"):
                st.session_state.size_table_data = {
                    "colunas": list(edited_df.columns),
                    "linhas": edited_df.values.tolist(),
                }
                try:
                    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
                    html_content = env.get_template("tabela_tamanhos.html").render(
                        colunas=st.session_state.size_table_data["colunas"],
                        linhas=st.session_state.size_table_data["linhas"],
                    )
                    table_path = render_html_to_image(
                        html_content, os.path.join(OUTPUT_DIR, "tabela_tamanhos.png")
                    )
                    st.session_state.size_table_image = table_path
                except Exception as e:
                    st.warning(f"Erro ao regerar tabela: {e}")
                st.rerun()

    # ZIP download
    if st.session_state.generated_images:
        st.markdown("---")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for ck, ci in st.session_state.generated_images.items():
                safe_c = ck.replace(" ", "_")
                for slot, zip_name in SLOT_ZIP_NAMES.items():
                    path = ci.get(slot)
                    if path and os.path.exists(path):
                        zf.write(path, f"{safe_c}/{zip_name}")
            tabela_img = st.session_state.size_table_image
            if tabela_img and os.path.exists(tabela_img):
                zf.write(tabela_img, "tabela_tamanhos.png")
        zip_buffer.seek(0)
        st.download_button(
            label="Baixar Todas as Imagens (.zip)",
            data=zip_buffer,
            file_name="imagens_anuncio.zip",
            mime="application/zip",
            key="dl_zip_var",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ONE-SHOT
# ═══════════════════════════════════════════════════════════════════════════════

def _run_full_pipeline(
    product_data: dict,
    marketplaces: list[str],
    imagens_por_cor: dict[str, list[str]],
    cores_list: list[str],
    tabela_b64: str | None,
) -> None:
    """Executa listing + tabela + imagens em uma única ação."""
    status = st.status("Gerando anúncio completo...", expanded=True)

    # ── FASE 1: Listing + Tabela de medidas (paralelo) ──
    with status:
        st.write("Gerando texto do anúncio...")

    listing_result = {}
    listing_error = ""
    size_table_result = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        _LISTING_KEYS = {"tipo_peca", "genero", "material", "desenho_tecido", "cores", "tamanhos", "sku_base", "preco_original", "preco_desconto"}
        todas_imagens = [img for imgs in imagens_por_cor.values() for img in imgs]
        future_listing = executor.submit(
            ListingPipeline().run,
            marketplaces=marketplaces,
            imagens=todas_imagens,
            **{k: v for k, v in product_data.items() if k in _LISTING_KEYS},
        )
        future_table = None
        if tabela_b64:
            future_table = executor.submit(extract_size_table, tabela_b64)

        try:
            listing_result = future_listing.result()
        except Exception as e:
            listing_error = str(e)

        if future_table:
            size_table_result = future_table.result()

    st.session_state.listing = listing_result
    st.session_state.listing_error = listing_error
    if size_table_result:
        st.session_state.size_table_data = size_table_result

    with status:
        if listing_error:
            st.write(f"✗ Falha no anúncio: {listing_error}")
        else:
            st.write("✓ Texto do anúncio gerado")
        if tabela_b64:
            st.write("✓ Tabela de medidas extraída" if size_table_result else "✗ Falha na extração da tabela")

    if listing_error:
        status.update(label="Falha na geração", state="error", expanded=True)
        return

    # ── FASE 2: Prompts de imagem por cor ──
    if not any(imagens_por_cor.values()):
        with status:
            st.write("Sem imagens de referência — pulando geração de imagens.")
        status.update(label="Concluído (sem imagens)", state="complete", expanded=False)
        return

    active_listing = listing_result.get("ml") or listing_result.get("shopee")
    listing_titulo = getattr(active_listing, "titulo", getattr(active_listing, "nome", ""))
    listing_descricao = getattr(active_listing, "descricao", "") if active_listing else ""

    agent = ImageGeneratorAgent()
    slots = list(SLOT_LABELS.keys())
    all_prompts: dict = {}
    all_images: dict = {}
    all_errors: dict = {}
    all_bullets: dict = {}

    for color in cores_list:
        color_key = color or "produto"
        with status:
            st.write(f"Gerando prompts para {color_key}...")

        prompts_result = agent.run({
            **product_data,
            "imagens": imagens_por_cor.get(color, []),
            "color": color,
            "titulo": listing_titulo,
            "descricao": listing_descricao,
        })

        if "raw" in prompts_result:
            all_images[color_key] = {s: None for s in slots}
            all_errors[color_key] = {s: "Falha na geração de prompts" for s in slots}
            with status:
                st.write(f"✗ Prompts para {color_key} falharam")
            continue

        all_prompts[color_key] = prompts_result
        all_images[color_key] = {}
        all_errors[color_key] = {}

    # ── FASE 3: Geração de imagens (paralelo, max 4 workers) ──
    tasks = []
    for color_key, prompts in all_prompts.items():
        for slot in slots:
            tasks.append((color_key, slot, prompts))

    total = len(tasks)
    if total > 0:
        completed_count = 0
        with status:
            st.write(f"Gerando {total} imagens...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {}
            for color_key, slot, prompts in tasks:
                safe_color = color_key.replace(" ", "_")
                output_path = os.path.join(OUTPUT_DIR, f"{safe_color}_{slot}.png")
                future = executor.submit(generate_image, prompts[slot], imagens_por_cor.get(color_key, []), output_path)
                future_map[future] = (color_key, slot, prompts, output_path)

            for future in as_completed(future_map):
                color_key, slot, prompts, output_path = future_map[future]
                completed_count += 1

                try:
                    path = future.result()
                    if slot == "objecao" and path:
                        # Salvar imagem base (sem overlay) para re-render posterior
                        base_path = output_path.replace(".png", "_base.png")
                        shutil.copy2(path, base_path)

                        bullets = [prompts.get(f"bullet{k}", "") for k in range(1, 5)]
                        bullets = [b for b in bullets if b]
                        all_bullets[color_key] = bullets
                        if bullets:
                            path = overlay_objection_text(path, bullets, output_path)
                    all_images[color_key][slot] = path
                    all_errors[color_key][slot] = None
                except Exception as e:
                    all_images[color_key][slot] = None
                    all_errors[color_key][slot] = str(e)

                with status:
                    ok = "✓" if all_images[color_key][slot] else "✗"
                    st.write(f"Imagem {completed_count}/{total}: {color_key}/{SLOT_LABELS[slot]} {ok}")

    # ── FASE 4: Render tabela de tamanhos ──
    if st.session_state.size_table_data:
        with status:
            st.write("Gerando imagem da tabela de tamanhos...")
        try:
            env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
            table_data = st.session_state.size_table_data
            html_content = env.get_template("tabela_tamanhos.html").render(
                colunas=table_data["colunas"],
                linhas=table_data["linhas"],
            )
            table_path = render_html_to_image(
                html_content, os.path.join(OUTPUT_DIR, "tabela_tamanhos.png")
            )
            st.session_state.size_table_image = table_path
            with status:
                st.write("✓ Tabela de tamanhos renderizada")
        except Exception as e:
            with status:
                st.write(f"✗ Tabela de tamanhos: {e}")

    # ── Salvar resultados ──
    st.session_state.image_prompts = all_prompts
    st.session_state.generated_images = all_images
    st.session_state.image_errors = all_errors
    st.session_state.image_bullets = all_bullets

    has_errors = any(
        err for color_errs in all_errors.values() for err in color_errs.values() if err
    )
    label = "Concluído" + (" (com erros em algumas imagens)" if has_errors else "")
    status.update(label=label, state="complete" if not has_errors else "error", expanded=False)


# ═══════════════════════════════════════════════════════════════════════════════
# DADOS DO PRODUTO (compartilhados entre tabs)
# ═══════════════════════════════════════════════════════════════════════════════
_section_header("Produto")
col1, col2, col3, col4 = st.columns(4)
with col1:
    tipo_peca = st.text_input("Tipo de peça", placeholder="cropped, legging, camiseta...", help="Influencia o título e a descrição do anúncio")
with col2:
    genero = st.selectbox("Gênero", options=["Feminino", "Masculino", "Unissex"], help="Define o gênero da modelo nas imagens geradas")
with col3:
    material = st.text_input("Material", placeholder="poliamida, algodão, dry fit...", help="Aparece na descrição e nos atributos do produto")
with col4:
    desenho_tecido = st.text_input("Desenho do tecido", placeholder="liso, listras, floral...", help="Padrão visual do tecido (liso, listrado, floral...)")

col5, col6, col7, col8 = st.columns(4)
with col5:
    tamanhos = st.multiselect("Tamanhos", options=["PP", "P", "M", "G", "GG", "XGG"], help="Gera um SKU para cada combinação cor x tamanho")
with col6:
    sku_base = st.text_input("SKU base", placeholder="ex: CROPPED01", help="Prefixo dos SKUs gerados (ex: CROPPED01-PRETO-P)")
with col7:
    preco_original = st.number_input("Preço original (R$)", min_value=0.0, step=0.01, format="%.2f", help="Preço cheio antes do desconto")
with col8:
    preco_desconto = st.number_input("Preço com desconto (R$)", min_value=0.0, step=0.01, format="%.2f", help="Preço final que o comprador paga")

marketplaces = st.multiselect(
    "Marketplaces",
    options=["Mercado Livre", "Shopee"],
    default=["Mercado Livre"],
)

product_data = {
    "tipo_peca": tipo_peca,
    "genero": genero,
    "material": material,
    "desenho_tecido": desenho_tecido,
    "tamanhos": tamanhos,
    "sku_base": sku_base,
    "preco_original": preco_original,
    "preco_desconto": preco_desconto,
}

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPAIS
# ═══════════════════════════════════════════════════════════════════════════════
tab_simple, tab_kit = st.tabs(["Peça Simples", "Kit"])


# ═══════════════════════════════════════════════════════════════════════════════
# ABA: PEÇA SIMPLES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_simple:

    # ── Cores e Imagens ──
    _section_header("Cores e Imagens")
    cores_selecionadas = st.multiselect("Cores disponíveis", options=CORES_OPCOES, help="Uma variação de imagens + SKUs será gerada para cada cor")
    cores = ", ".join(cores_selecionadas)

    if cores_selecionadas:
        st.caption("Envie as imagens de referência para cada cor — o modelo seguirá a coloração exata")
        imagens_upload_por_cor: dict[str, list] = {}
        for _cor in cores_selecionadas:
            imagens_upload_por_cor[_cor] = st.file_uploader(
                f"Imagens — {_cor.capitalize()}",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=f"img_upload_{_cor}",
            ) or []
    else:
        imagens_upload_por_cor = {}

    # ── Estilo das Imagens ──
    _section_header("Estilo das Imagens")
    st.caption("Controle a aparência das fotos geradas")
    col_bg, col_life, col_pose = st.columns(3)
    with col_bg:
        fundo_capa = st.selectbox(
            "Fundo da capa",
            ["Branco puro", "Cinza claro", "Gradiente suave"],
            help="Cor de fundo da foto principal do anúncio",
        )
    with col_life:
        cenario_lifestyle = st.selectbox(
            "Cenário lifestyle",
            ["Urbano", "Praia", "Academia", "Casa/Cozy", "Natureza", "Estúdio"],
            help="Ambiente da foto de estilo de vida",
        )
    with col_pose:
        estilo_pose = st.selectbox(
            "Estilo da pose",
            ["Editorial/Fashion", "Natural/Casual", "Esportivo/Movimento"],
            help="Postura e energia da modelo nas fotos",
        )

    # ── Tabela de Medidas (opcional) ──
    _section_header("Tabela de Medidas")
    st.caption("Opcional — será extraída automaticamente ao gerar o anúncio")
    tabela_upload = st.file_uploader(
        "Imagem da tabela de medidas",
        type=["png", "jpg", "jpeg"],
        key="tabela_upload_simple",
        label_visibility="collapsed",
    )

    # ── Preview do que será gerado ──
    st.markdown("---")
    n_cores = len(cores_selecionadas) or 1
    n_imagens = n_cores * 4
    has_tabela = tabela_upload is not None
    mp_names = ", ".join(marketplaces) if marketplaces else "nenhum selecionado"
    st.caption(
        f"Será gerado: anúncio para **{mp_names}** · "
        f"**{n_imagens}** imagens ({n_cores} cor(es) × 4 tipos)"
        + (" · tabela de medidas" if has_tabela else ""),
        unsafe_allow_html=False,
    )

    # ── Botão único ──
    gerar = st.button(
        "Gerar Anúncio Completo",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.gen_running,
    )

    if gerar:
        if not marketplaces:
            st.error("Selecione ao menos um marketplace.")
            st.stop()

        st.session_state.gen_running = True
        st.session_state.listing = {}
        st.session_state.listing_error = ""
        st.session_state.image_prompts = {}
        st.session_state.generated_images = {}
        st.session_state.image_errors = {}
        st.session_state.image_bullets = {}
        st.session_state.size_table_data = None
        st.session_state.size_table_image = None

        imagens_por_cor: dict[str, list[str]] = {
            cor: [base64.b64encode(f.read()).decode("utf-8") for f in arquivos]
            for cor, arquivos in imagens_upload_por_cor.items()
        }
        simple_product_data = {
            **product_data,
            "cores": cores,
            "img_fundo": fundo_capa,
            "img_cenario": cenario_lifestyle,
            "img_pose": estilo_pose,
        }

        tabela_b64 = None
        if tabela_upload:
            tabela_upload.seek(0)
            tabela_b64 = base64.b64encode(tabela_upload.read()).decode("utf-8")

        cores_list = [c.strip() for c in cores.split(",") if c.strip()] or [""]

        st.session_state.imagens_por_cor = imagens_por_cor
        st.session_state.marketplaces = marketplaces
        st.session_state.inputs = simple_product_data

        _run_full_pipeline(simple_product_data, marketplaces, imagens_por_cor, cores_list, tabela_b64)

        st.session_state.gen_running = False
        st.rerun()

    # ── Resultados ──
    if st.session_state.listing_error and not st.session_state.listing:
        st.error(f"Falha ao gerar anúncio: {st.session_state.listing_error}")
        if st.button("Tentar Novamente", type="primary", key="btn_retry_listing"):
            st.session_state.listing_error = ""
            st.rerun()

    if st.session_state.listing:
        listing = st.session_state.listing
        inputs = st.session_state.inputs
        selected = st.session_state.marketplaces

        # Montar tabs de resultado
        result_tab_names = []
        if "Mercado Livre" in selected and listing.get("ml"):
            result_tab_names.append("Mercado Livre")
        if "Shopee" in selected and listing.get("shopee"):
            result_tab_names.append("Shopee")
        if st.session_state.generated_images or st.session_state.image_errors:
            result_tab_names.append("Variações")
        elif any(st.session_state.imagens_por_cor.values()):
            result_tab_names.append("Variações")

        if result_tab_names:
            st.divider()
            result_tabs = st.tabs(result_tab_names)
            result_tab_map = dict(zip(result_tab_names, result_tabs))

            if "Mercado Livre" in result_tab_map:
                with result_tab_map["Mercado Livre"]:
                    _show_ml_listing(listing.get("ml"), inputs)

            if "Shopee" in result_tab_map:
                with result_tab_map["Shopee"]:
                    _show_shopee_listing(listing.get("shopee"), inputs)

            if "Variações" in result_tab_map:
                with result_tab_map["Variações"]:
                    _show_variations(listing, inputs)


# ═══════════════════════════════════════════════════════════════════════════════
# ABA: KIT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_kit:
    st.subheader("Variantes de Cor")
    st.caption("Preencha pelo menos 2 variantes com imagem e nome da cor.")

    kit_cols = st.columns(4)
    raw_variants = []
    for i, col in enumerate(kit_cols):
        with col:
            suffix = " (opcional)" if i >= 2 else ""
            st.markdown(f"**Cor {i + 1}{suffix}**")
            img_file = st.file_uploader(
                "Imagem",
                type=["png", "jpg", "jpeg"],
                key=f"kit_img_{i}",
                label_visibility="collapsed",
            )
            color_name = st.text_input(
                "Nome da cor",
                placeholder="ex: Azul, Rosa...",
                key=f"kit_color_{i}",
                label_visibility="collapsed",
            )
            raw_variants.append({"img_file": img_file, "color": color_name})

    valid_indices = [
        i for i, v in enumerate(raw_variants)
        if v["img_file"] is not None and v["color"].strip()
    ]

    selected_combos: list[tuple[int, int]] = []
    if len(valid_indices) >= 2:
        st.subheader("Selecione as Combinações")
        for i in range(len(valid_indices)):
            for j in range(i + 1, len(valid_indices)):
                va_idx = valid_indices[i]
                vb_idx = valid_indices[j]
                ca = raw_variants[va_idx]["color"]
                cb = raw_variants[vb_idx]["color"]
                if st.checkbox(f"{ca} + {cb}", value=True, key=f"kit_combo_{va_idx}_{vb_idx}"):
                    selected_combos.append((va_idx, vb_idx))

    gerar_kit = st.button(
        "Gerar Kit",
        type="primary",
        use_container_width=True,
        disabled=len(selected_combos) == 0,
        key="btn_gerar_kit",
    )

    if gerar_kit and selected_combos:
        if not marketplaces:
            st.error("Selecione ao menos um marketplace.")
        else:
            processed: dict[int, dict] = {}
            for idx in valid_indices:
                v = raw_variants[idx]
                v["img_file"].seek(0)
                b64 = base64.b64encode(v["img_file"].read()).decode("utf-8")
                processed[idx] = {"color": v["color"].strip(), "images_b64": [b64]}

            combos_to_run = [
                (processed[va_idx], processed[vb_idx])
                for va_idx, vb_idx in selected_combos
            ]

            pipeline = KitPipeline()

            kit_status = st.status(
                f"Gerando {len(combos_to_run)} combinação(ões)...", expanded=True
            )

            kit_results: dict = {}
            with ThreadPoolExecutor(max_workers=len(combos_to_run)) as executor:
                future_to_name = {
                    executor.submit(
                        pipeline.run_combination,
                        v_a["images_b64"],
                        v_b["images_b64"],
                        v_a["color"],
                        v_b["color"],
                        product_data,
                        marketplaces,
                        OUTPUT_DIR,
                    ): f"{v_a['color']} + {v_b['color']}"
                    for v_a, v_b in combos_to_run
                }
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        kit_results[name] = future.result()
                        with kit_status:
                            st.write(f"✓ {name}")
                    except Exception as e:
                        kit_results[name] = {"error": str(e)}
                        with kit_status:
                            st.write(f"✗ {name}: {e}")

            has_kit_errors = any("error" in r for r in kit_results.values())
            kit_status.update(
                label="Kits gerados" + (" (com erros)" if has_kit_errors else ""),
                state="complete" if not has_kit_errors else "error",
                expanded=False,
            )
            st.session_state.kit_results = kit_results
            st.rerun()

    # ── Resultados do Kit ──
    if st.session_state.kit_results:
        for combo_name, result in st.session_state.kit_results.items():
            with st.expander(f"Kit: {combo_name}", expanded=True):
                if "error" in result:
                    st.error(f"Erro: {result['error']}")
                    if st.button("Tentar Novamente", key=f"btn_retry_kit_{combo_name}"):
                        del st.session_state.kit_results[combo_name]
                        st.rerun()
                    continue

                images = result.get("images", {})
                color_a = result.get("color_a", "A")
                color_b = result.get("color_b", "B")
                slot_labels = {
                    "capa_kit": "Capa do Kit",
                    "detalhes_a": f"Detalhes {color_a}",
                    "detalhes_b": f"Detalhes {color_b}",
                    "objecao": "Quebra de Objeção",
                    "lifestyle_kit": "Lifestyle",
                }
                img_cols = st.columns(len(slot_labels))
                for col, (slot, label) in zip(img_cols, slot_labels.items()):
                    with col:
                        st.markdown(f"**{label}**")
                        path = images.get(slot)
                        if path and os.path.exists(path):
                            st.image(path, use_container_width=True)
                            with open(path, "rb") as f:
                                st.download_button(
                                    "📥",
                                    data=f.read(),
                                    file_name=f"{combo_name.replace(' ', '_')}_{slot}.png",
                                    mime="image/png",
                                    key=f"kit_dl_{combo_name}_{slot}",
                                )
                        else:
                            st.warning("Falhou")

                valid_paths = [
                    (slot, images[slot])
                    for slot in slot_labels
                    if images.get(slot) and os.path.exists(images.get(slot, ""))
                ]
                if valid_paths:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for i, (slot, path) in enumerate(valid_paths):
                            zf.write(path, f"{i + 1:02d}_{slot}.png")
                    zip_buf.seek(0)
                    st.download_button(
                        "Baixar Todas (.zip)",
                        data=zip_buf,
                        file_name=f"kit_{combo_name.replace(' + ', '_')}.zip",
                        mime="application/zip",
                        key=f"kit_zip_{combo_name}",
                    )

                listings = result.get("listings", {})
                LISTING_KEY_TO_LABEL = {"ml": "Mercado Livre", "shopee": "Shopee"}
                listing_tab_names = [LISTING_KEY_TO_LABEL[k] for k in ["ml", "shopee"] if k in listings]
                if listing_tab_names:
                    st.divider()
                    kit_listing_tabs = st.tabs(listing_tab_names)
                    kit_tab_map = dict(zip(listing_tab_names, kit_listing_tabs))

                    kit_inputs = {**product_data, "preco_original": preco_original, "preco_desconto": preco_desconto}

                    safe_combo = combo_name.replace(" ", "_").replace("+", "")
                    if "Mercado Livre" in kit_tab_map:
                        with kit_tab_map["Mercado Livre"]:
                            _show_ml_listing(listings.get("ml"), kit_inputs, prefix=f"kit_ml_{safe_combo}")

                    if "Shopee" in kit_tab_map:
                        with kit_tab_map["Shopee"]:
                            _show_shopee_listing(listings.get("shopee"), kit_inputs, prefix=f"kit_shopee_{safe_combo}")

        # Tabela de tamanhos do Kit
        if st.session_state.size_table_data:
            if not st.session_state.size_table_image:
                with st.spinner("Gerando imagem da tabela de tamanhos..."):
                    try:
                        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
                        table_data = st.session_state.size_table_data
                        html_content = env.get_template("tabela_tamanhos.html").render(
                            colunas=table_data["colunas"],
                            linhas=table_data["linhas"],
                        )
                        table_path = render_html_to_image(
                            html_content, os.path.join(OUTPUT_DIR, "tabela_tamanhos.png")
                        )
                        st.session_state.size_table_image = table_path
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Não foi possível gerar a tabela de tamanhos: {e}")

            if st.session_state.size_table_image and os.path.exists(st.session_state.size_table_image):
                st.divider()
                st.subheader("Tabela de Tamanhos")
                st.image(st.session_state.size_table_image, use_container_width=False, width=600)
                with open(st.session_state.size_table_image, "rb") as f:
                    st.download_button(
                        "📥 Baixar Tabela",
                        data=f.read(),
                        file_name="tabela_tamanhos.png",
                        mime="image/png",
                        key="kit_dl_tabela",
                    )
