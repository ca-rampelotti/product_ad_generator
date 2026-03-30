import os
import re

from PIL import Image, ImageDraw, ImageFont

HEADER_BG = (107, 30, 53)       # #6b1e35
HEADER_FG = (255, 255, 255)
HEADER_BORDER = (90, 26, 45)
ROW_ODD_BG = (255, 255, 255)
ROW_EVEN_BG = (247, 247, 247)
ROW_FG = (45, 45, 45)
BORDER_COLOR = (224, 224, 224)
TITLE_COLOR = (26, 26, 26)
FOOTNOTE_COLOR = (170, 170, 170)

_FONT_CANDIDATES = [
    "assets/fonts/Montserrat-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
    return ImageFont.load_default()


def _parse_table_from_html(html: str) -> tuple[list[str], list[list[str]]]:
    headers = re.findall(r"<th[^>]*>(.*?)</th>", html, re.DOTALL)
    headers = [h.strip() for h in headers]
    rows = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
        if cells:
            rows.append([c.strip() for c in cells])
    return headers, rows


def render_html_to_image(
    html_content: str,
    output_path: str,
    width: int = 1200,
    height: int = 1200,
) -> str:
    headers, rows = _parse_table_from_html(html_content)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(90)
    header_font = _load_font(56)
    cell_font = _load_font(52)
    footnote_font = _load_font(34)

    # Title
    title = "GUIA DE TAMANHOS"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    title_x = (width - title_w) // 2
    title_y = 80
    draw.text((title_x, title_y), title, fill=TITLE_COLOR, font=title_font)

    # Accent line under title
    accent_y = title_y + title_h + 20
    accent_w = 72
    draw.rectangle(
        [(width - accent_w) // 2, accent_y, (width + accent_w) // 2, accent_y + 4],
        fill=HEADER_BG,
    )

    if not headers:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.save(output_path, "PNG")
        return output_path

    # Table layout
    margin_x = 120
    table_w = width - 2 * margin_x
    num_cols = len(headers)
    col_w = table_w // num_cols
    header_h = 120
    row_h = 108
    table_y = accent_y + 52

    # Header row
    for i, header in enumerate(headers):
        x0 = margin_x + i * col_w
        x1 = x0 + col_w
        y0 = table_y
        y1 = y0 + header_h
        draw.rectangle([x0, y0, x1, y1], fill=HEADER_BG)
        draw.rectangle([x0, y0, x1, y1], outline=HEADER_BORDER, width=1)
        bbox = draw.textbbox((0, 0), header, font=header_font)
        text_x = x0 + (col_w - (bbox[2] - bbox[0])) // 2
        text_y = y0 + (header_h - (bbox[3] - bbox[1])) // 2
        draw.text((text_x, text_y), header, fill=HEADER_FG, font=header_font)

    # Data rows
    for r, row in enumerate(rows):
        row_bg = ROW_ODD_BG if r % 2 == 0 else ROW_EVEN_BG
        y0 = table_y + header_h + r * row_h
        y1 = y0 + row_h
        for c, cell in enumerate(row[:num_cols]):
            x0 = margin_x + c * col_w
            x1 = x0 + col_w
            draw.rectangle([x0, y0, x1, y1], fill=row_bg)
            draw.rectangle([x0, y0, x1, y1], outline=BORDER_COLOR, width=1)
            bbox = draw.textbbox((0, 0), cell, font=cell_font)
            text_x = x0 + (col_w - (bbox[2] - bbox[0])) // 2
            text_y = y0 + (row_h - (bbox[3] - bbox[1])) // 2
            draw.text((text_x, text_y), cell, fill=ROW_FG, font=cell_font)

    # Footnote
    footnote = "* Medidas podem variar \u00b12cm"
    footnote_y = table_y + header_h + len(rows) * row_h + 30
    bbox = draw.textbbox((0, 0), footnote, font=footnote_font)
    footnote_x = (width - (bbox[2] - bbox[0])) // 2
    draw.text((footnote_x, footnote_y), footnote, fill=FOOTNOTE_COLOR, font=footnote_font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, "PNG")
    return output_path
