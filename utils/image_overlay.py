import os

from PIL import Image, ImageDraw, ImageFont

ACCENT_COLOR = (0, 210, 255, 255)   # electric cyan
TEXT_COLOR = (255, 255, 255, 255)
PANEL_COLOR = (10, 10, 20)           # near-black with blue tint

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


def _gradient_band(width: int, band_height: int) -> Image.Image:
    band = Image.new("RGBA", (width, band_height))
    draw = ImageDraw.Draw(band)
    for y in range(band_height):
        # Fade from transparent at top to solid near-black at bottom
        t = (y / band_height) ** 0.55
        alpha = int(230 * t)
        r, g, b = PANEL_COLOR
        draw.line([(0, y), (width - 1, y)], fill=(r, g, b, alpha))
    return band


def _accent_line(width: int, height: int = 3) -> Image.Image:
    line = Image.new("RGBA", (width, height), ACCENT_COLOR)
    return line


def overlay_objection_text(image_path: str, bullets: list[str], output_path: str) -> str:
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    band_height = int(h * 0.48)
    band_top = h - band_height

    # Gradient overlay
    band = _gradient_band(w, band_height)
    img.paste(band, (0, band_top), band)

    # Thin accent line at top of the band
    accent = _accent_line(w, max(3, int(h * 0.003)))
    img.paste(accent, (0, band_top), accent)

    font_size = max(52, int(w / 15))
    font = _load_font(font_size)
    check_font = _load_font(int(font_size * 1.15))

    draw = ImageDraw.Draw(img)

    bullets = bullets[:4]
    cols = 2
    rows = -(-len(bullets) // cols)  # ceiling division

    padding_x = int(w * 0.05)
    col_width = (w - padding_x * 2) // cols
    line_height = font_size + int(font_size * 0.55)
    total_h = rows * line_height
    y_start = band_top + int(band_height * 0.30) + (band_height - int(band_height * 0.30) - total_h) // 2

    for i, bullet in enumerate(bullets):
        col = i % cols
        row = i // cols
        x = padding_x + col * col_width
        y = y_start + row * line_height

        # Checkmark in accent color
        draw.text((x, y), "✓", fill=ACCENT_COLOR, font=check_font)
        check_w = int(check_font.getlength("✓  "))

        # Bullet text in white
        draw.text((x + check_w, y), bullet, fill=TEXT_COLOR, font=font)

    result = img.convert("RGB")
    result.save(output_path, "PNG")
    return output_path
