import os


def render_html_to_image(
    html_content: str,
    output_path: str,
    width: int = 1200,
    height: int = 1200,
) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright não está instalado. Execute: playwright install chromium"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(html_content, wait_until="networkidle")
        page.screenshot(
            path=output_path,
            clip={"x": 0, "y": 0, "width": width, "height": height},
        )
        browser.close()

    return output_path
