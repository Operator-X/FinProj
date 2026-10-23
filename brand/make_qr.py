"""Poster QR code for Margem: rounded modules, Capi in the middle, handwritten caption.

Usage: uv run --with "qrcode[pil]" python brand/make_qr.py <Caveat.ttf>
Error correction H (30%) keeps the code scannable with Capi covering the centre.
"""
import sys
from pathlib import Path

import qrcode
from PIL import Image, ImageDraw, ImageFont
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.colormasks import SolidFillColorMask
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer

URL = "https://operator-x.github.io/margem/"
OUT = Path(__file__).parent / "qr"
STYLES = {  # name: (page background, caption ink); the code itself is always dark-on-light so every phone reads it
    "margem-qr-light": ((255, 255, 255), (168, 115, 10)),
    "margem-qr-dark": ((11, 12, 14), (242, 211, 123)),
}
CODE_BG, CODE_FG = (250, 249, 246), (24, 22, 20)


def make(name: str, page, ink, font_path: str) -> Path:
    bg, fg, variant = CODE_BG, CODE_FG, "for-light-backgrounds"
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=28, border=4)  # 4-module quiet zone per the QR spec
    qr.add_data(URL)
    qr.make(fit=True)
    img = qr.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer(radius_ratio=1),
                        color_mask=SolidFillColorMask(back_color=bg, front_color=fg)).convert("RGBA")
    w = img.size[0]
    # Capi's head on a rounded badge in the centre (~24% of the width, well inside the 30% budget).
    badge = int(w * 0.24)
    logo = Image.open(Path(__file__).parent / "capi" / variant / "capi-logo.png").convert("RGBA").resize((badge, badge))
    pad = Image.new("RGBA", (badge + 24, badge + 24), (0, 0, 0, 0))
    ImageDraw.Draw(pad).rounded_rectangle([0, 0, badge + 23, badge + 23], radius=badge // 4, fill=bg + (255,))
    pad.alpha_composite(logo, (12, 12))
    img.alpha_composite(pad, ((w - pad.size[0]) // 2, (w - pad.size[1]) // 2))
    # Handwritten caption underneath.
    font = ImageFont.truetype(font_path, int(w * 0.075))
    font.set_variation_by_axes([700])
    cap_h, m = int(w * 0.16), int(w * 0.06) if page != CODE_BG else 0
    canvas = Image.new("RGBA", (w + 2 * m, w + cap_h + 2 * m), page + (255,))
    if m:  # dark page: the code sits on a rounded light card
        card = Image.new("L", img.size, 0)
        ImageDraw.Draw(card).rounded_rectangle([0, 0, w - 1, w - 1], radius=w // 14, fill=255)
        img.putalpha(card)
    canvas.alpha_composite(img, (m, m))
    d = ImageDraw.Draw(canvas)
    text = "scan me · meet Capi"
    tw = d.textlength(text, font=font)
    d.text(((w + 2 * m - tw) / 2, w + m + (int(w * 0.02) if m else -int(w * 0.03))), text, font=font, fill=ink)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.png"
    canvas.convert("RGB").save(path, dpi=(300, 300))
    return path


if __name__ == "__main__":
    for name, (page, ink) in STYLES.items():
        print(make(name, page, ink, sys.argv[1]))
