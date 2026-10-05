#!/usr/bin/env python3
"""Regenerate the course-homepage QR assets. Requires qrcode[pil]."""
from pathlib import Path
import qrcode
import qrcode.image.svg

URL = 'https://yaolangzhong.github.io/U_Tokyo_Comp_Econ_Course/'
ROOT = Path(__file__).resolve().parents[1]
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=12, border=4)
qr.add_data(URL)
qr.make(fit=True)
out = ROOT / 'assets'
out.mkdir(exist_ok=True)
qr.make_image(fill_color='black', back_color='white').save(out / 'course-qr.png')
qr.make_image(image_factory=qrcode.image.svg.SvgPathImage).save(out / 'course-qr.svg')
print(URL)
