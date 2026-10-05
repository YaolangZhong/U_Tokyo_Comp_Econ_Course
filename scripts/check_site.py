#!/usr/bin/env python3
"""Check rendered local links and enforce the public site's file boundary."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys

ROOT = Path(__file__).resolve().parents[1] / '_site'
PREFIX = '/U_Tokyo_Comp_Econ_Course/'

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                self.links.append(value)


def main():
    errors = []
    pages = list(ROOT.rglob('*.html'))
    if not pages:
        errors.append('No rendered pages found.')
    for page in pages:
        parser = Links()
        parser.feed(page.read_text())
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            path = unquote(url.path)
            if path.startswith(PREFIX):
                target = ROOT / path[len(PREFIX):]
            elif path.startswith('/'):
                target = ROOT / path.lstrip('/')
            else:
                target = page.parent / path
            if not target.exists():
                errors.append(f'{page.relative_to(ROOT)}: missing {link}')
    for forbidden in ['archive', 'build', 'Final_Project/Paper_List', '.git', 'scripts', 'docs']:
        if (ROOT / forbidden).exists():
            errors.append(f'Non-website directory published: {forbidden}')
    for n in range(1, 14):
        if not (ROOT / f'weeks/week_{n:02}/index.html').is_file():
            errors.append(f'Missing week {n}')
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f'PASS: {len(pages)} HTML pages, all local file links, 13 weekly pages, publication boundary.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
