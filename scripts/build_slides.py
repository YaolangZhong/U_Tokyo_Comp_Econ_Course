#!/usr/bin/env python3
"""Build course slide decks and publish only successful PDFs (Python standard library)."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def select_sources(args):
    if args.week is not None:
        if not 1 <= args.week <= 13:
            raise ValueError('Week must be between 1 and 13.')
        parts = [args.part] if args.part else ['lecture', 'lab']
        folders = [ROOT / f'weeks/week_{args.week:02}' / part / 'slides/tex' for part in parts]
    elif args.all:
        folders = sorted(ROOT.glob('weeks/week_*/[!_]*/slides/tex')) + sorted(ROOT.glob('topics/*/slides/tex'))
    else:
        source = Path(args.source).expanduser()
        if source.suffix != '.tex':
            source = source.with_suffix('.tex')
        if not source.is_absolute():
            source = source if source.is_file() else ROOT / source
        return [validate_source(source)]
    sources = []
    for folder in folders:
        decks = [p for p in sorted(folder.glob('*.tex'))
                 if re.search(r'\\documentclass\b', re.sub(r'(?<!\\)%[^\n]*', '', p.read_text()))]
        if not decks:
            print(f'SKIP {folder.relative_to(ROOT)}: no standalone TeX deck', flush=True)
        sources.extend(validate_source(p) for p in decks)
    return sources


def validate_source(source):
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f'Source not found: {source}')
    try:
        rel = source.relative_to(ROOT)
    except ValueError:
        raise ValueError('Source must be inside this course repository.')
    if rel.parts[0] not in ('weeks', 'topics') or source.parent.name != 'tex' or source.parent.parent.name != 'slides':
        raise ValueError('Source must be in weeks/.../slides/tex/ or topics/.../slides/tex/.')
    return source


def build(source, latexmk):
    rel = source.relative_to(ROOT)
    out = ROOT / 'build' / rel.parent.parent.parent / source.stem
    published = source.parent.parent / 'pdf' / (source.stem + '.pdf')
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Root-relative shared assets remain compatible; local inputs are also searchable.
    for var in ('TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS'):
        env[var] = str(source.parent) + '//' + os.pathsep + env.get(var, '')
    print(f'BUILD {rel}', flush=True)
    result = subprocess.run([latexmk, '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                             '-file-line-error', '-synctex=0', f'-outdir={out}', str(source)],
                            cwd=ROOT, env=env)
    product = out / published.name
    if result.returncode or not product.is_file() or product.stat().st_size == 0:
        print(f'FAILED {rel}; previous handout preserved. Logs: {out}', file=sys.stderr)
        return False
    published.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a partial published PDF if copying is interrupted.
    with tempfile.NamedTemporaryFile(dir=published.parent, suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(product, temporary)
        os.replace(temporary, published)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'PDF {published.relative_to(ROOT)}', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('source', nargs='?', help='one .tex source; relative to cwd or repository root')
    group.add_argument('--week', type=int, help='build available lecture and lab decks for week 1–13')
    group.add_argument('--all', action='store_true', help='build all available weekly and topic decks')
    parser.add_argument('--part', choices=['lecture', 'lab'], help='restrict --week to lecture or lab')
    parser.add_argument('--list', action='store_true', help='show selected decks without compiling')
    args = parser.parse_args()
    if args.part and args.week is None:
        parser.error('--part requires --week')
    try:
        sources = select_sources(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not sources:
        print('No decks available; nothing compiled or published.')
        return 0
    if args.list:
        for source in sources:
            print(source.relative_to(ROOT))
        return 0
    latexmk = shutil.which('latexmk')
    if not latexmk and Path('/Library/TeX/texbin/latexmk').is_file():
        latexmk = '/Library/TeX/texbin/latexmk'
        os.environ['PATH'] = '/Library/TeX/texbin' + os.pathsep + os.environ.get('PATH', '')
    if not latexmk:
        print('latexmk not found. Use a TeX installation containing latexmk and pdflatex; no PDFs changed.', file=sys.stderr)
        return 127
    results = [build(source, latexmk) for source in sources]
    print(f'{sum(results)}/{len(results)} decks built successfully.')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
