#!/usr/bin/env python3
"""Generate download bundles and material links from existing weekly files."""
from pathlib import Path
import shutil
from urllib.parse import quote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / 'downloads'
ALLOWED = {'.ipynb', '.py', '.md', '.txt', '.csv', '.json', '.png', '.jpg', '.jpeg', '.svg', '.npz', '.npy', '.yaml', '.yml'}
EXCLUDED = {'__pycache__', '.ipynb_checkpoints', 'results', 'local', 'slides', '.venv', 'venv'}


def copy_download(source, relative):
    destination = DOWNLOADS / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return quote(relative.as_posix())


def main():
    # This directory is generated only by this script and ignored by Git.
    if DOWNLOADS.exists():
        shutil.rmtree(DOWNLOADS)
    DOWNLOADS.mkdir()
    for week in sorted((ROOT / 'weeks').glob('week_*')):
        lines = []
        for part in ['lecture', 'lab']:
            lines += [f'## {part.title()} materials', '']
            pdfs = sorted((week / part / 'slides/pdf').glob('*.pdf'))
            for pdf in pdfs:
                url = copy_download(pdf, Path(week.name) / part / pdf.name)
                lines += [f'[Download {part} slides (PDF)](../../downloads/{url}){{.btn .btn-outline-primary}}', '']
            if not pdfs:
                lines += [f'{part.title()} slide handouts are being prepared.', '']
            if part != 'lab':
                continue
            lab = week / 'lab'
            files = sorted(p for p in lab.rglob('*') if p.is_file() and p.suffix in ALLOWED
                           and not any(x in EXCLUDED or x.startswith('.') for x in p.relative_to(lab).parts))
            if files:
                bundle = DOWNLOADS / week.name / f'{week.name}-lab.zip'
                bundle.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
                    for p in files:
                        z.write(p, str(Path(f'{week.name}-lab') / p.relative_to(lab)))
                lines += [f'[Download complete lab folder](../../downloads/{week.name}/{bundle.name}){{.btn .btn-primary}}', '']
            for nb in sorted(lab.glob('*.ipynb')):
                label = nb.stem.replace('_', ' ')
                if nb.stem == 'temp':
                    label = 'Lab notebook'
                url = copy_download(nb, Path(week.name) / 'lab' / nb.name)
                lines += [f'- **{label}** — [Read online](lab/{quote(nb.name)}) · [Download notebook](../../downloads/{url})']
            if not files:
                lines += ['Lab exercises will be added here.']
            lines += ['']
        (week / '_materials.qmd').write_text('\n'.join(lines))
    lines = []
    for topic in sorted((ROOT / 'topics').iterdir()):
        lines += [f'## {topic.name.replace("_", " ").title()}', '']
        for pdf in sorted(topic.glob('slides/pdf/*.pdf')):
            url = copy_download(pdf, Path('topics') / topic.name / pdf.name)
            lines += [f'[Download handout (PDF)](downloads/{url})', '']
    (ROOT / '_topic-materials.qmd').write_text('\n'.join(lines))
    print('Prepared website downloads and weekly material links.')


if __name__ == '__main__':
    main()
