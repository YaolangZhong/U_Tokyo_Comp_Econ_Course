#!/usr/bin/env python3
"""Generate download bundles and material links from existing weekly files."""
from pathlib import Path
import shutil
import json
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
    plan = json.loads((ROOT / 'course_plan.json').read_text())
    schedules = {f"week_{item['number']:02}": item for item in plan['weeks']}
    for week in sorted((ROOT / 'weeks').glob('week_*')):
        lines = []
        for part in ['lecture', 'lab']:
            lines += [f'## {part.title()} materials', '']
            if part == 'lecture':
                for topic_id in schedules[week.name]['lecture_topics']:
                    topic = plan['topics'][topic_id]
                    title = topic['title']
                    pdf = ROOT / topic['slides_dir'] / 'pdf' / (topic['stem'] + '.pdf')
                    lines += [f"### {title}", '']
                    if topic['status'] == 'placeholder':
                        lines += ['Slide deck placeholder — content and length to be discussed.', '']
                    elif not pdf.is_file():
                        raise FileNotFoundError(f"Missing assigned topic PDF: {pdf}")
                    else:
                        url = copy_download(pdf, Path(week.name) / part / pdf.name)
                        lines += [f'[Download {title} (PDF)](../../downloads/{url}){{.btn .btn-outline-primary}}', '']
                continue
            pdfs = sorted((week / part / 'slides/pdf').glob('*.pdf'))
            for pdf in pdfs:
                url = copy_download(pdf, Path(week.name) / part / pdf.name)
                title = pdf.stem.replace('_', ' ')
                lines += [f'[Download {title} (PDF)](../../downloads/{url}){{.btn .btn-outline-primary}}', '']
            if not pdfs:
                lines += ['Research-agent/tools slide deck placeholder — topic and activity to be decided.', '']
            lab = week / 'lab'
            files = sorted(p for p in lab.rglob('*') if p.is_file() and p.suffix in ALLOWED
                           and not any(x in EXCLUDED or x.startswith('.') for x in p.relative_to(lab).parts))
            if any(p.name != 'README.md' and not p.name.endswith('.placeholder.md') for p in files):
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
            if not list(lab.glob('*.ipynb')):
                lines += ['The new lab activity has not been assigned yet.']
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
