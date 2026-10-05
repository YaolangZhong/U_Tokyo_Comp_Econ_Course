"""Workflow tests with a fake compiler; no TeX installation or course PDFs touched."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_slides


class BuildWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'weeks/week_09/lecture/slides/tex/deck.tex'
        self.source.parent.mkdir(parents=True)
        self.source.write_text(r'\documentclass{beamer}')
        self.pdf = self.source.parent.parent / 'pdf/deck.pdf'
        self.pdf.parent.mkdir()
        self.pdf.write_bytes(b'old handout')
        self.compiler = self.root / 'fake-latexmk'
        self.compiler.write_text('''#!/bin/sh
for arg do
  case "$arg" in -outdir=*) out="${arg#-outdir=}" ;; esac
done
printf 'new handout' > "$out/deck.pdf"
''')
        self.compiler.chmod(0o755)
        self.root_patch = patch.object(build_slides, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def run_build(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return build_slides.build(self.source, str(self.compiler))

    def test_publish_and_failed_rebuild_preserves_handout(self):
        self.assertTrue(self.run_build())
        self.assertEqual(self.pdf.read_bytes(), b'new handout')
        self.compiler.write_text('#!/bin/sh\nexit 1\n')
        self.assertFalse(self.run_build())
        self.assertEqual(self.pdf.read_bytes(), b'new handout')
        self.assertEqual(list(self.pdf.parent.glob('*.tmp')), [])

    def test_success_without_pdf_does_not_replace_handout(self):
        self.compiler.write_text('#!/bin/sh\nexit 0\n')
        self.assertFalse(self.run_build())
        self.assertEqual(self.pdf.read_bytes(), b'old handout')

    def test_week_selection_skips_fragments_and_missing_lab(self):
        (self.source.parent / 'fragment.tex').write_text('a fragment')
        (self.source.parent / 'comment.tex').write_text('% \\documentclass{beamer}')
        args = argparse.Namespace(week=9, part=None, all=False)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(build_slides.select_sources(args), [self.source])
        args.week = 14
        with self.assertRaises(ValueError):
            build_slides.select_sources(args)

    def test_reject_source_outside_slide_folder(self):
        source = self.root / 'other.tex'
        source.write_text('test')
        with self.assertRaises(ValueError):
            build_slides.validate_source(source)


if __name__ == '__main__':
    unittest.main()
