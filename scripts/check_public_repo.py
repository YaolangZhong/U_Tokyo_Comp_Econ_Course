#!/usr/bin/env python3
"""Reject accidentally tracked internal course directories."""
import subprocess
import sys

private = {'docs', 'archive', 'Final_Project', '.vscode', 'backend'}
files = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
violations = [name for name in files if name.split('/')[0] in private]
if violations:
    print('Internal files must remain local:\n' + '\n'.join(violations), file=sys.stderr)
    sys.exit(1)
print('PASS: internal course directories are not tracked.')
