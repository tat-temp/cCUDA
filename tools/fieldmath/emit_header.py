#!/usr/bin/env python3
# Splice the generated product cores into ec_backend.cuh between the BEGIN/END markers.
# Run from anywhere:  python3 tools/fieldmath/emit_header.py
# Regenerate the cores first:
#   python3 tools/fieldmath/gen.py    | tr -d '\r' > tools/fieldmath/split.inc
#   python3 tools/fieldmath/gensqr.py | tr -d '\r' > tools/fieldmath/sqrsplit.inc
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
HDR = os.path.join(ROOT, 'ec_backend.cuh')

BEGIN = '// ---- BEGIN GENERATED (tools/fieldmath) ----'
END = '// ---- END GENERATED ----'


def main():
    mul = open(os.path.join(HERE, 'split.inc')).read().replace('\r\n', '\n').rstrip('\n')
    sqr = open(os.path.join(HERE, 'sqrsplit.inc')).read().replace('\r\n', '\n').rstrip('\n')
    src = open(HDR).read().replace('\r\n', '\n')

    i, j = src.find(BEGIN), src.find(END)
    if i < 0 or j < 0:
        sys.exit('ec_backend.cuh: BEGIN/END GENERATED markers not found')

    body = BEGIN + '\n' + mul + '\n\n' + sqr + '\n' + END
    out = src[:i] + body + src[j + len(END):]

    with open(HDR, 'w', newline='\n') as f:
        f.write(out)
    print('ec_backend.cuh: spliced %d + %d generated lines'
          % (mul.count('\n') + 1, sqr.count('\n') + 1))


if __name__ == '__main__':
    main()
