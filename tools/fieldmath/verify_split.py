import re, random, sys, os
src = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split.inc')).read()
body = src[src.index('{')+1:src.rindex('}')]
prog = []
for line in body.split('\n'):
    line = line.strip()
    m = re.match(r'(mad_lo_cc|madc_lo_cc|madc_hi_cc|addc_32|add_cc_32|addc_cc_32)\((.*)\);$', line)
    if m:
        prog.append((m.group(1), [x.strip() for x in m.group(2).split(',')]))

def idx(tok):
    m = re.match(r'([eop])\[(\d+)\]$', tok); return (m.group(1), int(m.group(2))) if m else None
def ab(tok):
    m = re.match(r'([ab])\[(\d+)\]$', tok); return (m.group(1), int(m.group(2))) if m else None

M = (1 << 32) - 1
def run(A, B):
    aw = [(A >> (32*i)) & M for i in range(8)]
    bw = [(B >> (32*i)) & M for i in range(8)]
    reg = {'e': [0]*16, 'o': [0]*16, 'p': [0]*16}
    cf = 0
    for op, args in prog:
        if op in ('mad_lo_cc', 'madc_lo_cc', 'madc_hi_cc'):
            d = idx(args[0]); x = ab(args[1]); y = ab(args[2]); s = idx(args[3])
            prod = (aw[x[1]] if x[0]=='a' else bw[x[1]]) * (aw[y[1]] if y[0]=='a' else bw[y[1]])
            part = (prod & M) if op.endswith('lo_cc') else (prod >> 32)
            cin = cf if op.startswith('madc') else 0
            t = part + reg[s[0]][s[1]] + cin
            reg[d[0]][d[1]] = t & M; cf = t >> 32
        elif op == 'addc_32':
            d = idx(args[0]); s = idx(args[1])
            t = reg[s[0]][s[1]] + int(args[2]) + cf
            reg[d[0]][d[1]] = t & M; cf = 0
        elif op == 'add_cc_32':
            d = idx(args[0]); s1 = idx(args[1]); s2 = idx(args[2])
            t = reg[s1[0]][s1[1]] + reg[s2[0]][s2[1]]
            reg[d[0]][d[1]] = t & M; cf = t >> 32
        elif op == 'addc_cc_32':
            d = idx(args[0]); s1 = idx(args[1]); s2 = idx(args[2])
            t = reg[s1[0]][s1[1]] + reg[s2[0]][s2[1]] + cf
            reg[d[0]][d[1]] = t & M; cf = t >> 32
    reg['p'][0] = reg['e'][0]
    return sum(reg['p'][i] << (32*i) for i in range(16))

random.seed(1)
bad = 0
tests = [(0,0), ((1<<256)-1, (1<<256)-1), (1, (1<<256)-1), ((1<<255), (1<<255))]
tests += [(random.getrandbits(256), random.getrandbits(256)) for _ in range(300)]
for A, B in tests:
    got = run(A, B); want = A*B
    if got != want:
        bad += 1
        if bad < 3: print("MISMATCH", hex(A), hex(B), "diff", hex(got-want) if got>want else "-"+hex(want-got))
print(f"{len(tests)-bad}/{len(tests)} pass;  instructions={len(prog)}")
