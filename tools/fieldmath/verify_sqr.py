import re, random, sys, os
src = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sqrsplit.inc')).read()
body = src[src.index('{')+1:src.rindex('}')]
prog=[]
for line in body.split('\n'):
    line=line.strip()
    m=re.match(r'(mad_lo_cc|madc_lo_cc|madc_hi_cc|addc_32|add_cc_32|addc_cc_32)\((.*)\);$',line)
    if m: prog.append((m.group(1),[x.strip() for x in m.group(2).split(',')]))
    m2=re.match(r'p\[(\d+)\]\s*=\s*e\[(\d+)\];$',line)
    if m2: prog.append(('mov',[f'p[{m2.group(1)}]',f'e[{m2.group(2)}]']))
def idx(t):
    m=re.match(r'([eop])\[(\d+)\]$',t); return (m.group(1),int(m.group(2))) if m else None
def ai(t):
    m=re.match(r'a\[(\d+)\]$',t); return int(m.group(1)) if m else None
M=(1<<32)-1
def run(A):
    aw=[(A>>(32*i))&M for i in range(8)]
    reg={'e':[0]*16,'o':[0]*16,'p':[0]*16}; cf=0
    for op,args in prog:
        if op=='mov':
            d=idx(args[0]); s=idx(args[1]); reg[d[0]][d[1]]=reg[s[0]][s[1]]
        elif op in ('mad_lo_cc','madc_lo_cc','madc_hi_cc'):
            d=idx(args[0]); x=ai(args[1]); y=ai(args[2]); s=idx(args[3])
            prod=aw[x]*aw[y]
            part=(prod&M) if op.endswith('lo_cc') else (prod>>32)
            cin=cf if op.startswith('madc') else 0
            t=part+reg[s[0]][s[1]]+cin; reg[d[0]][d[1]]=t&M; cf=t>>32
        elif op=='addc_32':
            d=idx(args[0]); s=idx(args[1]); t=reg[s[0]][s[1]]+int(args[2])+cf
            reg[d[0]][d[1]]=t&M; cf=0
        elif op=='add_cc_32':
            d=idx(args[0]); s1=idx(args[1]); s2=idx(args[2])
            t=reg[s1[0]][s1[1]]+reg[s2[0]][s2[1]]; reg[d[0]][d[1]]=t&M; cf=t>>32
        elif op=='addc_cc_32':
            d=idx(args[0]); s1=idx(args[1]); s2=idx(args[2])
            t=reg[s1[0]][s1[1]]+reg[s2[0]][s2[1]]+cf; reg[d[0]][d[1]]=t&M; cf=t>>32
    return sum(reg['p'][i]<<(32*i) for i in range(16))
random.seed(7)
tests=[0,1,(1<<256)-1,(1<<255),(1<<128)-1]+[random.getrandbits(256) for _ in range(300)]
bad=0
for A in tests:
    if run(A)!=A*A:
        bad+=1
        if bad<3: print("MISMATCH",hex(A))
print(f"{len(tests)-bad}/{len(tests)} pass; instructions={len(prog)}")
