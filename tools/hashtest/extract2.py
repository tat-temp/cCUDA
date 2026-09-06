#!/usr/bin/env python3
"""Extract the hot kernel from a baseline ref and from the working tree, for difftest.cpp.

Emits kernel_old.inc (baseline) and kernel_new.inc (working tree) next to this script, with the
kernel renamed and its hash call sites redirected to the HOOK_* wrappers difftest.cpp defines.

    BASE_REF=main python3 extract2.py      # default baseline is `main`

Pin BASE_REF to the commit the change is being judged against. Once a change is merged, comparing
against `main` compares the tree with itself (a trivial pass) -- set BASE_REF to the pre-change
commit instead.
"""
import io
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_REF = os.environ.get("BASE_REF", "main")


def kernel_text(src):
    """Return just the hot kernel function (from its __launch_bounds__ line to its close)."""
    src = src.replace("\r\n", "\n")
    kpos = src.index("__global__ void kernel_point_add_and_check_oneinv")
    lb = src.rindex("__launch_bounds__", 0, kpos)
    bpos = src.index("{", kpos)
    depth = 0
    for k in range(bpos, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[lb:k + 1]
    raise AssertionError("unbalanced braces in kernel")


def rename(body, newname):
    body = body.replace("kernel_point_add_and_check_oneinv", newname)
    # redirect the hash entry points through difftest.cpp's logging wrappers
    body = body.replace("getHash160_w2_from_limbs(", "HOOK_w2(")
    body = body.replace("getHash160_w2_x2(", "HOOK_w2x2(")
    return body


old_src = subprocess.check_output(
    ["git", "-C", REPO, "show", "%s:CUDACyclone.cu" % BASE_REF]).decode("utf-8", "replace")
new_src = io.open(os.path.join(REPO, "CUDACyclone.cu"), encoding="utf-8", newline="").read()

old_k = rename(kernel_text(old_src), "kernel_old")
new_k = rename(kernel_text(new_src), "kernel_new")

io.open(os.path.join(HERE, "kernel_old.inc"), "w", encoding="utf-8", newline="\n").write(old_k + "\n")
io.open(os.path.join(HERE, "kernel_new.inc"), "w", encoding="utf-8", newline="\n").write(new_k + "\n")

print("baseline ref : %s" % BASE_REF)
print("old kernel   : %d lines, %d 1-wide + %d 2-wide hash sites"
      % (old_k.count("\n") + 1, old_k.count("HOOK_w2("), old_k.count("HOOK_w2x2(")))
print("new kernel   : %d lines, %d 1-wide + %d 2-wide hash sites"
      % (new_k.count("\n") + 1, new_k.count("HOOK_w2("), new_k.count("HOOK_w2x2(")))
