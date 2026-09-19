"""Print a GGUF file's metadata, parameter count and quantization mix.

Standard library only, on purpose: the point of this script is to let someone check
what a downloaded model actually is before installing anything, and to let the
README make claims about the shipped weights that a reader can verify in one
command.

    python scripts/gguf_header.py models/docker_gemma3_4b_q4km.gguf

Format reference: GGUF v2/v3, ggml-org/llama.cpp docs/gguf.md.
"""

from __future__ import annotations

import collections
import struct
import sys

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

_SCALAR = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0",
}

# K-quants pack 256 elements per superblock, so they cannot be used on a tensor
# whose row length is not a multiple of 256. llama.cpp falls back to a non-K type
# for those, which is why a "Q4_K_M" file can be mostly Q5_0.
SUPERBLOCK = 256


class _Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"truncated: wanted {n} bytes, got {len(b)}")
        return b

    def scalar(self, t: int):
        fmt, n = _SCALAR[t]
        return struct.unpack(fmt, self.raw(n))[0]

    def string(self) -> str:
        return self.raw(self.scalar(U64)).decode("utf-8", "replace")

    def value(self, t: int):
        if t == STRING:
            return self.string()
        if t == ARRAY:
            elem = self.scalar(U32)
            return [self.value(elem) for _ in range(self.scalar(U64))]
        return self.scalar(t)


def main(path: str) -> int:
    try:
        f = open(path, "rb")
    except OSError as exc:
        print(f"cannot read {path}: {exc.strerror}", file=sys.stderr)
        return 1

    with f:
        r = _Reader(f)

        magic = r.raw(4)
        if magic != b"GGUF":
            print(f"not a GGUF file: magic is {magic!r}", file=sys.stderr)
            return 1

        version = r.scalar(U32)
        n_tensors = r.scalar(U64)
        n_kv = r.scalar(U64)

        print(f"file          : {path}")
        print(f"gguf version  : {version}")
        print(f"tensors       : {n_tensors}")
        print(f"metadata keys : {n_kv}")
        print()

        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.scalar(U32))

        print("--- metadata ---")
        for key, val in kv.items():
            if isinstance(val, list):
                print(f"{key} = [len={len(val)}] {val[:6]}{' ...' if len(val) > 6 else ''}")
            elif isinstance(val, str) and len(val) > 120:
                print(f"{key} = {val[:117]!r}...")
            else:
                print(f"{key} = {val!r}")

        print()
        print("--- tensors ---")
        params = 0
        by_type = collections.Counter()
        params_by_type = collections.Counter()
        fallbacks = collections.Counter()

        for _ in range(n_tensors):
            r.string()  # name
            dims = [r.scalar(U64) for _ in range(r.scalar(U32))]
            ttype = r.scalar(U32)
            r.scalar(U64)  # offset

            n = 1
            for d in dims:
                n *= d
            params += n
            by_type[ttype] += 1
            params_by_type[ttype] += n

            if len(dims) >= 2:
                name = GGML_TYPES.get(ttype, str(ttype))
                fallbacks[(dims[0] % SUPERBLOCK == 0, "_K" in name)] += 1

        print(f"total parameters : {params:,}  ({params / 1e9:.3f} B)")
        print("quantization mix :")
        for ttype, count in by_type.most_common():
            name = GGML_TYPES.get(ttype, str(ttype))
            print(f"  {name:8s} {count:4d} tensors  {params_by_type[ttype]:>14,} params")

        print()
        print(f"2-D tensors by whether row length is a multiple of {SUPERBLOCK}:")
        for (divisible, is_k), count in sorted(fallbacks.items()):
            print(f"  row length divisible by {SUPERBLOCK}: {str(divisible):5s}   "
                  f"K-quant: {str(is_k):5s}   {count} tensors")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(f"usage: python {sys.argv[0]} <model.gguf>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
