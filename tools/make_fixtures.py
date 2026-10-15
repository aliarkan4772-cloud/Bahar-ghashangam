#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fixtures.py — ساخت فایل‌های fixture مدل برای پروژهٔ Pocket-TTS فارسی.
================================================================================
این ابزار فایل `lut_weight.npy` (جدول embedding توکن‌های متنی) را می‌سازد؛
همان فایلی که کلاس `LutEmbedding.java` می‌خواند. همچنین می‌تواند یک فایل
`reference_01.wav` نمونه (برای تست دسکتاپ) تولید کند.

چرا این فایل لازم است؟
    موتور TTS برای تبدیل هر توکن متنی به بردار embedding به یک جدول
    [vocab_size × dim] نیاز دارد. در مدل واقعی این جدول از وزن‌های آموزش‌دیده
    می‌آید؛ اما برای اجرای بدون‌خطای اپ و تست pipeline، یک جدول fixture
    با ابعاد درست کافی است.

ابعاد واقعی مدل را نمی‌دانیم؟ مشکلی نیست:
    مقادیر پیش‌فرض (vocab=1000, dim=512) «معقول و قابل تنظیم» هستند و با
    آرگومان‌های خط فرمان قابل تغییرند. برای بازتولید با ابعاد واقعی مدل،
    کافی است vocab و dim را از config مدل خود بردارید و همین اسکریپت را
    دوباره اجرا کنید:

        python3 tools/make_fixtures.py --vocab 4096 --dim 512

    سپس فایل خروجی را در مسیر زیر قرار دهید:
        app/src/main/assets/models/lut_weight.npy

نکتهٔ مهم دربارهٔ «معناداری»:
    این جدول fixture فقط برای اجرای بدون‌خطا و تست ابعاد/پارس است. برای
    کیفیت گفتار واقعی باید جدول وزن‌های آموزش‌دیدهٔ مدل خودتان را جایگزین کنید.
================================================================================
"""

import argparse
import json
import os
import struct
import sys

try:
    import numpy as np
except ImportError:
    sys.stderr.write(
        "خطا: کتابخانهٔ numpy نصب نیست.\n"
        "نصب: pip install numpy\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# نوشتن فایل .npy به‌صورت دستی (بدون وابستگی به np.save)
# ---------------------------------------------------------------------------
def write_npy_manual(path, array):
    """
    نوشتن یک آرایهٔ numpy در قالب فایل .npy (نسخهٔ ۱.۰).

    ساختار فایل:
        magic   : b"\\x93NUMPY"  (۶ بایت)
        version : b"\\x01\\x00"   (major=1, minor=0)
        hlen    : uint16 LE      (طول هدر)
        header  : ASCII          (دیکشنری پایتون، با فاصله پر شده، \\n پایانی)
        data    : خام، row-major
    """
    array = np.ascontiguousarray(array, dtype="<f4")
    if array.ndim != 2:
        raise ValueError("آرایه باید دوبُعدی باشد [vocab, dim]")

    rows, cols = array.shape
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (rows, cols)

    # هدر باید مضربی از ۶۴ بایت باشد (قرارداد numpy) و با \n تمام شود.
    magic = b"\x93NUMPY"
    version = b"\x01\x00"
    base = len(magic) + len(version) + 2  # +2 برای فیلد hlen
    total = base + len(header) + 1        # +1 برای \n پایانی
    pad = (64 - (total % 64)) % 64
    header_padded = header + (" " * pad) + "\n"
    hlen = len(header_padded)

    with open(path, "wb") as f:
        f.write(magic)
        f.write(version)
        f.write(struct.pack("<H", hlen))
        f.write(header_padded.encode("ascii"))
        f.write(array.tobytes(order="C"))


# ---------------------------------------------------------------------------
# ساخت جدول embedding
# ---------------------------------------------------------------------------
def make_lut(vocab, dim, seed, scale):
    """
    ساخت جدول embedding با توزیع نرمال کوچک.

    از توزیع نرمال با انحراف معیار `scale` استفاده می‌کنیم تا مقادیر
    در بازهٔ معقولی بمانند (شبیه وزن‌های واقعی embedding که کوچک‌اند).
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((vocab, dim)) * scale).astype("<f4")


# ---------------------------------------------------------------------------
# ساخت یک WAV مرجع نمونه (برای تست دسکتاپ)
# ---------------------------------------------------------------------------
def make_reference_wav(path, seconds=1.0, sample_rate=24000, freq=180.0):
    """
    ساخت یک WAV مونو ۱۶ بیتی با یک تُن سادهٔ هارمونیک (به‌عنوان صدای مرجع).
    این فایل فقط برای تست pipeline است، نه صدای واقعی گوینده.
    """
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    # چند هارمونیک + پوش کوچک تا شبیه یک صدای مرجع ساده باشد
    wave = (0.5 * np.sin(2 * np.pi * freq * t)
            + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.12 * np.sin(2 * np.pi * 3 * freq * t))
    env = np.minimum(1.0, np.minimum(t * 20.0, (seconds - t) * 20.0))
    wave = np.clip(wave * env, -1.0, 1.0)
    pcm = (wave * 32767.0).astype("<i2")

    with open(path, "wb") as f:
        data = pcm.tobytes()
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))            # PCM
        f.write(struct.pack("<H", 1))            # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))
        f.write(struct.pack("<H", 2))            # block align
        f.write(struct.pack("<H", 16))           # bits
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="ساخت فایل‌های fixture مدل (lut_weight.npy و reference_01.wav)")
    ap.add_argument("--vocab", type=int, default=1000,
                    help="تعداد توکن‌ها (vocab size). پیش‌فرض: 1000")
    # نکته: پیش‌فرض dim از model_contract.json خوانده می‌شود تا هرگز با text_dim ناهمخوان نشود.
    ap.add_argument("--dim", type=int, default=None,
                    help="بُعد embedding. اگر داده نشود، از فیلد text_dim در "
                         "app/src/main/assets/models/model_contract.json خوانده می‌شود"
                         " (و در نبود آن، پیش‌فرض 512).")
    ap.add_argument("--seed", type=int, default=20261015,
                    help="دانهٔ تصادفی برای تکرارپذیری. پیش‌فرض: 20261015")
    ap.add_argument("--scale", type=float, default=0.05,
                    help="انحراف معیار مقادیر جدول. پیش‌فرض: 0.05")
    ap.add_argument("--out", type=str, default=None,
                    help="مسیر خروجی lut_weight.npy (پیش‌فرض: assets/models/)")
    ap.add_argument("--wav", action="store_true",
                    help="ساخت reference_01.wav نمونه هم")
    ap.add_argument("--wav-out", type=str, default=None,
                    help="مسیر خروجی reference_01.wav")
    args = ap.parse_args()

    # مسیر پیش‌فرض: کنار همین اسکریپت → ../app/src/main/assets/models/
    here = os.path.dirname(os.path.abspath(__file__))
    default_models = os.path.join(here, "..", "app", "src", "main", "assets", "models")
    default_models = os.path.normpath(default_models)

    out = args.out or os.path.join(default_models, "lut_weight.npy")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # تعیین بُعد: اولویت با --dim، سپس text_dim در model_contract.json،
    # در نهایت پیش‌فرض 512. این کار از تناقض text_dim و ابعاد فایل جلوگیری می‌کند.
    # ─────────────────────────────────────────────────────────────
    contract_path = os.path.join(default_models, "model_contract.json")
    contract = {}
    contract_dim = None
    if os.path.exists(contract_path):
        try:
            with open(contract_path, "r", encoding="utf-8") as f:
                contract = json.load(f)
            contract_dim = int(contract.get("text_dim"))
        except Exception as e:
            print("⚠ قرارداد مدل خوانده نشد (%s) — از پیش‌فرض استفاده می‌شود" % e)
            contract, contract_dim = {}, None

    if args.dim is not None and contract_dim is not None and args.dim != contract_dim:
        print("✗ تناقض ابعاد:")
        print("    --dim = %d  اما  text_dim در model_contract.json = %d" % (args.dim, contract_dim))
        print("    برای رفع تناقض، یکی از دو کار را انجام دهید:")
        print("      • فایل را با ابعاد قرارداد بسازید:  --dim %d" % contract_dim)
        print("      • یا text_dim را در model_contract.json و Dims.TEXT_DIM به %d تغییر دهید" % args.dim)
        sys.exit(2)

    dim = args.dim if args.dim is not None else (contract_dim if contract_dim is not None else 512)

    print("ساخت جدول embedding ...")
    print("  vocab = %d" % args.vocab)
    print("  dim   = %d%s" % (dim, "  (از model_contract.json)" if args.dim is None and contract_dim else ""))
    print("  seed  = %d" % args.seed)

    lut = make_lut(args.vocab, dim, args.seed, args.scale)
    write_npy_manual(out, lut)

    # آفست واقعی داده را گزارش کن (مهارت: آفست ۱۲۸ ثابت نیست)
    with open(out, "rb") as f:
        head = f.read(12)
        major = head[6]
        if major == 1:
            hlen = head[8] | (head[9] << 8)
            hstart = 10
        else:
            hlen = int.from_bytes(head[8:12], "little")
            hstart = 12
        data_offset = hstart + hlen

    size = os.path.getsize(out)
    print("✓ نوشته شد: %s" % out)
    print("  shape = %s  |  dtype = float32  |  حجم = %d بایت" % (str(lut.shape), size))
    print("  آفست شروع داده در npy = %d بایت (ثابت نیست!)  |  headerLen = %d" % (data_offset, hlen))

    # اعتبارسنجی: فایل را دوباره با numpy بخوان و ابعاد را بررسی کن
    check = np.load(out)
    assert check.shape == (args.vocab, dim), "ابعاد فایل نوشته‌شده درست نیست!"
    assert check.dtype == np.float32, "نوع داده باید float32 باشد!"
    print("✓ اعتبارسنجی با numpy.load موفق بود (shape=%s, dtype=%s)"
          % (str(check.shape), check.dtype))

    # ─────────────────────────────────────────────────────────────
    # به‌روزرسانی model_contract.json تا text_dim همواره با فایل همخوان باشد
    # ─────────────────────────────────────────────────────────────
    try:
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, ensure_ascii=False, indent=2)
        print("✓ contract: text_dim = %d در model_contract.json نوشته شد" % dim)
    except Exception as e:
        print("⚠ نوشتن model_contract.json ناموفق: %s" % e)

    # هشدار: اگر Dims.TEXT_DIM در Java همخوان نباشد
    dims_java = os.path.join(here, "..", "app", "src", "main", "java",
                            "com", "pockettts", "persian", "Dims.java")
    if os.path.exists(dims_java):
        try:
            src = open(dims_java, "r", encoding="utf-8").read()
            import re as _re
            m = _re.search(r"TEXT_DIM\s*=\s*(\d+)", src)
            if m and int(m.group(1)) != dim:
                print("✗ هشدار: Dims.TEXT_DIM = %s است، ولی فایل با dim=%d ساخته شد."
                      % (m.group(1), dim))
                print("   این دو باید یکی باشند؛ Dims.java را اصلاح کنید.")
        except Exception:
            pass

    if args.wav:
        wav_out = args.wav_out or os.path.join(default_models, "reference_01.wav")
        os.makedirs(os.path.dirname(os.path.abspath(wav_out)), exist_ok=True)
        make_reference_wav(wav_out)
        print("✓ نوشته شد: %s (%d بایت)" % (wav_out, os.path.getsize(wav_out)))

    print("\nحالا فایل را در مسیر زیر قرار دهید (اگر --out را تغییر نداده‌اید، همین‌جا هست):")
    print("  app/src/main/assets/models/lut_weight.npy")


if __name__ == "__main__":
    main()
