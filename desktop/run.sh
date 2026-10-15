#!/usr/bin/env bash
# run.sh — کامپایل و اجرای نسخهٔ دسکتاپ موتور TTS فارسی.
# ============================================================
# استفاده:
#   ./desktop/run.sh "متن فارسی دلخواه"
#   ./desktop/run.sh "متن" out.wav
#
# نکته: کلاس TtsEngine به ONNX Runtime وابسته است؛ برای کامپایل بدون JAR واقعی،
# استاب‌های سبک در desktop/stubs/ قرار دارند و همین‌جا کامپایل می‌شوند.
# کلاس MainActivity (اندرویدی) از کامپایل دسکتاپ کنار گذاشته می‌شود.
# ============================================================
set -e

# ریشهٔ پروژه = پوشهٔ والد همین اسکریپت
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TEXT="${1:-سلام، این یک آزمایش موتور تبدیل متن به گفتار فارسی است.}"
OUT="${2:-desktop_out.wav}"

echo "▶ کامپایل کلاس‌ها (شامل استاب‌های ONNX، بدون MainActivity) ..."
rm -rf build-desktop
mkdir -p build-desktop

# همهٔ کلاس‌های مشترک به‌جز MainActivity (که فقط روی اندروید معنا دارد)
SOURCES=$(find desktop/stubs app/src/main/java desktop/src -name '*.java' \
          ! -name 'MainActivity.java')

javac -d build-desktop $SOURCES

echo "▶ اجرا ..."
java -cp build-desktop com.pockettts.persian.DesktopMain \
     --lut app/src/main/assets/models/lut_weight.npy \
     --out "$OUT" \
     --text "$TEXT"

echo "✓ خروجی: $OUT"
