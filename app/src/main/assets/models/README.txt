# محل قرار دادن فایل‌های مدل

فایل‌های زیر را در همین پوشه (`app/src/main/assets/models/`) قرار دهید:

| فایل | توضیح |
|------|-------|
| `flow_lm_step.onnx` | مدل FlowLM — تولید latentها از واژه‌ها |
| `mimi_decoder_step_kv.onnx` | دیکودر Mimi — تبدیل latent به نمونهٔ صوتی |
| `mimi_encoder.onnx` | انکودر Mimi — ساخت voice cache از صدای مرجع |
| `reference_01.wav` | صدای مرجع (۶۴ توکن) برای voice cache |
| `lut_weight.npy` | **جدول embedding توکن‌های متنی** (خوانده‌شده توسط `LutEmbedding`) |

## دربارهٔ `lut_weight.npy`

این فایل جدول `[vocab_size × dim]` است که هر شناسهٔ توکن را به یک بردار
embedding تبدیل می‌کند. فرمت آن استاندارد NumPy (`.npy`) است.

**اگر این فایل را ندارید، با ابزار `tools/make_fixtures.py` بسازید:**

```bash
# مقادیر پیش‌فرض (vocab=1000, dim=512)
python3 tools/make_fixtures.py

# با ابعاد واقعی مدل خودتان
python3 tools/make_fixtures.py --vocab 4096 --dim 512
```

> ⚠️ جدول ساخته‌شده با این ابزار یک **fixture** است (مقادیر تصادفی کوچک) و فقط
> برای اجرای بدون‌خطا و تست ابعاد/پارس کافی است. برای کیفیت گفتار واقعی، جدول
> وزن‌های آموزش‌دیدهٔ مدل خودتان را جایگزین کنید.

> فایل‌های `.onnx` و `reference_01.wav` به‌دلیل حجم بالا در مخزن قرار داده
> نشده‌اند. پس از افزودن آن‌ها، پروژه را با `./gradlew assembleDebug` بسازید.
