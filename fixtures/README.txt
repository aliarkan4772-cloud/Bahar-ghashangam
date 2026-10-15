# پوشهٔ fixtures — فایل‌های نمونه برای تست دسکتاپ

این پوشه دو فایل تست دارد:

| فایل | ابعاد | توضیح |
|------|-------|-------|
| `lut_bad_dim.npy` | (200, 1024) | نمونهٔ **ناهمخوان** — باید توسط LutEmbedding رد شود |
| `lut_weight.npy` | (50257, 64) | نمونهٔ **درست** — برای اجرای تست لازم است |

⚠️ فایل `lut_weight.npy` عیناً همان فایل `app/src/main/assets/models/lut_weight.npy` است.
برای صرفه‌جویی در حجم، نسخهٔ تکراری در zip قرار داده نشده. آن را این‌گونه بسازید:

    cp app/src/main/assets/models/lut_weight.npy fixtures/lut_weight.npy

سپس تست را اجرا کنید:

    javac -nowarn -d out $(find ../stubs desktop/src ../app/src/main/java -name '*.java' ! -name 'MainActivity.java')
    java -cp out com.pockettts.persian.LutDimTest fixtures/lut_weight.npy fixtures/lut_bad_dim.npy

خروجی مورد انتظار: «۴ موفق، ۰ ناموفق».
