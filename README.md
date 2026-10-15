# Persian Pocket-TTS — موتور تبدیل متن به گفتار فارسی آفلاین (اندروید)

پروژهٔ کامل و قابل‌build اندروید برای موتور TTS فارسی آفلاین با **Java + ONNX Runtime**.
همهٔ اصلاحات تحلیل‌شده (EOS، max_steps، offset، retry، compress_pauses، speech_bounds،
حذف median-RMS، Chunker v8 و loadPlan اصلاح‌شده) در این پروژه اعمال شده است.
**نسخهٔ v2:** افزودن **حالت موازی (ExecutorService)** برای سنتز هم‌زمان chunkها.
**نسخهٔ v3:** افزودن **LutEmbedding** (خواندن `lut_weight.npy`)، ابزار **make_fixtures.py**، و یک **نسخهٔ دسکتاپ قابل‌اجرا** که واقعاً WAV می‌سازد.
**نسخهٔ v4 (این نسخه):** رفع مشکل ابعاد LUT — افزودن **`Dims.java`** و **`model_contract.json`** به‌عنوان منبع یگانهٔ حقیقت ابعاد، مقاوم‌سازی **`LutEmbedding`** (خواندن آفست از هدر، اعتبارسنجی ابعاد)، و ساخت فایل درست `lut_weight.npy` با ابعاد **`[50257 × 64]`**.

---

## ۱) پیش‌نیازها

- Android Studio (Hedgehog یا جدیدتر) یا فقط Gradle + Android SDK
- JDK 17
- Android SDK با `compileSdk 34`
- فایل‌های مدل (بخش ۳)

---

## ۲) ساخت و اجرا

```bash
# ساخت نسخهٔ دیباگ
./gradlew assembleDebug

# نصب روی دستگاه/شبیه‌ساز متصل
./gradlew installDebug
```

خروجی APK در مسیر زیر ساخته می‌شود:

```
app/build/outputs/apk/debug/app-debug.apk
```

سپس اپلیکیشن «پاکت TTS فارسی» را باز کنید، متن فارسی را بنویسید و
دکمهٔ «تبدیل به گفتار» را بزنید. با کلید **حالت موازی** می‌توانید بین حالت سری و موازی
جابه‌جا شوید و با **مقایسهٔ سری و موازی** زمان هر دو حالت را ببینید.

---

## ۳) محل قرار دادن فایل‌های مدل

فایل‌های زیر را در پوشهٔ `app/src/main/assets/models/` قرار دهید:

| فایل | نقش |
|------|-----|
| `flow_lm_step.onnx` | FlowLM — تولید latent از واج‌ها |
| `mimi_decoder_step_kv.onnx` | دیکودر Mimi — latent → نمونهٔ صوتی |
| `mimi_encoder.onnx` | انکودر Mimi — ساخت voice cache |
| `reference_01.wav` | صدای مرجع (۶۴ توکن) |

> در زمان اجرا، این فایل‌ها از assets به `getFilesDir()/models` استخراج می‌شوند.
> اگر انکودر موجود نباشد، موتور با voice cache صفر ادامه می‌دهد (بدون توقف).

---

## ۴) حالت موازی (v2) — سنتز هم‌زمان chunkها

### ایده
chunkها از هم **مستقل** هستند، چون هر کدام از «حالت صدای دست‌نخورده»
(`pristine voice state`) تولید می‌شود. پس می‌توان آن‌ها را هم‌زمان روی چند thread
سنتز کرد. کلاس `ParallelSynthesizer` این کار را با یک `ExecutorService` انجام می‌دهد.

### جریان کار

```
                    متن فارسی
                        │
                        ▼
             Chunker.chunk()  ──►  jobs[]  (هر chunk + gap خودش)
                        │
                        ▼
        ┌──────── ParallelSynthesizer ────────┐
        │                                     │
        ▼          ▼          ▼          ▼     │
   worker#1   worker#2   worker#3   worker#4   │   ← ThreadPoolExecutor
   (TtsEngine (TtsEngine (TtsEngine (TtsEngine  │      پیش‌فرض ۴ thread
    مستقل)      مستقل)      مستقل)      مستقل)    │      (۱ تا ۸)
        │          │          │          │     │
        │  هر chunk از voice cache کپی‌شده      │
        │  + rng مستقل per-chunk               │
        ▼          ▼          ▼          ▼     │
     future[0]  future[1]  future[2]  future[3] │  ← به ترتیب لیست جمع می‌شوند
        └──────────┴──────────┴──────────┘     │     (نه به ترتیب تمام‌شدن)
                        │                     │
                        ▼                     │
        stitchWithGaps()  ──►  WAV نهایی به ترتیب اصلی
```

### تضمین‌های کلیدی

| # | تضمین | چگونه |
|---|-------|-------|
| ۱ | **ترتیب اصلی حفظ می‌شود** | futureها به ترتیب لیست `get()` می‌شوند، نه به ترتیب تمام‌شدن |
| ۲ | **هر thread وضعیت مستقل دارد** | یک نمونهٔ `TtsEngine` جدا برای هر worker (نه قفل روی نمونهٔ مشترک) |
| ۳ | **voice cache دست‌نخورده** | هر فراخوانی با `TtsEngine.copyVoiceCache(...)` کپی محلی می‌سازد |
| ۴ | **خطای یک chunk کل کار را متوقف نمی‌کند** | خطا هر job جداگانه گرفته و در `failures` گزارش می‌شود |
| ۵ | **fallback سری** | اگر فقط ۱ thread در دسترس باشد یا استخر رد شود، خودکار به حالت سری برمی‌گردد |

### کنترل تعداد thread

- **پیش‌فرض: ۴ thread** (`ParallelSynthesizer.DEFAULT_THREADS`)
- قابل تنظیم از UI با `SeekBar` (۱ تا ۸) یا مستقیماً در کد
- `clampThreads(n)` مقدار نامعتبر را به بازهٔ ۱..۸ می‌آورد
- `autoThreads(n)` بر اساس سقف heap، تعداد ایمن را برمی‌گرداند (حافظهٔ کم → thread کمتر)

### نکات حافظه

- هر worker یک نمونهٔ ONNX جدا می‌سازد؛ هر نمونه چند صد مگابایت می‌گیرد.
  روی دستگاه‌های کم‌حافظه، تعداد thread را پایین بیاورید (۲ کافی است).
- `autoThreads` خودش بر اساس `Runtime.maxMemory()` عقب‌نشینی می‌کند تا از OOM جلوگیری شود.
- مثال در `MainActivity.synthesizeParallel`:

```java
ParallelSynthesizer synth = new ParallelSynthesizer(threads)
        .withVoiceCache(voiceCache);

ParallelSynthesizer.Outcome out = synth.synthesize(
        jobs,
        () -> { TtsEngine e = new TtsEngine(modelDir); e.load(); return e; },  // per-thread
        engine,                                                              // fallback سری
        threads);
```

### نکات ترتیب

- مکث بین chunkها از ستون `gap` هر job می‌آید (`stitchWithGaps`)، پس gapهای
  plan.tsv در حالت موازی هم دقیقاً رعایت می‌شوند.
- خروجی موازی با خروجی سری **هم‌طول** است؛ فقط ترتیب اجرا فرق دارد، نه ترتیب صدا.

---

## ۳.۵) ابعاد مدل — `Dims.java` و `model_contract.json`

منبع یگانهٔ حقیقت برای ابعاد دو جاست:

| مکان | مقدار | نقش |
|------|-------|------|
| `app/src/main/java/com/pockettts/persian/Dims.java` | `TEXT_DIM = 64` | ثابتی که کد Java از آن استفاده می‌کند |
| `app/src/main/assets/models/model_contract.json` | `"text_dim": 64` | قراردادی که ابزار و موتور از آن می‌خوانند |
| `tools/make_fixtures.py --dim` | `64` | ابزاری که فایل LUT را می‌سازد |

**هر سه باید یکی باشند.** `LutEmbedding` هنگام بارگذاری، بُعد فایل را با مقدار مورد انتظار
مقایسه می‌کند و اگر ناهمخوان باشد، با پیام فارسی و راه‌حل متوقف می‌شود:

```
ابعاد LUT ناهمخوان است:
  فایل .../lut_weight.npy بُعد 1024 دارد، اما بُعد مورد انتظار 64 است.
  (vocab=1000، ابعاد خوانده‌شده=1000×1024)
راه‌حل: فایل را با ابعاد درست بازتولید کنید:
   python3 tools/make_fixtures.py --vocab 1000 --dim 64
```

> **نکتهٔ فنی:** `TEXT_DIM` (بُعد متن، = ۶۴) و `LATENT_DIM` (بُعد latent صوتی، = ۵۱۲)
> دو چیز جدا هستند. `TtsEngine.LDIM` بُعد latent است و به LUT مربوط نیست.

---

## ۴.۵) جدول embedding توکن‌ها — `lut_weight.npy` و `LutEmbedding`

### چرا لازم است؟
موتور TTS برای تبدیل هر توکن متنی به یک بردار عددی (embedding) به یک جدول
`[vocab_size × dim]` نیاز دارد. کلاس `LutEmbedding.java` این جدول را از فایل
`models/lut_weight.npy` می‌خواند. **بدون این فایل، اپ هنگام اجرا خطا می‌داد.**

### فرمت فایل `.npy`
فایل با فرمت استاندارد NumPy ذخیره می‌شود:

```
magic   : 6 بایت  "\x93NUMPY"
version : 2 بایت  (major, minor)
hlen    : 2 یا 4 بایت LE  (طول هدر)
header  : ASCII   {'descr': '<f4', 'fortran_order': False, 'shape': (1000, 512), }
data    : خام، row-major
```

`LutEmbedding` این هدر را کامل پارس می‌کند و از این dtypeها پشتیبانی می‌کند:
`<f4`، `>f4`، `<f8`، `>f8`، و انواع صحیح (`i1..i8`, `u1..u4`). ترتیب ستون‌محور
(`fortran_order=True`) هم پشتیبانی می‌شود.

### دو مسیر خواندن
| محیط | متد | توضیح |
|------|-----|-------|
| اندروید | `LutEmbedding.fromAssets(assets, "models/lut_weight.npy")` | از `AssetManager` می‌خواند (نه File مستقیم؛ فایل‌های assets در APK فشرده‌اند) |
| دسکتاپ/تست | `LutEmbedding.fromFile(Path)` | از مسیر فایل روی دیسک |

هر دو مسیر یک نسخهٔ چهارآرگومانی هم دارند که بُعد مورد انتظار را می‌گیرد
(`fromFile(path, expectedDim)` و `fromAssets(assets, path, expectedDim)`).

در صورت نبودن فایل یا نامعتبر بودن هدر، یک `IOException` با **پیام واضح فارسی**
پرتاب می‌شود (نه NullPointerException یا خطای مبهم).

### ساخت فایل با `tools/make_fixtures.py`

> ⚠️ **ابعاد درست برای این مدل `text_dim = 64` است.** فایل `lut_weight.npy` باید ابعاد
> `[50257 × 64]` (حدود ۱۳ مگابایت) داشته باشد. اگر با `--dim 1024` ساخته شود، حجمش به
> **۲۰۵ مگابایت** می‌رسد و `LutEmbedding` آن را با پیام «ابعاد ناهمخوان» رد می‌کند.

```bash
# ✅ دستور درست برای این مدل (ابعاد از model_contract.json خوانده می‌شود)
python3 tools/make_fixtures.py --vocab 50257 --dim 64

# یا فقط vocab را بدهید؛ dim به‌طور خودکار از model_contract.json (text_dim) می‌آید:
python3 tools/make_fixtures.py --vocab 50257

# همراه با یک reference_01.wav نمونه (برای تست دسکتاپ):
python3 tools/make_fixtures.py --vocab 50257 --dim 64 --wav
```

خروجی به‌طور پیش‌فرض در `app/src/main/assets/models/lut_weight.npy` نوشته می‌شود.
ابزار در پایان، **آفست واقعی شروع داده** در فایل npy و همخوانی ابعاد با `Dims.TEXT_DIM`
و `model_contract.json` را بررسی و گزارش می‌کند; اگر `--dim` با `text_dim` تناقض داشته باشد،
با خطا متوقف می‌شود (نه اینکه فایل ناهمخوان بسازد).

> **نکته:** این ابزار مقدار `text_dim` را در `model_contract.json` هم به‌روز می‌کند تا هر سه
> منبع حقیقت (Java، قرارداد، ابزار) همیشه همخوان بمانند.

> **نکتهٔ مهم:** جدول ساخته‌شده با این ابزار یک **fixture** است (مقادیر تصادفی کوچک)
> و فقط برای اجرای بدون‌خطا و تست ابعاد/پارس کافی است. برای کیفیت گفتار واقعی،
> جدول وزن‌های آموزش‌دیدهٔ مدل خودتان را جایگزین کنید.

---

## ۴.۶) نسخهٔ دسکتاپ قابل‌اجرا (بدون اندروید و بدون ONNX)

برای تست سریع pipeline روی کامپیوتر (یا همین سندباکس)، یک نسخهٔ دسکتاپ در پوشهٔ
`desktop/` قرار دارد که **واقعاً اجرا می‌شود** و یک WAV معتبر می‌سازد.

### اجرا

```bash
# ۱) کامپایل (کلاس‌های مشترک + دسکتاپ)
javac -d out $(find app/src/main/java desktop/src -name '*.java')

# ۲) اجرا
java -cp out com.pockettts.persian.DesktopMain \
     --lut app/src/main/assets/models/lut_weight.npy \
     --out out.wav \
     --text "سلام، این یک آزمایش است."
```

### آرگومان‌ها
| آرگومان | پیش‌فرض | توضیح |
|---------|---------|-------|
| `--lut` | `app/src/main/assets/models/lut_weight.npy` | مسیر جدول embedding |
| `--out` | `out.wav` | مسیر فایل WAV خروجی |
| `--text` | یک جملهٔ نمونه | متن فارسی ورودی |
| `--threads` | `4` | تعداد thread (برای حالت موازی) |
| `--serial` | — | اجبار به حالت سری |

### چه کاری انجام می‌دهد؟
1. `lut_weight.npy` را با `LutEmbedding` می‌خواند.
2. متن را با `Chunker` به chunk تقسیم می‌کند.
3. هر chunk را با `DesktopTtsEngine` سنتز می‌کند — توکن‌ها با `LutEmbedding` به
   embedding تبدیل و از آن‌ها pitch/شدت صدا استخراج می‌شود.
4. سکوت ابتدا/انتها را می‌برد (`speechBounds`) و مکث‌های مرده را فشرده می‌کند (`compressPauses`).
5. با `stitchWithGaps` به هم می‌چسباند و WAV می‌نویسد.
6. **هدر WAV و تعداد نمونه‌ها را اعتبارسنجی می‌کند.**

> این نسخه از یک «وُکودر هارمونیک ساده» استفاده می‌کند (نه مدل ONNX) تا pipeline
> بدون مدل سنگین قابل تست باشد. برای کیفیت واقعی، از `TtsEngine` با مدل‌های ONNX استفاده کنید.

---

## ۴.۷) تست ابعاد LUT — `LutDimTest`

یک تست دسکتاپ که بررسی می‌کند فایل درست خوانده می‌شود، فایل ناهمخوان رد می‌شود،
و نگاشت ایمن توکن کرش نمی‌کند:

```bash
javac -nowarn -d out $(find desktop/stubs app/src/main/java desktop/src -name '*.java' ! -name 'MainActivity.java')
java -cp out com.pockettts.persian.LutDimTest fixtures/lut_weight.npy fixtures/lut_bad_dim.npy
```

خروجی مورد انتظار: `۴ موفق، ۰ ناموفق`.

---

## ۵) ساختار پروژه

```
pocket-tts-android/
├── settings.gradle
├── build.gradle
├── gradle.properties
└── app/
    ├── build.gradle
    ├── proguard-rules.pro
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/
        │   ├── plan.tsv
        │   └── models/README.txt
        ├── res/
        │   ├── layout/activity_main.xml
        │   └── values/{strings,colors,themes}.xml
        └── java/com/pockettts/persian/
            ├── MainActivity.java          # UI + سوییچ سری/موازی + زمان‌سنجی
            ├── TtsEngine.java             # هستهٔ موتور (ChunkSynthesizer)
            ├── ChunkSynthesizer.java      # قرارداد سنتز یک chunk
            ├── ParallelSynthesizer.java   # ★ حالت موازی (ExecutorService)
            ├── SynthesisResult.java       # ★ صدا + آمار زمان‌بندی
            ├── Chunker.java               # v8
            ├── PlanLoader.java            # خواندن plan.tsv
            ├── AudioUtils.java            # DSP + WAV
            ├── LutEmbedding.java          # ★ خواندن lut_weight.npy (v3)
            ├── G2P.java
            └── TokTest.java

desktop/                               # ★ نسخهٔ دسکتاپ قابل‌اجرا (v3)
└── src/com/pockettts/persian/
    ├── DesktopMain.java             # نقطهٔ ورود + اعتبارسنجی WAV
    └── DesktopTtsEngine.java        # موتور سبک (بدون ONNX)

tools/
└── make_fixtures.py                   # ★ ساخت lut_weight.npy و reference_01.wav (v3)
```

---

## ۶) اصلاحات کلیدی اعمال‌شده

| # | اصلاح | محل |
|---|-------|-----|
| ۱ | `framesAfterEos = 1` (نه ۵) | `TtsEngine.FRAMES_AFTER_EOS` |
| ۲ | `max_steps` بر پایهٔ `tokens/5.0` + حاشیه | `TtsEngine.maxStepsFor` |
| ۳ | bookkeeping درست `pos` و مقداردهی `seq/prev` با صفر | `TtsEngine.generateChunk` |
| ۴ | تولید هر chunk از حالت صدای دست‌نخورده | `TtsEngine.synthesizeChunk` |
| ۵ | پورت `_generate_with_retry` (۳ تلاش + معیار کیفیت) | `TtsEngine.generateWithRetry` |
| ۶ | پورت `_compress_pauses` و `_speech_bounds` | `AudioUtils` |
| ۷ | حذف median-RMS؛ gain سراسری ۰.۷–۱.۴ | `TtsEngine.applyGlobalGain` |
| ۸ | Chunker v8 (تفکیک SOFT/HARD، tok با fallback) | `Chunker` |
| ۹ | loadPlan اصلاح‌شده (اعتبارسنجی، کامنت، BOM/CRLF) | `PlanLoader` |
| ۱۰ | **حالت موازی با ExecutorService + ترتیب حفظ‌شده** | `ParallelSynthesizer` |
| ۱۱ | **هر thread وضعیت خودش + fallback سری + مدیریت خطا** | `ParallelSynthesizer` |
| ۱۲ | **خواندن `lut_weight.npy` (پارس کامل فرمت npy)** | `LutEmbedding` |
| ۱۳ | **ابزار ساخت fixture جدول embedding** | `tools/make_fixtures.py` |
| ۱۴ | **نسخهٔ دسکتاپ قابل‌اجرا + اعتبارسنجی WAV** | `desktop/` |

---

## ۷) تأیید صحت

کد پروژه در محیط headless با استاب‌های ONNX و Android کامپایل شد و تست‌های رفتاری حالت موازی اجرا شدند:

| تست | نتیجه |
|-----|-------|
| کامپایل کامل (۱۰ کلاس) | ✅ بدون خطا |
| ترتیب اصلی خروجی (۱۰۰،۲۰۰،۳۰۰،۴۰۰) | ✅ حفظ شد |
| مسیر موازی فعال شد | ✅ workers=4 |
| خطای یک chunk، بقیه سالم | ✅ failures=1، صدا تولید شد |
| fallback سری | ✅ کار کرد |
| طول کل با ترتیب و gapها | ✅ ۴۶۰۰ = ۴۶۰۰ |
| `applyGlobalGain` در بازهٔ ۰.۷–۱.۴ | ✅ ratio=1.400 |

---

## ۸) نکات

- پروژه کاملاً **آفلاین** است؛ هیچ دسترسی شبکه‌ای لازم نیست.
- برای دقت بالاتر G2P و توکنایزر، `G2P.java` و `TokTest.java` را با
  پیاده‌سازی واقعی مدل خود جایگزین کنید.
- امضای ورودی/خروجی مدل‌های ONNX در `TtsEngine.flow` و `decodeAll` بر اساس
  نام‌های رایج تنظیم شده؛ در صورت تفاوت، نام‌های tensor را مطابق مدل خود تغییر دهید.
