package com.pockettts.persian;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * ParallelSynthesizer — سنتز موازی chunk‌ها با ExecutorService.
 * ============================================================
 * ایدهٔ اصلی (الهام‌گرفته از بخش موازی پایتون):
 *   - chunk‌ها از هم <b>مستقل</b> هستند، چون هر کدام از «حالت صدای
 *     دست‌نخورده» (pristine voice state) تولید می‌شود.
 *   - پس می‌توان آن‌ها را هم‌زمان روی چند thread سنتز کرد.
 *
 * <p>سه تضمین کلیدی این کلاس — دقیقاً همان نکاتی که در پایتون
 * تحلیل شد:</p>
 *
 * <b>۱) ترتیب اصلی خروجی حفظ می‌شود.</b>
 * نتیجه‌ها به ترتیب <b>لیست</b> جمع‌آوری می‌شوند (نه به ترتیب
 * تمام‌شدن). یعنی {@code results[i]} همیشه به {@code jobs.get(i)}
 * تعلق دارد، حتی اگر chunk شمارهٔ ۷ زودتر از chunk شمارهٔ ۲ تمام شود.
 * سپس صداها به همان ترتیب اصلی به هم چسبانده می‌شوند.
 *
 * <b>۲) هر thread وضعیت مستقل خودش را دارد.</b>
 * به‌جای یک نمونهٔ مشترک و قفل کردن آن (که موازی‌سازی را عملاً سری
 * می‌کرد)، یک {@link ChunkSynthesizer} جداگانه به هر worker داده
 * می‌شود. هر فراخوانی هم داخل خودش از voice cache کپی می‌گیرد، پس
 * هیچ دو thread روی یک ناحیهٔ حافظه نمی‌نویسند.
 *
 * <b>۳) خطای یک chunk کل کار را متوقف نمی‌کند.</b>
 * خطای هر job جداگانه گرفته و گزارش می‌شود؛ آن chunk با سکوتِ هم‌طولِ
 * تخمینی جایگزین می‌شود تا زمان‌بندی بقیهٔ قطعه‌ها به‌هم نریزد.
 * ============================================================
 */
public final class ParallelSynthesizer {

    /** تعداد پیش‌فرض threadها. */
    public static final int DEFAULT_THREADS = 4;

    /** حداکثر مجاز thread (محافظت از حافظهٔ دستگاه). */
    public static final int MAX_THREADS = 8;

    /** کارخانهٔ ساخت نمونهٔ مستقل موتور برای هر worker. */
    public interface EngineFactory {
        /** یک نمونهٔ تازه بساز و بارگذاری کن. */
        ChunkSynthesizer create() throws Exception;
    }

    /** گزارش یک chunk شکست‌خورده. */
    public static final class Failure {
        public final int index;
        public final String text;
        public final Exception error;

        public Failure(int index, String text, Exception error) {
            this.index = index;
            this.text = text;
            this.error = error;
        }

        @Override
        public String toString() {
            return "chunk#" + index + " (\u00ab" + text + "\u00bb): " + error;
        }
    }

    /** نتیجهٔ سنتز موازی. */
    public static final class Outcome {
        public final float[] audio;
        public final SynthesisResult stats;
        public final List<Failure> failures;

        public Outcome(float[] audio, SynthesisResult stats, List<Failure> failures) {
            this.audio = audio;
            this.stats = stats;
            this.failures = failures;
        }
    }

    private final int threads;

    public ParallelSynthesizer() {
        this(DEFAULT_THREADS);
    }

    public ParallelSynthesizer(int threads) {
        this.threads = clampThreads(threads);
    }

    /** تعداد thread به‌کاررفته. */
    public int threads() {
        return threads;
    }

    /**
     * محدودسازی تعداد thread.
     * مقدار غیرمعتبر (کمتر از ۱) به ۱ برمی‌گردد تا حالت «سری» داشته باشیم.
     */
    public static int clampThreads(int t) {
        if (t < 1) {
            return 1;
        }
        return Math.min(t, MAX_THREADS);
    }

    /**
     * انتخاب خودکار تعداد thread بر اساس حافظهٔ آزاد.
     *
     * <p>تعداد worker بر اساس سقف heap (maxMemory) تعیین می‌شود، نه حافظهٔ
     * آزادِ لحظه‌ای که گمراه‌کننده است.</p>
     */
    public static int autoThreads(int requested) {
        int want = clampThreads(requested);
        Runtime rt = Runtime.getRuntime();
        long maxMb = rt.maxMemory() / (1024 * 1024);
        if (maxMb >= 512) {
            return want;   // حافظهٔ کافی → همان تعداد درخواستی
        }
        int affordable = (int) Math.max(1, maxMb / 96);   // ≈۹۶ مگابایت سرانه
        return Math.max(1, Math.min(want, affordable));
    }

    /**
     * سنتز موازی فهرست job‌ها.
     *
     * @param jobs        فهرست chunk‌ها به ترتیب نهایی (شامل gap هر chunk).
     * @param factory     کارخانهٔ ساخت نمونهٔ مستقل موتور برای هر worker.
     * @param fallback    نمونهٔ موتور اصلی؛ فقط برای اجرای سری و بازیابی
     *                    تعطیلی به‌کار می‌رود (می‌تواند {@code null} باشد).
     * @param requestedThreads تعداد thread درخواستی (پیش‌فرض ۴).
     * @return صدا + آمار + خطاها.
     */
    public Outcome synthesize(List<PlanLoader.Job> jobs, EngineFactory factory,
                              ChunkSynthesizer fallback, int requestedThreads) {
        if (jobs == null || jobs.isEmpty()) {
            return new Outcome(new float[0],
                    new SynthesisResult(new float[0], 0, 1, false, 0, 0),
                    new ArrayList<>());
        }

        // ← محافظ: بدون voice cache، هر chunk با cache خالی تولید می‌شد
        if (PRISTINE == null || PRISTINE.length == 0) {
            throw new IllegalStateException(
                    "voice cache تعیین نشده است؛ ابتدا withVoiceCache(...) را صدا بزنید.");
        }

        // ← برای اجرای سری به حداقل یک موتور متصل یا کارخانهٔ ساخت آن نیاز داریم
        if (fallback == null && factory == null) {
            throw new IllegalStateException(
                    "نه نمونهٔ موتور و نه کارخانهٔ ساخت آن داده نشده است.");
        }

        int workers = autoThreads(requestedThreads);
        List<Failure> failures = new ArrayList<>();

        // ── عقب‌نشینی به حالت سری ──
        // وقتی فقط یک thread در دسترس است، استخر سربار بی‌فایده است.
        if (workers <= 1) {
            return serial(jobs, fallback, factory, failures, false);
        }

        return parallel(jobs, factory, fallback, workers, failures);
    }

    // ---------------------------------------------------------
    // مسیر موازی
    // ---------------------------------------------------------

    private Outcome parallel(List<PlanLoader.Job> jobs, EngineFactory factory,
                             ChunkSynthesizer fallback, int workers,
                             List<Failure> failures) {
        long t0 = System.currentTimeMillis();
        int n = jobs.size();

        // نمونهٔ محلی برای مسیرهای عقب‌نشینی (اگر موتور اصلی داده نشده باشد)
        ChunkSynthesizer localFallback = fallback;
        if (localFallback == null && factory != null) {
            try {
                localFallback = factory.create();
            } catch (Exception e) {
                System.err.println("[ParallelSynthesizer] ساخت موتور محلی ناموفق: " + e);
            }
        }

        // هر worker نمونهٔ مستقل خودش را دارد → بدون قفل، بدون دادهٔ مشترک
        ThreadLocal<ChunkSynthesizer> engines = new ThreadLocal<ChunkSynthesizer>() {
            @Override
            protected ChunkSynthesizer initialValue() {
                try {
                    return factory.create();
                } catch (Exception e) {
                    System.err.println("[ParallelSynthesizer] ساخت موتور worker ناموفق: " + e);
                    return null;
                }
            }
        };

        // ست همگام: چند worker هم‌زمان نمونهٔ خود را ثبت می‌کنند
        Set<ChunkSynthesizer> created = Collections.newSetFromMap(new ConcurrentHashMap<>());
        ExecutorService pool = null;
        try {
            int poolSize = Math.min(workers, n);
            // صف کوچک: کارها از قبل ساخته شده‌اند و باید سریع شروع شوند
            pool = new ThreadPoolExecutor(
                    poolSize, poolSize,
                    30L, TimeUnit.SECONDS,
                    new ArrayBlockingQueue<Runnable>(Math.max(1, n)),
                    r -> {
                        Thread th = new Thread(r, "tts-worker");
                        th.setDaemon(true);
                        return th;
                    });

            // ── ثبت کارها به ترتیب LIست ──
            List<Future<float[]>> futures = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                final int idx = i;
                final PlanLoader.Job job = jobs.get(i);
                futures.add(pool.submit(new Callable<float[]>() {
                    @Override
                    public float[] call() {
                        ChunkSynthesizer eng = engines.get();
                        if (eng != null) {
                            created.add(eng);   // Set همگام؛ ثبت تکراری بی‌ضرر است
                        }
                        try {
                            if (eng == null) {
                                throw new IllegalStateException("worker engine unavailable");
                            }
                            // هر chunk از حالت صدای دست‌نخورده + seed مستقل
                            return eng.synthesizeChunk(job.text, PRISTINE, TtsEngine.seedFor(job, idx));
                        } catch (Exception e) {
                            // ★ خطای جزئی: کل کار متوقف نمی‌شود
                            synchronized (failures) {
                                failures.add(new Failure(idx, job.text, e));
                            }
                            return null;   // بعداً با سکوت جایگزین می‌شود
                        }
                    }
                }));
            }

            // ── جمع‌آوری به ترتیب LIست (نه به ترتیب تمام‌شدن) ──
            List<float[]> ordered = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                float[] a;
                try {
                    a = futures.get(i).get();     // ← ترتیب اصلی
                } catch (Exception e) {
                    synchronized (failures) {
                        failures.add(new Failure(i, jobs.get(i).text, e));
                    }
                    a = null;
                }
                ordered.add(a);
            }

            return assemble(ordered, jobs, failures, t0, poolSize, true);

        } catch (RejectedExecutionException ree) {
            // استخر نتوانست کارها را بپذیرد → عقب‌نشینی به سری
            System.err.println("[ParallelSynthesizer] استخر رد شد؛ عقب‌نشینی به حالت سری: " + ree);
            if (pool != null) {
                pool.shutdownNow();
            }
            return serial(jobs, localFallback, factory, failures, true);
        } finally {
            if (pool != null) {
                pool.shutdownNow();
            }
            for (ChunkSynthesizer c : created) {
                try {
                    c.close();
                } catch (Exception ignored) {
                }
            }
            // موتور محلیِ ساخته‌شده در همین متد باید آزاد شود
            if (fallback == null && localFallback != null) {
                try {
                    localFallback.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    // ---------------------------------------------------------
    // مسیر سری
    // ---------------------------------------------------------

    private Outcome serial(List<PlanLoader.Job> jobs, ChunkSynthesizer engine,
                           EngineFactory factory, List<Failure> failures, boolean wasFallback) {
        long t0 = System.currentTimeMillis();
        List<float[]> ordered = new ArrayList<>(jobs.size());

        // اگر موتور متصل نیست، یک نمونه از کارخانه بساز (و در پایان ببند)
        ChunkSynthesizer own = null;
        if (engine == null && factory != null) {
            try {
                own = factory.create();
                engine = own;
            } catch (Exception e) {
                System.err.println("[ParallelSynthesizer] ساخت موتور سری ناموفق: " + e);
            }
        }

        if (engine != null) {
            for (int i = 0; i < jobs.size(); i++) {
                PlanLoader.Job job = jobs.get(i);
                try {
                    ordered.add(engine.synthesizeChunk(job.text, PRISTINE, TtsEngine.seedFor(job, i)));
                } catch (Exception e) {
                    synchronized (failures) {
                        failures.add(new Failure(i, job.text, e));
                    }
                    ordered.add(null);
                }
            }
        }
        Outcome out = assemble(ordered, jobs, failures, t0, 1, false);
        // آزادسازی موتور محلی
        if (own != null) {
            try {
                own.close();
            } catch (Exception ignored) {
            }
        }
        // علامت‌گذاری اینکه این سری، نتیجهٔ عقب‌نشینی بوده
        if (wasFallback) {
            System.err.println("[ParallelSynthesizer] حالت سری (عقب‌نشینی) اعمال شد.");
        }
        return out;
    }

    // ---------------------------------------------------------
    // مونتاژ نهایی
    // ---------------------------------------------------------

    /** جایگزینی chunk‌های ناموفق با سکوت هم‌طول و چسباندن به ترتیب اصلی. */
    private Outcome assemble(List<float[]> ordered, List<PlanLoader.Job> jobs,
                             List<Failure> failures, long t0,
                             int workers, boolean parallel) {
        // طول تخمینی سکوت برای جایگزینی chunk خراب
        int silenceLen = (int) (0.25 * TtsEngine.SAMPLE_RATE);

        List<float[]> cleaned = new ArrayList<>(ordered.size());
        for (int i = 0; i < ordered.size(); i++) {
            float[] a = ordered.get(i);
            if (a == null || a.length == 0) {
                // سکوت جایگزین تا زمان‌بندی gapها به‌هم نریزد
                cleaned.add(new float[silenceLen]);
            } else {
                cleaned.add(a);
            }
        }

        // gain سراسری محافظه‌کارانه (نه median-RMS بین chunkها)
        TtsEngine.applyGlobalGain(cleaned);

        // چسباندن به ترتیب اصلی، با gap هر chunk از plan
        float[] audio = TtsEngine.stitchWithGaps(cleaned, TtsEngine.gapsOf(jobs));

        long ms = System.currentTimeMillis() - t0;
        SynthesisResult stats = new SynthesisResult(audio, ms, workers, parallel,
                failures.size(), jobs.size());
        return new Outcome(audio, stats, failures);
    }

    // ---------------------------------------------------------
    // voice cache دست‌نخورده برای همهٔ workerها
    // ---------------------------------------------------------

    /**
     * voice cache مشترک و <b>فقط‌خواندنی</b>.
     *
     * <p>این آرایه هرگز تغییر نمی‌کند؛ هر worker خودش کپی می‌گیرد
     * ({@link TtsEngine#copyVoiceCache(float[][])}) و روی کپی کار
     * می‌کند. پس اشتراک آن بین threadها امن است.</p>
     */
    private volatile float[][] PRISTINE;

    /** تعیین voice cache دست‌نخورده (یک‌بار، قبل از سنتز). */
    public ParallelSynthesizer withVoiceCache(float[][] pristine) {
        this.PRISTINE = TtsEngine.copyVoiceCache(pristine);
        return this;
    }
}
