package com.pockettts.persian;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * TtsEngine — موتور تبدیل متن به گفتار فارسی آفلاین (Java + ONNX Runtime).
 * ============================================================
 * پایپ‌لاین: متن فارسی → G2P → واج‌ها → LUT embedding → FlowLM → Mimi decoder → WAV
 *
 * مدل‌ها:
 *   - flow_lm_step.onnx          : تولید latentها از واج‌ها (FlowLM)
 *   - mimi_decoder_step_kv.onnx  : تبدیل latent به نمونهٔ صوتی (Mimi decoder)
 *   - mimi_encoder.onnx          : ساخت voice cache از صدای مرجع
 *
 * ثابت‌ها: CAP=384، TEMP=0.3، voice_off=64، نرخ فریم Mimi ≈ 12.5Hz (۸۰ms)
 *
 * ── اصلاحات اعمال‌شده (نسبت به نسخهٔ معیوب) ──────────────────
 *   1) framesAfterEos = 1 (نه ۵ فریم اضافه) → جلوگیری از کشیدگی/شتابزدگی.
 *   2) max_steps بر پایهٔ tps واقعی (tokens/5.0 + حاشیه) نه tokens/3.0.
 *   3) bookkeeping درست offset/pos و مقداردهی seq/prev با صفر (نه NaN).
 *   4) تولید هر chunk از حالت صدای دست‌نخورده (pristine voice state).
 *   5) پورت _generate_with_retry با معیارهای کیفیت و انتخاب بهترین تلاش.
 *   6) پورت _compress_pauses و _speech_bounds.
 *   7) حذف نرمال‌سازی median-RMS بین chunkها؛ به‌جای آن gain سراسری ۰.۷–۱.۴.
 * ============================================================
 */
public class TtsEngine implements ChunkSynthesizer {

    // ---------------------------------------------------------
    // ثابت‌های مدل
    // ---------------------------------------------------------
    public static final int SAMPLE_RATE = 24000;   // نرخ نمونه‌برداری Mimi
    public static final int CAP = 384;             // حداکثر طول context
    public static final float TEMP = 0.3f;         // دمای نمونه‌برداری
    public static final int VOICE_OFF = 64;        // طول voice cache (۶۴ توکن)
    public static final int LDIM = 512;            // بُعد latent
    public static final double FRAME_RATE = 12.5;  // نرخ فریم Mimi (Hz) → ۸۰ms

    /** نرخ واقعی توکن به ثانیه (پایتون: ۴–۷). مقدار ۵ محافظه‌کارانه است. */
    public static final double TPS_EST = 5.0;

    /** حاشیهٔ اطمینان برای سقف max_steps. */
    public static final double MAX_STEPS_MARGIN = 1.6;

    /** تعداد فریم‌های اضافه بعد از EOS (اصلاح‌شده: ۱ به‌جای ۵). */
    public static final int FRAMES_AFTER_EOS = 1;

    /** کف gain سراسری (جلوگیری از تضعیف شدید قطعه‌های کوتاه). */
    public static final double GLOBAL_GAIN_MIN = 0.7;

    /** سقف gain سراسری (جلوگیری از تقویت شدید و «پمپاژ بلندی»). */
    public static final double GLOBAL_GAIN_MAX = 1.4;

    // ---------------------------------------------------------
    // آستانه‌های کیفیت (پارامتری‌شده)
    // ---------------------------------------------------------
    public static final class Quality {
        public double rmsFloor = 0.02;        // کف مطلق بلندی
        public double solidMin = 0.55;        // حداقل کسر گفتار پیوسته
        public double durLow = 0.30;          // کران پایین مدت گفتار (نسبت به انتظار)
        public double durHigh = 1.60;         // کران بالای مدت گفتار
        public double deadAirMax = 0.35;      // حداکثر سکوت مردهٔ مجاز (ثانیه)
        public double silenceRms = 0.01;      // آستانهٔ سکوت
        public double maxPause = 0.30;        // حداکثر مکث قبل از فشرده‌سازی
        public double keepPause = 0.12;       // مقدار نگه‌داشته‌شده از مکث
        public int attempts = 3;              // تعداد تلاش‌ها
    }

    private final Quality quality = new Quality();

    // ---------------------------------------------------------
    // وضعیت ONNX
    // ---------------------------------------------------------
    private OrtEnvironment env;
    private OrtSession flowSession;
    private OrtSession decoderSession;
    private OrtSession encoderSession;
    private boolean ready = false;

    private final Path modelDir;

    /** جدول embedding توکن‌های متنی (اختیاری؛ اگر نباشد seq با صفر مقداردهی می‌شود). */
    private LutEmbedding lut;

    public TtsEngine(Path modelDir) {
        this.modelDir = modelDir;
    }

    /**
     * بارگذاری جدول embedding از مسیر فایل (دسکتاپ/تست).
     *
     * <p>اگر این متد صدا زده نشود، موتور همچنان کار می‌کند و {@code seq} با
     * بردار صفر مقداردهی می‌شود؛ اما با بارگذاری LUT، زمینهٔ متنی هر chunk
     * (میانگین embedding توکن‌ها) به‌جای صفر به مدل داده می‌شود.</p>
     */
    /**
     * بُعد مورد انتظار embedding متن.
     *
     * <p>ابتدا تلاش می‌کند {@code model_contract.json} را از پوشهٔ مدل بخواند؛ اگر فایل
     * موجود نبود یا خوانده نشد، به {@link Dims#TEXT_DIM} برمی‌گردد. بنابراین فقط همین یک
     * جا باید به‌روز شود تا کل پروژه هماهنگ بماند.</p>
     */
    public int expectedTextDim() {
        Path contract = modelDir.resolve("model_contract.json");
        if (Files.exists(contract)) {
            try {
                String json = new String(Files.readAllBytes(contract), java.nio.charset.StandardCharsets.UTF_8);
                java.util.regex.Matcher m = java.util.regex.Pattern
                        .compile("\"text_dim\"\\s*:\\s*(\\d+)").matcher(json);
                if (m.find()) {
                    int d = Integer.parseInt(m.group(1));
                    if (d > 0) {
                        return d;
                    }
                }
            } catch (Exception ignore) {
                // اگر قرارداد خوانده نشد، به Dims.TEXT_DIM برمی‌گردیم (رفتار ایمن)
            }
        }
        return Dims.TEXT_DIM;
    }

    /**
     * بارگذاری جدول embedding از مسیر فایل (دسکتاپ/تست) + اعتبارسنجی ابعاد.
     *
     * @throws IOException اگر فایل نباشد، هدر خراب باشد، یا بُعد با قرارداد مدل ناهمخوان باشد
     */
    public void loadLut(Path lutPath) throws IOException {
        this.lut = LutEmbedding.fromFile(lutPath, expectedTextDim());
    }

    /** بارگذاری جدول embedding از assets اندروید + اعتبارسنجی ابعاد. */
    public void loadLutFromAssets(android.content.res.AssetManager assets, String assetPath)
            throws IOException {
        this.lut = LutEmbedding.fromAssets(assets, assetPath, expectedTextDim());
    }

    /** جدول embedding بارگذاری‌شده (یا null). */
    public LutEmbedding lut() {
        return lut;
    }

    /** بارگذاری مدل‌های ONNX. */
    public synchronized void load() throws OrtException, IOException {
        if (ready) {
            return;
        }
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));

        flowSession = env.createSession(resolve("flow_lm_step.onnx").toString(), opts);
        decoderSession = env.createSession(resolve("mimi_decoder_step_kv.onnx").toString(), opts);
        // انکودر اختیاری است (فقط برای ساخت voice cache از WAV)
        try {
            encoderSession = env.createSession(resolve("mimi_encoder.onnx").toString(), opts);
        } catch (Exception e) {
            System.err.println("[TtsEngine] انکودر Mimi بارگذاری نشد (اختیاری): " + e);
        }
        ready = true;
    }

    public boolean isReady() {
        return ready;
    }

    private Path resolve(String name) throws IOException {
        Path p = modelDir.resolve(name);
        if (!Files.exists(p)) {
            throw new IOException("فایل مدل یافت نشد: " + p);
        }
        return p;
    }

    // ---------------------------------------------------------
    // Voice cache
    // ---------------------------------------------------------

    /** ساخت voice cache از یک فایل WAV مرجع (۶۴ توکن). */
    public float[][] voiceCache(Path referenceWav) throws OrtException, IOException {
        float[] pcm = AudioUtils.readWav(referenceWav);
        if (encoderSession == null) {
            // اگر انکودر نبود، یک cache صفر برمی‌گردانیم تا پایپ‌لاین متوقف نشود.
            System.err.println("[TtsEngine] انکودر در دسترس نیست؛ voice cache صفر ساخته شد.");
            return new float[VOICE_OFF][LDIM];
        }
        long[] shape = {1, 1, pcm.length};
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(pcm), shape)) {
            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("audio", in);
            try (OrtSession.Result res = encoderSession.run(feeds)) {
                float[][][] codes = (float[][][]) res.get(0).getValue();
                // برش/پدینگ به ۶۴ توکن
                float[][] cache = new float[VOICE_OFF][LDIM];
                int n = Math.min(VOICE_OFF, codes[0].length);
                for (int i = 0; i < n; i++) {
                    System.arraycopy(codes[0][i], 0, cache[i], 0, Math.min(LDIM, codes[0][i].length));
                }
                return cache;
            }
        }
    }

    // ---------------------------------------------------------
    // تولید
    // ---------------------------------------------------------

    /** خروجی یک گام FlowLM. */
    private static final class FlowOut {
        float[] lat;
        boolean eos;
    }

    /**
     * محاسبهٔ سقف گام‌ها بر پایهٔ نرخ واقعی توکن/ثانیه.
     * اصلاح‌شده: tokens/5.0 (نه tokens/3.0) با حاشیهٔ اطمینان.
     */
    static int maxStepsFor(int tokens) {
        double expectedFrames = (tokens / TPS_EST) * FRAME_RATE;
        return (int) Math.ceil(expectedFrames * MAX_STEPS_MARGIN) + 4;
    }

    /**
     * تولید latentهای یک chunk از حالت صدای دست‌نخورده.
     *
     * @param voiceCache cache صدای مرجع (کپی می‌شود تا دست‌نخورده بماند)
     * @param voiceOff   آفست شروع (۶۴)
     * @param chunk      متن chunk
     * @param rng        مولد نویز مستقل این chunk
     */
    private List<float[]> generateChunk(float[][] voiceCache, int voiceOff,
                                        String chunk, Random rng) throws OrtException {
        int[] tokens = TokTest.encode(chunk);
        int max = maxStepsFor(tokens.length);

        // ── مقداردهی اولیه: seq و prev با صفر (نه NaN) ──
        // اگر LUT بارگذاری شده باشد، seq = میانگین embedding توکن‌های این chunk
        // (زمینهٔ متنی)؛ در غیر این صورت بردار صفر.
        float[] seq = (lut != null) ? lut.meanEmbeddingSafe(tokens) : new float[LDIM];
        float[] prev = new float[LDIM];         // بردار صفر
        int pos = voiceOff;                     // موقعیت جاری در context

        List<float[]> latents = new ArrayList<>();

        // گام نخست: با متن (واج‌ها) به‌عنوان ورودی
        float[] noise = gaussianNoise(rng);
        FlowOut z = flow(voiceCache, pos, seq, tokens, noise);
        latents.add(z.lat);
        prev = z.lat;
        pos += 1 + tokens.length;               // bookkeeping هماهنگ

        Integer eosStep = null;
        for (int step = 0; step < max; step++) {
            noise = gaussianNoise(rng);
            z = flow(voiceCache, pos, prev, new int[0], noise);

            if (z.eos && eosStep == null) {
                eosStep = step;
            }
            // اصلاح‌شده: فقط ۱ فریم بعد از EOS (نه ۵)
            if (eosStep != null && step >= eosStep + FRAMES_AFTER_EOS) {
                break;
            }
            latents.add(z.lat);
            prev = z.lat;
            pos++;                              // یک latent = یک موقعیت
        }
        return latents;
    }

    /** یک گام FlowLM (پوشش روی ONNX). */
    private FlowOut flow(float[][] voiceCache, int pos, float[] prev,
                         int[] tokens, float[] noise) throws OrtException {
        // ورودی‌ها بر اساس امضای مدل flow_lm_step.onnx
        long[] cacheShape = {1, voiceCache.length, LDIM};
        float[] flatCache = new float[voiceCache.length * LDIM];
        for (int i = 0; i < voiceCache.length; i++) {
            System.arraycopy(voiceCache[i], 0, flatCache, i * LDIM, LDIM);
        }
        long[] tokShape = {1, Math.max(1, tokens.length)};
        float[] tokF = new float[Math.max(1, tokens.length)];
        for (int i = 0; i < tokens.length; i++) {
            tokF[i] = tokens[i];
        }

        Map<String, OnnxTensor> feeds = new HashMap<>();
        try (OnnxTensor tCache = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatCache), cacheShape);
             OnnxTensor tPrev = OnnxTensor.createTensor(env, FloatBuffer.wrap(prev), new long[]{1, LDIM});
             OnnxTensor tTok = OnnxTensor.createTensor(env, FloatBuffer.wrap(tokF), tokShape);
             OnnxTensor tNoise = OnnxTensor.createTensor(env, FloatBuffer.wrap(noise), new long[]{1, LDIM});
             OnnxTensor tPos = OnnxTensor.createTensor(env, new long[]{pos})) {

            feeds.put("voice_cache", tCache);
            feeds.put("prev", tPrev);
            feeds.put("tokens", tTok);
            feeds.put("noise", tNoise);
            feeds.put("pos", tPos);

            try (OrtSession.Result res = flowSession.run(feeds)) {
                FlowOut out = new FlowOut();
                // API واقعی ONNX Runtime: get(String) یک Optional<OnnxValue> برمی‌گرداند.
                float[][] lat = (float[][]) res.get("latent").get().getValue();
                out.lat = lat[0].clone();
                // EOS: اگر مدل خروجی eos دارد از آن استفاده کن، وگرنه از آستانه
                if (res.get("eos").isPresent()) {
                    float[][] e = (float[][]) res.get("eos").get().getValue();
                    out.eos = e[0][0] > 0.5f;
                } else {
                    out.eos = false;
                }
                return out;
            }
        }
    }

    /** دیکود همهٔ latentها به نمونهٔ صوتی. */
    private float[] decodeAll(List<float[]> latents) throws OrtException {
        List<Float> out = new ArrayList<>();
        float[] state = new float[LDIM];
        for (float[] lat : latents) {
            Map<String, OnnxTensor> feeds = new HashMap<>();
            try (OnnxTensor tLat = OnnxTensor.createTensor(env, FloatBuffer.wrap(lat), new long[]{1, LDIM});
                 OnnxTensor tState = OnnxTensor.createTensor(env, FloatBuffer.wrap(state), new long[]{1, LDIM})) {
                feeds.put("latent", tLat);
                feeds.put("state", tState);
                try (OrtSession.Result res = decoderSession.run(feeds)) {
                    float[][] audio = (float[][]) res.get("audio").get().getValue();
                    for (float s : audio[0]) {
                        out.add(s);
                    }
                    if (res.get("state_out").isPresent()) {
                        float[][] st = (float[][]) res.get("state_out").get().getValue();
                        state = st[0].clone();
                    }
                }
            }
        }
        float[] arr = new float[out.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = out.get(i);
        }
        return arr;
    }

    // ---------------------------------------------------------
    // تولید با retry (پورت _generate_with_retry)
    // ---------------------------------------------------------

    /**
     * تولید یک chunk با تلاش مجدد و انتخاب بهترین تلاش بر اساس معیارهای کیفیت.
     * معیارها: rms مطلق، کسر گفتار پیوسته، مدت گفتار واقعی، بازه‌های سکوت مرده.
     */
    private float[] generateWithRetry(float[][] voiceCache, int voiceOff,
                                      String chunk, Random rng) throws OrtException {
        int tokens = TokTest.encode(chunk).length;
        double expectedSpeech = tokens / TPS_EST;   // ثانیه (محافظه‌کارانه)

        float[] best = null;
        double[] bestKey = null;

        for (int attempt = 0; attempt < quality.attempts; attempt++) {
            // هر تلاش با rng مستقل (نه مشترک) تا نویز همبسته نباشد
            Random attemptRng = new Random(rng.nextLong());
            List<float[]> latents = generateChunk(voiceCache, voiceOff, chunk, attemptRng);
            float[] audio = decodeAll(latents);

            // برش سکوت ابتدا/انتها
            int[] b = AudioUtils.speechBounds(audio, quality.silenceRms);
            float[] speech = slice(audio, b[0], b[1]);
            // فشرده‌سازی مکث‌های مرده
            speech = AudioUtils.compressPauses(speech, quality.maxPause, quality.keepPause);

            double dur = speech.length / (double) SAMPLE_RATE;
            double rms = AudioUtils.rms(speech);
            boolean loudOk = rms >= quality.rmsFloor;
            double frac = AudioUtils.solidFraction(speech, quality.silenceRms);
            double solidDur = frac * dur;
            boolean durOk = solidDur >= quality.durLow * expectedSpeech
                    && solidDur <= quality.durHigh * expectedSpeech;
            List<int[]> regions = AudioUtils.deadAirRegions(speech, 0.10);
            double dead = 0.0;
            for (int[] r : regions) {
                dead = Math.max(dead, (r[1] - r[0]) / (double) SAMPLE_RATE);
            }

            // کلید ترتیب اولویت: بلند و درست > ساکت > زباله
            double[] key = {
                loudOk ? 1 : 0,
                durOk ? 1 : 0,
                solidDur,
                dead <= quality.deadAirMax ? 0 : -dead
            };
            if (bestKey == null || compare(key, bestKey) > 0) {
                best = speech;
                bestKey = key;
            }
            if (loudOk && durOk && frac >= quality.solidMin && dead <= quality.deadAirMax) {
                break;   // تلاش تمیز؛ نیازی به ادامه نیست
            }
        }
        // جلوگیری از بازگشت null
        if (best == null) {
            best = new float[0];
        }
        return best;
    }

    // ---------------------------------------------------------
    // سنتز کامل
    // ---------------------------------------------------------

    /**
     * سنتز یک متن کامل به نمونهٔ صوتی.
     *
     * @param text متن فارسی
     * @param voiceCache cache صدای مرجع
     * @param seed دانهٔ تصادفی (اختیاری)
     */
    public float[] synthesize(String text, float[][] voiceCache, Long seed) throws OrtException {
        if (!ready) {
            throw new IllegalStateException("موتور بارگذاری نشده است؛ ابتدا load() را صدا بزنید.");
        }
        List<String> chunks = Chunker.chunk(text);
        if (chunks.isEmpty()) {
            return new float[0];
        }
        Random rng = (seed != null) ? new Random(seed) : new Random();

        List<float[]> audios = new ArrayList<>();
        for (String chunk : chunks) {
            // هر chunk از حالت صدای دست‌نخورده تولید می‌شود
            audios.add(generateWithRetry(voiceCache, VOICE_OFF, chunk, rng));
        }

        // ── gain سراسری محافظه‌کارانه (۰.۷ تا ۱.۴) به‌جای median-RMS بین chunkها ──
        applyGlobalGain(audios);

        // چسباندن با مکث پیش‌فرض بین chunkها
        return stitch(audios, 0.12);
    }

    /** چسباندن قطعه‌های صوتی با مکث مشخص. */
    private float[] stitch(List<float[]> audios, double gapSeconds) {
        int gap = (int) (gapSeconds * SAMPLE_RATE);
        int total = 0;
        for (float[] a : audios) {
            total += a.length;
        }
        total += gap * Math.max(0, audios.size() - 1);
        float[] out = new float[total];
        int cursor = 0;
        for (int i = 0; i < audios.size(); i++) {
            float[] a = audios.get(i);
            System.arraycopy(a, 0, out, cursor, a.length);
            cursor += a.length;
            if (i < audios.size() - 1) {
                cursor += gap;   // سکوت بین قطعه‌ها
            }
        }
        return out;
    }

    // ---------------------------------------------------------
    // کمکی‌ها
    // ---------------------------------------------------------

    // ---------------------------------------------------------
    // API عمومی برای حالت موازی (ParallelSynthesizer)
    // ---------------------------------------------------------

    /**
     * کپی عمیق voice cache.
     *
     * <p>هر thread پیش از شروع کار، یک نسخهٔ کاملاً مستقل می‌گیرد تا
     * «حالت صدای دست‌نخورده» (pristine voice state) برای همهٔ chunkها
     * حفظ شود و هیچ worker‌ای روی دادهٔ مشترک ننویسد.</p>
     */
    public static float[][] copyVoiceCache(float[][] src) {
        if (src == null) {
            return new float[0][];
        }
        float[][] out = new float[src.length][];
        for (int i = 0; i < src.length; i++) {
            out[i] = (src[i] == null) ? null : src[i].clone();
        }
        return out;
    }

    /**
     * سنتز یک chunk از حالت صدای دست‌نخورده — نقطهٔ ورود حالت موازی.
     *
     * <p>هر فراخوانی مستقل و بدون حالت مشترک است: کپی محلی voice cache،
     * مولّد نویز محلی، و اگر {@code attempts<=0} باشد خروجی خالی برمی‌گردد
     * (نه {@code null}).</p>
     */
    @Override
    public float[] synthesizeChunk(String chunk, float[][] pristineVoiceCache, long seed)
            throws OrtException {
        if (!ready) {
            throw new IllegalStateException("موتور بارگذاری نشده است؛ ابتدا load() را صدا بزنید.");
        }
        if (chunk == null || chunk.trim().isEmpty()) {
            return new float[0];
        }
        float[][] local = copyVoiceCache(pristineVoiceCache);
        return generateWithRetry(local, VOICE_OFF, chunk, new Random(seed));
    }

    /** ساخت فهرست job از یک متن خام (برای UI). */
    public static List<PlanLoader.Job> jobsFromText(String text) {
        List<PlanLoader.Job> jobs = new ArrayList<>();
        for (String chunk : Chunker.chunk(text)) {
            jobs.add(new PlanLoader.Job(chunk,
                    jobs.isEmpty() ? 0.0 : PlanLoader.DEFAULT_INNER_GAP, -1L));
        }
        return jobs;
    }

    /** استخراج آرایهٔ gap از فهرست job. */
    public static double[] gapsOf(List<PlanLoader.Job> jobs) {
        double[] gaps = new double[jobs.size()];
        for (int i = 0; i < jobs.size(); i++) {
            gaps[i] = jobs.get(i).gap;
        }
        return gaps;
    }

    /**
     * چسباندن قطعه‌های صوتی با مکث <b>متغیر</b> برای هر قطعه.
     *
     * <p>برخلاف {@link #stitch(List, double)} که یک مکث ثابت می‌گیرد،
     * اینجا هر قطعه مکث خودش را دارد (حاصل از ستون gap در plan.tsv).
     * مقدار gap قطعهٔ اول نادیده گرفته می‌شود (چون مکثی قبل از آن نیست).</p>
     */
    public static float[] stitchWithGaps(List<float[]> audios, double[] gaps) {
        if (audios == null || audios.isEmpty()) {
            return new float[0];
        }
        long total = 0;
        for (int i = 0; i < audios.size(); i++) {
            float[] a = audios.get(i);
            total += (a == null) ? 0 : a.length;
            if (i > 0) {
                double g = (gaps != null && i < gaps.length) ? gaps[i] : 0.12;
                total += Math.max(0, (int) (g * SAMPLE_RATE));
            }
        }
        float[] out = new float[(int) total];
        int cursor = 0;
        for (int i = 0; i < audios.size(); i++) {
            float[] a = audios.get(i);
            if (i > 0) {
                double g = (gaps != null && i < gaps.length) ? gaps[i] : 0.12;
                cursor += Math.max(0, (int) (g * SAMPLE_RATE));   // مکث قبل از این قطعه
            }
            if (a != null && a.length > 0) {
                System.arraycopy(a, 0, out, cursor, a.length);
                cursor += a.length;
            }
        }
        return out;
    }

    /**
     * gain سراسری محافظه‌کارانه ۰.۷ تا ۱.۴.
     *
     * <p>جایگزین نرمال‌سازی median-RMS بین chunkها: به‌جای اینکه هر chunk
     * جداگانه تا ۲.۵ برابر تقویت شود (که باعث «پمپاژ بلندی» بین قطعه‌ها
     * می‌شد)، یک هدف سراسری گرفته و gain در محدودهٔ باریک مهار می‌شود.
     * این کار فقط <b>بلندی</b> را تنظیم می‌کند و به سرعت/کشیدگی صدا
     * کاری ندارد.</p>
     */
    public static void applyGlobalGain(List<float[]> audios) {
        if (audios == null || audios.isEmpty()) {
            return;
        }
        double[] rmsArr = new double[audios.size()];
        for (int i = 0; i < audios.size(); i++) {
            rmsArr[i] = AudioUtils.rms(audios.get(i));
        }
        double target = AudioUtils.median(rmsArr);
        if (target <= 1e-6) {
            return;
        }
        for (float[] a : audios) {
            if (a == null || a.length == 0) {
                continue;
            }
            double g = target / Math.max(1e-6, AudioUtils.rms(a));
            g = Math.max(GLOBAL_GAIN_MIN, Math.min(GLOBAL_GAIN_MAX, g));
            for (int i = 0; i < a.length; i++) {
                a[i] = (float) (a[i] * g);
            }
        }
    }

    /** دانهٔ تکرارپذیر برای chunk شمارهٔ {@code index} (وقتی seed صریح ندارد). */
    public static long seedFor(PlanLoader.Job job, int index) {
        if (job != null && job.seed >= 0) {
            return job.seed;
        }
        return 20261015L + index;   // تکرارپذیر و مستقل برای هر chunk
    }

    /** سنتز سری (تک‌thread) — مرجع مقایسهٔ حالت موازی. */
    public SynthesisResult synthesizeSerial(List<PlanLoader.Job> jobs,
                                            float[][] pristineVoiceCache) throws OrtException {
        long t0 = System.currentTimeMillis();
        List<float[]> audios = new ArrayList<>();
        for (int i = 0; i < jobs.size(); i++) {
            PlanLoader.Job job = jobs.get(i);
            float[][] local = copyVoiceCache(pristineVoiceCache);
            audios.add(generateWithRetry(local, VOICE_OFF, job.text,
                    new Random(seedFor(job, i))));
        }
        applyGlobalGain(audios);
        float[] audio = stitchWithGaps(audios, gapsOf(jobs));
        return new SynthesisResult(audio, System.currentTimeMillis() - t0,
                1, false, 0, jobs.size());
    }

    private static float[] slice(float[] x, int from, int to) {
        from = Math.max(0, from);
        to = Math.min(x.length, to);
        if (to <= from) {
            return new float[0];
        }
        float[] out = new float[to - from];
        System.arraycopy(x, from, out, 0, out.length);
        return out;
    }

    private static float[] gaussianNoise(Random rng) {
        float[] n = new float[LDIM];
        for (int i = 0; i < LDIM; i++) {
            n[i] = (float) (rng.nextGaussian() * Math.sqrt(TEMP));
        }
        return n;
    }

    /** مقایسهٔ لغوی دو کلید کیفیت. */
    private static int compare(double[] a, double[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            int c = Double.compare(a[i], b[i]);
            if (c != 0) {
                return c;
            }
        }
        return 0;
    }

    /** آزادسازی منابع ONNX. */
    public synchronized void close() {
        try { if (flowSession != null) flowSession.close(); } catch (Exception ignored) { }
        try { if (decoderSession != null) decoderSession.close(); } catch (Exception ignored) { }
        try { if (encoderSession != null) encoderSession.close(); } catch (Exception ignored) { }
        ready = false;
    }
}
