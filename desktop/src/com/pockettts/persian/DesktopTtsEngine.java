package com.pockettts.persian;

import java.util.ArrayList;
import java.util.List;

/**
 * DesktopTtsEngine — موتور TTS سبک برای اجرای دسکتاپ (بدون اندروید و بدون ONNX).
 * ============================================================
 * <p>این کلاس همان قرارداد {@link ChunkSynthesizer} را پیاده می‌کند تا نسخهٔ
 * دسکتاپ بتواند <b>واقعاً روی همین سندباکس</b> اجرا شود و یک WAV معتبر بسازد،
 * حتی وقتی فایل‌های مدل ONNX در دسترس نیستند.</p>
 *
 * <p><b>نقش LutEmbedding:</b> توکن‌های هر chunk با {@link TokTest} استخراج و با
 * {@link LutEmbedding} به بردار embedding تبدیل می‌شوند. از این بردارها برای
 * تعیین «زیر و بمی» (pitch) و «شدت» (amplitude) هر هجا استفاده می‌شود؛ بنابراین
 * خروجی واقعاً به جدول embedding وابسته است (نه یک تُن ثابت).</p>
 *
 * <p><b>تفاوت با موتور واقعی:</b> در موتور واقعی، embedding به FlowLM و سپس
 * Mimi decoder داده می‌شود. اینجا یک «وُکودر سادهٔ هارمونیک» جایگزین شده تا
 * pipeline (توکن → embedding → صدا → برش سکوت → فشرده‌سازی مکث → WAV) بدون
 * مدل سنگین قابل تست باشد. برای کیفیت واقعی، {@link TtsEngine} را با مدل‌های
 * ONNX به‌کار ببرید.</p>
 * ============================================================
 */
public final class DesktopTtsEngine implements ChunkSynthesizer {

    private final LutEmbedding lut;
    private final int sampleRate;

    /** مدت پایهٔ هر توکن (ثانیه) — تقریبی از نرخ واقعی گفتار. */
    private static final double TOKEN_SECONDS = 0.11;

    public DesktopTtsEngine(LutEmbedding lut, int sampleRate) {
        this.lut = lut;
        this.sampleRate = sampleRate;
    }

    @Override
    public float[] synthesizeChunk(String chunk, float[][] pristineVoiceCache, long seed) {
        if (chunk == null || chunk.trim().isEmpty()) {
            return new float[0];
        }
        int[] tokens = TokTest.encode(chunk);
        if (tokens.length == 0) {
            return new float[0];
        }

        // ── ۱) تبدیل توکن‌ها به embedding (نقش LutEmbedding) ──
        float[][] emb = new float[tokens.length][];
        for (int i = 0; i < tokens.length; i++) {
            int id = Math.floorMod(tokens[i], lut.vocabSize());   // نگاشت ایمن به بازهٔ جدول
            emb[i] = lut.embed(id);
        }

        // ── ۲) تولید یک قطعهٔ صوتی برای هر توکن ──
        List<float[]> parts = new ArrayList<>();
        for (int i = 0; i < emb.length; i++) {
            parts.add(voicedSegment(emb[i], tokens[i], seed + i));
        }

        // ── ۳) چسباندن با مکث کوتاه بین توکن‌ها ──
        float[] raw = stitch(parts, 0.02);

        // ── ۴) برش سکوت ابتدا/انتها (پورت _speech_bounds) ──
        int[] b = AudioUtils.speechBounds(raw, 0.01);
        float[] speech = slice(raw, b[0], b[1]);

        // ── ۵) فشرده‌سازی مکث‌های مرده (پورت _compress_pauses) ──
        speech = AudioUtils.compressPauses(speech, 0.30, 0.12);

        return speech;
    }

    /**
     * تولید یک قطعهٔ صوتی هارمونیک برای یک توکن.
     * pitch از میانگین embedding و شدت از نُرم آن گرفته می‌شود.
     */
    private float[] voicedSegment(float[] e, int tokenId, long seed) {
        // میانگین و نُرم بردار embedding → پارامترهای صدا
        double mean = 0.0, norm = 0.0;
        for (float v : e) {
            mean += v;
            norm += (double) v * v;
        }
        mean /= e.length;
        norm = Math.sqrt(norm / e.length);

        // pitch در بازهٔ ۹۰..۲۲۰ هرتز (صدای انسان)
        double pitch = 90.0 + 130.0 * sigmoid(mean * 6.0);
        // شدت در بازهٔ ۰.۲۵..۰.۸۵
        double amp = 0.25 + 0.60 * sigmoid(norm * 8.0 - 1.0);

        // توکن‌های نشانه‌گذاری کوتاه‌تر و کم‌شدت‌تر
        boolean punct = (tokenId == 2);
        double dur = punct ? 0.05 : TOKEN_SECONDS * (0.85 + 0.3 * sigmoid(norm * 4.0));

        int n = Math.max(1, (int) (dur * sampleRate));
        float[] out = new float[n];
        double phase = 0.0;
        for (int i = 0; i < n; i++) {
            double t = i / (double) sampleRate;
            // پوش نرم (fade in/out) برای جلوگیری از کلیک
            double env = Math.min(1.0, Math.min(t * 40.0, (dur - t) * 40.0));
            env = Math.max(0.0, env);
            // هارمونیک‌ها: پایه + دو هارمونیک (شبیه واکه)
            double s = Math.sin(phase)
                    + 0.45 * Math.sin(2 * phase)
                    + 0.20 * Math.sin(3 * phase);
            out[i] = (float) (amp * env * s / 1.65);
            phase += 2 * Math.PI * pitch / sampleRate;
        }
        return out;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private float[] stitch(List<float[]> parts, double gapSeconds) {
        int gap = (int) (gapSeconds * sampleRate);
        int total = 0;
        for (float[] p : parts) {
            total += p.length;
        }
        total += gap * Math.max(0, parts.size() - 1);
        float[] out = new float[total];
        int cursor = 0;
        for (int i = 0; i < parts.size(); i++) {
            float[] p = parts.get(i);
            System.arraycopy(p, 0, out, cursor, p.length);
            cursor += p.length;
            if (i < parts.size() - 1) {
                cursor += gap;
            }
        }
        return out;
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

    @Override
    public void close() {
        // منبع خارجی برای آزادسازی وجود ندارد
    }
}
