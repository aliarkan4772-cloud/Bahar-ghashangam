package com.pockettts.persian;

/**
 * SynthesisResult — نتیجهٔ یک اجرای سنتز همراه با آمار زمان‌بندی.
 * ============================================================
 * برای نمایش «سری در برابر موازی» در UI و برای تست‌های خودکار.
 * ============================================================
 */
public final class SynthesisResult {

    /** نمونه‌های صوتی نهایی (بعد از gain سراسری و چسباندن). */
    public final float[] audio;

    /** زمان سپری‌شده بر حسب میلی‌ثانیه. */
    public final long millis;

    /** تعداد workerهای به‌کاررفته (۱ = سری). */
    public final int workers;

    /** آیا واقعاً موازی اجرا شد؟ */
    public final boolean parallel;

    /** تعداد chunkهای شکست‌خورده (خطای جزئی، بدون توقف کل کار). */
    public final int failed;

    /** تعداد chunkها در این اجرا. */
    public final int chunks;

    public SynthesisResult(float[] audio, long millis, int workers,
                           boolean parallel, int failed, int chunks) {
        this.audio = (audio == null) ? new float[0] : audio;
        this.millis = millis;
        this.workers = workers;
        this.parallel = parallel;
        this.failed = failed;
        this.chunks = chunks;
    }

    /** توصیف کوتاه فارسی برای نمایش در UI. */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        sb.append(parallel ? "موازی" : "سری");
        sb.append(" · ").append(workers).append(" thread");
        sb.append(" · ").append(chunks).append(" chunk");
        sb.append(" · ").append(millis).append(" ms");
        if (failed > 0) {
            sb.append(" · ").append(failed).append(" خطا");
        }
        return sb.toString();
    }
}
