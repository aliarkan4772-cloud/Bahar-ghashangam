package com.pockettts.persian;

/**
 * ChunkSynthesizer — قرارداد سنتز یک chunk.
 * ============================================================
 * این واسط، {@link ParallelSynthesizer} را از پیاده‌سازی مشخص
 * ({@link TtsEngine}) جدا می‌کند؛ بنابراین:
 *   - هر worker می‌تواند نمونهٔ مستقل خودش را داشته باشد؛
 *   - تست واحد می‌تواند پیاده‌سازی جعلی (fake) تزریق کند
 *     بدون نیاز به مدل ONNX واقعی.
 *
 * نکتهٔ کلیدی: پیاده‌سازی باید <b>حالت‌دار (stateful)</b> باشد و
 * هیچ دو thread نباید هم‌زمان از یک نمونه استفاده کنند؛ به همین
 * دلیل {@link ParallelSynthesizer} یک نمونهٔ اختصاصی به هر thread
 * می‌دهد (نه یک نمونهٔ مشترک با قفل).
 * ============================================================
 */
public interface ChunkSynthesizer extends AutoCloseable {

    /**
     * سنتز یک chunk.
     *
     * @param chunk              متن chunk.
     * @param pristineVoiceCache voice cache دست‌نخورده؛ پیاده‌سازی باید
     *                           از آن <b>کپی</b> بگیرد و اصل را تغییر ندهد.
     * @param seed               دانهٔ نویز این chunk (تکرارپذیری).
     * @return نمونه‌های صوتی chunk (ممکن است خالی باشد).
     */
    float[] synthesizeChunk(String chunk, float[][] pristineVoiceCache, long seed) throws Exception;

    /** آزادسازی منابع ONNX. */
    @Override
    void close();
}
