package com.pockettts.persian;

import java.util.ArrayList;
import java.util.List;

/**
 * TokTest — پوشش نازک روی توکنایزر واقعی مدل.
 *
 * <p>در پروژهٔ اصلی، توکنایزر (SentencePiece/BPE) از فایل مدل بارگذاری می‌شود.
 * اینجا یک پیاده‌سازی سبک ارائه شده که قرارداد {@code load()} و {@code encode()}
 * را رعایت می‌کند تا {@link Chunker} بدون وابستگی به کتابخانهٔ خارجی کامپایل شود.
 * برای استفادهٔ واقعی، این کلاس را با توکنایزر ONNX/Java خودتان جایگزین کنید.</p>
 *
 * <p>نکتهٔ مهم: در نسخهٔ قبلی، خطای بارگذاری بی‌صدا بلعیده می‌شد و همهٔ شمارش‌ها
 * صفر می‌شد. اینجا خطا پرتاب می‌شود تا {@link Chunker} بتواند به حالت تخمینی
 * عقب‌نشینی کند.</p>
 */
public final class TokTest {

    private static boolean loaded = false;

    private TokTest() { }

    /** بارگذاری توکنایزر. در صورت شکست، استثنا پرتاب می‌شود (نه بلعیدن بی‌صدا). */
    public static synchronized void load() throws Exception {
        // در پیاده‌سازی واقعی: بارگذاری vocab از assets/models/tokenizer.model
        // اینجا فقط وضعیت را علامت می‌زنیم.
        loaded = true;
    }

    public static boolean isLoaded() {
        return loaded;
    }

    /**
     * تبدیل متن به شناسهٔ توکن‌ها.
     * پیاده‌سازی نمونه: هر «واژه» یک توکن، به‌علاوهٔ یک توکن برای هر نویسهٔ
     * نشانه‌گذاری. این تقریب برای chunking کافی است و با توکنایزر واقعی
     * جایگزین می‌شود.
     */
    public static int[] encode(String text) {
        if (text == null || text.trim().isEmpty()) {
            return new int[0];
        }
        List<Integer> ids = new ArrayList<>();
        for (String w : text.trim().split("\\s+")) {
            if (w.isEmpty()) {
                continue;
            }
            ids.add(1); // توکن واژه
            // نویسه‌های نشانه‌گذاری جداگانه توکن می‌شوند
            for (int i = 0; i < w.length(); i++) {
                char c = w.charAt(i);
                if (c == '.' || c == ',' || c == '!' || c == '?' || c == '؟' || c == '…') {
                    ids.add(2);
                }
            }
        }
        int[] out = new int[ids.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ids.get(i);
        }
        return out;
    }
}
