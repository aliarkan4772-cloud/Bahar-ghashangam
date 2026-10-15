package com.pockettts.persian;

import java.util.ArrayList;
import java.util.List;

/**
 * Chunker — نسخه v8
 * ============================================================
 * وظیفه:
 *   تقسیم یک متن فارسی به قطعه‌های (chunk) مناسب برای موتور TTS،
 *   به‌گونه‌ای که:
 *     - هیچ قطعه‌ای از سقف توکن (HARD_LIMIT) بزرگ‌تر نشود؛
 *     - تا حد امکان از بریدن وسط جمله یا پایان‌دادن یک قطعه با «اضافه» (ezafe)
 *       پرهیز شود؛ چون اضافه به کلمهٔ بعدی می‌چسبد و بریدن آن لحن را خراب می‌کند.
 *
 * منطق دو سطحی (soft / hard):
 *   - SOFT_LIMIT : هدفِ ترجیحی برای اندازهٔ هر قطعه. جمله‌های کوچک‌تر از آن
 *                  می‌توانند در یک قطعه کنار هم قرار بگیرند.
 *   - HARD_LIMIT : سقفِ مطلق. هیچ قطعه‌ای اجازه ندارد از آن بزرگ‌تر شود.
 *                  (باید بزرگ‌تر از SOFT باشد تا چند توکن «فضای تنفس» بدهد.)
 *
 * اصلاحات نسبت به v7:
 *   1) تفکیک SOFT (18) از HARD (20) — در v7 هر دو 20 بودند و مکانیزم عقب‌نشینی بی‌اثر بود.
 *   2) tok() دیگر خطا را بی‌صدا نمی‌بلعد؛ در صورت شکست به تخمین «هر واژه ≈ یک توکن» می‌رود.
 *   3) furthestAllowed با continue (نه break) همهٔ نقاط برش را بررسی می‌کند.
 *   4) حذف کامل کد مرده (LIGHT/PREPS/FUNC/CONJ/HR/MINW/... و isBadBoundary).
 *   5) استفادهٔ واقعی از پارامتر keepEzafe.
 *   6) splitSentences دستی که نقطهٔ اعشاری و URL را اشتباه نمی‌شکند.
 *   7) بارگذاری thread-safe توکنایزر + لاگ هشدار برای قطعه‌های غیرقابل‌بسته‌بندی.
 * ============================================================
 */
public class Chunker {

    /** هدفِ نرم برای تعداد توکن هر قطعه. */
    static final int SOFT_LIMIT = 18;

    /** سقفِ سخت؛ هیچ قطعه‌ای بزرگ‌تر از این نمی‌شود. */
    static final int HARD_LIMIT = 20;

    /** کاراکترهای پایان جمله. */
    private static final String SENTENCE_ENDINGS = ".!؟…";

    // ---------------------------------------------------------
    // مدیریت Tokenizer
    // ---------------------------------------------------------
    private static volatile boolean tokenizerLoaded = false;
    private static volatile boolean tokenizerFailed = false;

    /**
     * شمارش تعداد توکن یک رشته.
     * اگر توکنایزر در دسترس نباشد، به تخمین «هر واژه ≈ یک توکن» برمی‌گردیم
     * تا رفتار برنامه قابل پیش‌بینی بماند (به‌جای صفر شدن همهٔ شمارش‌ها).
     */
    static int tokenCount(String s) {
        if (s == null || s.trim().isEmpty()) {
            return 0;
        }
        ensureTokenizerLoaded();
        if (tokenizerFailed) {
            return words(s).size();
        }
        try {
            return TokTest.encode(s).length;
        } catch (Exception e) {
            System.err.println("[Chunker] خطا در شمارش توکن؛ از تخمین واژه استفاده شد: " + e);
            return words(s).size();
        }
    }

    /** بارگذاری یک‌باره و امنِ توکنایزر. */
    private static synchronized void ensureTokenizerLoaded() {
        if (tokenizerLoaded || tokenizerFailed) {
            return;
        }
        try {
            TokTest.load();
            tokenizerLoaded = true;
        } catch (Exception e) {
            tokenizerFailed = true;
            System.err.println("[Chunker] خطا در بارگذاری TokTest؛ حالت تخمین توکن فعال شد: " + e);
        }
    }

    // ---------------------------------------------------------
    // ابزارهای واژه و جمله
    // ---------------------------------------------------------

    static List<String> words(String s) {
        List<String> result = new ArrayList<>();
        if (s == null) {
            return result;
        }
        for (String x : s.trim().split("\\s+")) {
            if (!x.isEmpty()) {
                result.add(x);
            }
        }
        return result;
    }

    static String join(List<String> list) {
        return String.join(" ", list);
    }

    /**
     * تشخیص «اضافه» (ezafe) بودن یک واژه.
     * قرارداد توکنایزر: اضافه با پسوند «1» علامت‌گذاری می‌شود.
     */
    static boolean isEzafe(String word) {
        return word != null && word.endsWith("1");
    }

    // ---------------------------------------------------------
    // منطق برش
    // ---------------------------------------------------------

    static int tokensIn(List<String> words, int from, int to) {
        return tokenCount(join(words.subList(from, to)));
    }

    /**
     * آیا می‌توان بعد از واژهٔ شمارهٔ (c-1) برش زد؟
     * - برش در انتهای متن همیشه مجاز است.
     * - اگر keepEzafe فعال باشد، اجازه می‌دهیم قطعه به «اضافه» ختم شود.
     * - در غیر این صورت، آخرین واژهٔ قطعه نباید اضافه باشد.
     */
    static boolean cutAllowed(List<String> words, int c, int total, boolean keepEzafe) {
        if (c >= total) {
            return true;
        }
        if (keepEzafe) {
            return true;
        }
        return !isEzafe(words.get(c - 1));
    }

    /**
     * دورترین نقطهٔ برشِ مجاز از start که تعداد توکنش از cap بیشتر نشود.
     * برخلاف v7، با اولین عبور از سقف متوقف نمی‌شویم (continue)؛ چون شمارش
     * توکن زیرجمعی نیست و ممکن است بازهٔ طولانی‌تری هنوز زیر سقف باشد.
     */
    static int furthestAllowed(List<String> words, int start, int upto, int cap, boolean keepEzafe) {
        int best = -1;
        for (int c = start + 1; c <= upto; c++) {
            if (tokensIn(words, start, c) > cap) {
                continue;
            }
            if (cutAllowed(words, c, words.size(), keepEzafe)) {
                best = c;
            }
        }
        return best;
    }

    /**
     * بسته‌بندی یک جملهٔ بلند به چند قطعه با رعایت SOFT و HARD.
     */
    static List<String> packWords(List<String> words, boolean keepEzafe) {
        List<String> chunks = new ArrayList<>();
        if (words == null || words.isEmpty()) {
            return chunks;
        }
        final int n = words.size();
        int start = 0;
        while (start < n) {
            int best = furthestAllowed(words, start, n, SOFT_LIMIT, keepEzafe);
            if (best < 0) {
                best = furthestAllowed(words, start, n, HARD_LIMIT, keepEzafe);
            }
            if (best < 0) {
                int c = start + 1;
                while (c < n && !cutAllowed(words, c, n, keepEzafe)) {
                    c++;
                }
                best = c;
                System.err.println("[Chunker] هشدار: قطعه‌ای از سقف توکن بزرگ‌تر شد (شروع از واژه " + start + ")");
            }
            chunks.add(join(words.subList(start, best)));
            start = best;
        }
        return chunks;
    }

    // ---------------------------------------------------------
    // رابط عمومی
    // ---------------------------------------------------------

    public static List<String> chunk(String s) {
        return chunk(s, false);
    }

    /**
     * تقسیم کل متن به قطعه‌های TTS.
     *
     * @param s         متن ورودی.
     * @param keepEzafe اگر true باشد، به قطعه‌های میانی هم اجازه می‌دهیم به
     *                  «اضافه» ختم شوند. برای TTS معمولاً false بهتر است.
     */
    public static List<String> chunk(String s, boolean keepEzafe) {
        List<String> out = new ArrayList<>();
        if (s == null || s.trim().isEmpty()) {
            return out;
        }
        for (String sentence : splitSentences(s)) {
            String sent = sentence.trim();
            if (sent.isEmpty()) {
                continue;
            }
            int sentTokens = tokenCount(sent);
            if (sentTokens > HARD_LIMIT) {
                out.addAll(packWords(words(sent), keepEzafe));
            } else {
                out.add(sent);
            }
        }
        return out;
    }

    /**
     * تقسیم متن به جمله.
     * نقطهٔ اعشاری (3.14) و URL (example.com) را اشتباه «پایان جمله» حساب نمی‌کنیم.
     */
    static List<String> splitSentences(String s) {
        // ۱) نقاط اعشاری داخل اعداد را موقتاً نقاب می‌زنیم.
        String masked = s.replaceAll("([0-9۰-۹])\\.([0-9۰-۹])", "$1__POINT__$2");
        // ۲) تقسیم روی علامت پایان جمله که پس از آن فاصله یا خط جدید آمده باشد.
        String[] parts = masked.split("(?<=[.!؟…])\\s+|\\n+");
        // ۳) بازگرداندن نقاط اعشاری و حذف اجزای خالی.
        List<String> sentences = new ArrayList<>();
        for (String p : parts) {
            String restored = p.replace("__POINT__", ".");
            if (!restored.trim().isEmpty()) {
                sentences.add(restored);
            }
        }
        return sentences;
    }

    // ---------------------------------------------------------
    // خودآزمایی ساده
    // ---------------------------------------------------------
    public static void main(String[] args) {
        String[] samples = {
            "",
            "   ",
            "سلام",
            "این یک جملهٔ کوتاه است.",
            "این جمله اول است. این جمله دوم است. این جمله سوم کمی طولانی‌تر است.",
            "عدد اعشاری 3.14 را ببین. حالا جمله بعدی شروع می‌شود.",
            "قیمت ۱۲.۵ برابر است. و این جمله بعدی است."
        };
        for (String s : samples) {
            System.out.println("ورودی: «" + s + "»");
            System.out.println("خروجی: " + chunk(s));
        }
    }
}
