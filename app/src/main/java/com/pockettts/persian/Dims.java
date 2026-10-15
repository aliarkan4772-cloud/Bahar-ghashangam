package com.pockettts.persian;

/**
 * Dims — منبع یگانهٔ حقیقت برای همهٔ ابعاد مدل.
 * ============================================================
 * <p><b>چرا این کلاس لازم است؟</b> پیش‌تر عدد ابعاد در چند جای مختلف تکرار شده بود
 * (مقدار latent در {@link TtsEngine#LDIM}، مقدار پیش‌فرض در {@code tools/make_fixtures.py}،
 * و مقدار {@code text_dim} در {@code model_contract.json}). اگر این‌ها با هم همخوان
 * نباشند، برنامه یا با خطای «ابعاد ناهمخوان» متوقف می‌شود یا — بدتر — با داده‌های غلط
 * ادامه می‌دهد و صدای بی‌معنا تولید می‌کند.</p>
 *
 * <p><b>قاعدهٔ طلایی:</b> مقدار {@link #TEXT_DIM} در این کلاس <b>باید</b> با فیلد
 * {@code text_dim} در فایل {@code app/src/main/assets/models/model_contract.json}
 * و با پارامتر {@code --dim} در هنگام ساخت {@code lut_weight.npy} یکی باشد.</p>
 * ============================================================
 */
public final class Dims {

    private Dims() {
        // کلاس ابزاری — نمونه‌سازی ممنوع
    }

    // ---------------------------------------------------------
    // ابعاد متن / embedding
    // ---------------------------------------------------------

    /**
     * بُعد بردار embedding توکن‌های متنی.
     *
     * <p>این همان مقدار {@code text_dim} در {@code model_contract.json} است و همان
     * بُعدی است که {@code lut_weight.npy} باید داشته باشد: {@code [vocab_size × TEXT_DIM]}.</p>
     *
     * <p>⚠️ مقدار درست برای این مدل <b>۶۴</b> است (طبق {@code model_contract.json}).
     * فایل {@code lut_weight.npy} باید با {@code python3 tools/make_fixtures.py --dim 64}
     * ساخته شود تا ابعادش {@code [50257 × 64]} شود، نه ۱۰۲۴.</p>
     */
    public static final int TEXT_DIM = 64;

    /**
     * بُعد latent مدل FlowLM.
     *
     * <p>این مقدار <b>مستقل</b> از {@link #TEXT_DIM} است: یکی ورودی متن را توصیف می‌کند و
     * دیگری فضای latent صوتی را. به‌همین دلیل دو ثابت جدا تعریف شده‌اند تا تغییر یکی
     * به‌اشتباه دیگری را عوض نکند.</p>
     */
    public static final int LATENT_DIM = TtsEngine.LDIM;

    /** طول voice cache (تعداد توکن‌های صدای مرجع). */
    public static final int VOICE_OFF = TtsEngine.VOICE_OFF;

    /** حداکثر طول context مدل. */
    public static final int CAP = TtsEngine.CAP;

    // ---------------------------------------------------------
    // کمکی
    // ---------------------------------------------------------

    /** نام پیش‌فرض فایل LUT داخل پوشهٔ assets. */
    public static final String LUT_ASSET = "models/lut_weight.npy";

    /** نام فایل قرارداد مدل داخل پوشهٔ assets. */
    public static final String CONTRACT_ASSET = "models/model_contract.json";

    /**
     * بررسی همخوانی بُعد یک LUT بارگذاری‌شده با {@link #TEXT_DIM}.
     *
     * @param lutDim بُعد واقعی خوانده‌شده از فایل npy
     * @throws IllegalStateException اگر ابعاد ناهمخوان باشند
     */
    public static void checkTextDim(int lutDim) {
        if (lutDim != TEXT_DIM) {
            throw new IllegalStateException(
                    "ابعاد LUT ناهمخوان است: فایل lut_weight.npy بُعد " + lutDim
                            + " دارد، اما Dims.TEXT_DIM = " + TEXT_DIM + " است.\n"
                            + "راه‌حل: فایل را با ابعاد درست دوباره بسازید:\n"
                            + "   python3 tools/make_fixtures.py --dim " + TEXT_DIM);
        }
    }

    /** توضیح یک‌خطی ابعاد برای گزارش/لاگ. */
    public static String summary() {
        return "text_dim=" + TEXT_DIM + "  latent_dim=" + LATENT_DIM
                + "  voice_off=" + VOICE_OFF + "  cap=" + CAP;
    }
}
