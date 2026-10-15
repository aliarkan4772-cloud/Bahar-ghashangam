package com.pockettts.persian;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * LutEmbedding — جدول جست‌وجوی embedding توکن‌های متنی (LUT).
 * ============================================================
 * <p>این کلاس فایل {@code models/lut_weight.npy} را می‌خواند و هر شناسهٔ توکن
 * (خروجی توکنایزر) را به یک بردار embedding با بُعد {@code dim} تبدیل می‌کند.
 * این بردارها ورودی مرحلهٔ FlowLM هستند (پیش از تولید latent).</p>
 *
 * <p><b>فرمت فایل .npy</b> (فرمت استاندارد NumPy):</p>
 * <pre>
 *   ┌───────────────────────────────┬──────────────────────────────────────┐
 *   │ magic  : 6 بایت "\x93NUMPY"   │ امضای فایل                           │
 *   │ version: 2 بایت (major,minor) │ نسخه (۱ یا ۲/۳)                      │
 *   │ hlen   : 2 یا 4 بایت LE       │ طول هدر (نسخهٔ ۱: uint16، ۲+: uint32)│
 *   │ header : hlen بایت ASCII      │ دیکشنری پایتون: descr/fortran_order/ │
 *   │        :                      │ shape — با فاصله پر شده و \n پایانی  │
 *   │ data   : بقیهٔ فایل           │ دادهٔ خام با ترتیب row-major         │
 *   └───────────────────────────────┴──────────────────────────────────────┘
 * </pre>
 *
 * <p><b>پشتیبانی از dtype:</b> {@code <f4} (float32، رایج)، {@code >f4}،
 * {@code <f8}/{@code >f8} (float64)، و انواع صحیح ({@code i1,i2,i4,i8,u1,u2,u4,u8})
 * که به float تبدیل می‌شوند. ترتیب ستون‌محور (fortran_order=True) نیز پشتیبانی می‌شود.</p>
 *
 * <p><b>دو مسیر خواندن:</b></p>
 * <ul>
 *   <li>اندروید: {@link #fromAssets(android.content.res.AssetManager, String)} —
 *       از {@code AssetManager} می‌خواند (نه File مستقیم؛ چون فایل‌های assets
 *       داخل APK فشرده‌اند و مسیر فایل واقعی ندارند).</li>
 *   <li>دسکتاپ/تست: {@link #fromFile(Path)} — از مسیر فایل روی دیسک.</li>
 * </ul>
 *
 * <p>در صورت نبودن فایل یا نامعتبر بودن هدر، یک {@link IOException} با پیام
 * <b>واضح و فارسی</b> پرتاب می‌شود (نه NullPointerException یا خطای مبهم).</p>
 * ============================================================
 */
public final class LutEmbedding {

    /** امضای فایل npy. */
    private static final byte[] MAGIC = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};

    private final float[][] table;   // [vocabSize][dim]
    private final int vocabSize;
    private final int dim;
    private final String source;     // برای پیام‌های خطا/لاگ
    private final int dataOffset;    // آفست واقعی شروع داده در فایل npy

    private LutEmbedding(float[][] table, String source, int dataOffset) {
        this.table = table;
        this.vocabSize = table.length;
        this.dim = (table.length == 0) ? 0 : table[0].length;
        this.source = source;
        this.dataOffset = dataOffset;
    }

    /**
     * آفست واقعی شروع داده در فایل npy (بر حسب بایت).
     *
     * <p>⚠️ نکتهٔ مهم: این مقدار <b>ثابت نیست</b>. برای هدرهای کوتاه معمولاً ۱۲۸ است
     * (۱۰ بایت magic+version+hlen و ۱۱۸ بایت هدر)، اما اگر دیکشنری هدر طولانی‌تر شود
     * (مثلاً ابعاد چندرقمی یا آرایهٔ چندبُعدی) به ۱۹۲ یا بیشتر می‌رسد. هرجا آفست
     * شروع داده لازم بود، <b>از همین مقدار</b> استفاده کنید، نه از عدد ثابت ۱۲۸.</p>
     */
    public int dataOffset() {
        return dataOffset;
    }

    // ---------------------------------------------------------
    // سازنده‌های عمومی (مسیرهای مختلف خواندن)
    // ---------------------------------------------------------

    /** خواندن از مسیر فایل روی دیسک (دسکتاپ/تست). */
    public static LutEmbedding fromFile(Path path) throws IOException {
        // پیشفرض: اعتبارسنجی با Dims.TEXT_DIM (مسیر دسکتاپ/تست)
        return fromFile(path, Dims.TEXT_DIM);
    }

    /**
     * خواندن از مسیر فایل + اعتبارسنجی با بُعد مورد انتظار (مثلاً از model_contract.json).
     *
     * @param expectedDim بُعد مورد انتظار؛ اگر ≤ ۰ باشد، اعتبارسنجی انجام نمی‌شود
     */
    public static LutEmbedding fromFile(Path path, int expectedDim) throws IOException {
        if (path == null) {
            throw new IOException("مسیر فایل lut_weight.npy داده نشده است.");
        }
        if (!Files.exists(path)) {
            throw new IOException("فایل lut_weight.npy یافت نشد: " + path
                    + "\nراه‌حل: آن را با tools/make_fixtures.py بسازید و در "
                    + "app/src/main/assets/models/ قرار دهید.");
        }
        byte[] bytes = Files.readAllBytes(path);
        LutEmbedding lut = fromBytes(bytes, path.toString());
        if (expectedDim > 0) {
            lut.verifyDim(expectedDim);
        }
        logDims(lut);
        return lut;
    }

    /** چاپ ابعاد و آفست داده (برای تشخیص سریع مشکل در logcat/ترمینال). */
    private static void logDims(LutEmbedding lut) {
        System.out.println("[LutEmbedding] " + lut.source()
                + "  vocab=" + lut.vocabSize()
                + "  dim=" + lut.dim()
                + "  dataOffset=" + lut.dataOffset()
                + "  (" + Dims.summary() + ")");
    }

    /**
     * بررسی همخوانی بُعد این جدول با بُعد مورد انتظار.
     *
     * @throws IOException اگر ابعاد ناهمخوان باشند (با پیام فارسی و راه‌حل)
     */
    public void verifyDim(int expectedDim) throws IOException {
        if (dim != expectedDim) {
            throw new IOException("ابعاد LUT ناهمخوان است:" + "\n"
                    + "  فایل " + source + " بُعد " + dim + " دارد، اما بُعد مورد انتظار " + expectedDim + " است.\n"
                    + "  (vocab=" + vocabSize + "، ابعاد خوانده‌شده=" + vocabSize + "×" + dim + ")\n"
                    + "راه‌حل: فایل را با ابعاد درست بازتولید کنید:\n"
                    + "   python3 tools/make_fixtures.py --vocab " + vocabSize + " --dim " + expectedDim + "\n"
                    + "و سپس آن را در app/src/main/assets/models/lut_weight.npy قرار دهید.\n"
                    + "نکته: مقدار مورد انتظار از Dims.TEXT_DIM می‌آید و باید با text_dim در "
                    + "model_contract.json یکی باشد.");
        }
    }

    /**
     * خواندن از {@code AssetManager} اندروید.
     *
     * <p>نکتهٔ مهم: فایل‌های داخل {@code assets} در APK فشرده می‌شوند و
     * مسیر فایل واقعی ندارند؛ بنابراین باید از {@code AssetManager.open()}
     * استفاده کرد، نه {@code new File(...)}.</p>
     *
     * @param assets    نمونهٔ AssetManager (مثلاً {@code getAssets()})
     * @param assetPath مسیر نسبی داخل assets، مثل {@code "models/lut_weight.npy"}
     */
    public static LutEmbedding fromAssets(android.content.res.AssetManager assets,
                                          String assetPath) throws IOException {
        // پیشفرض: اعتبارسنجی با Dims.TEXT_DIM
        return fromAssets(assets, assetPath, Dims.TEXT_DIM);
    }

    /**
     * خواندن از assets + اعتبارسنجی با بُعد مورد انتظار (مثلاً از model_contract.json).
     *
     * <p>نکته: خطای خواندن فایل و خطای ناهمخوانی ابعاد جدا گزارش می‌شوند تا
     * پیام «فایل پیدا نشد» با پیام «ابعاد ناهمخوان است» قاطی نشوند.</p>
     */
    public static LutEmbedding fromAssets(android.content.res.AssetManager assets,
                                          String assetPath, int expectedDim) throws IOException {
        if (assets == null) {
            throw new IOException("AssetManager داده نشده است.");
        }
        byte[] bytes;
        try (InputStream in = assets.open(assetPath)) {
            bytes = readAll(in);
        } catch (IOException e) {
            throw new IOException("خواندن " + assetPath + " از assets ناموفق بود: "
                    + e.getMessage()
                    + "\nراه‌حل: مطمئن شوید فایل در app/src/main/assets/models/ قرار دارد "
                    + "و با tools/make_fixtures.py ساخته شده است.", e);
        }
        LutEmbedding lut = fromBytes(bytes, "assets:" + assetPath);
        if (expectedDim > 0) {
            lut.verifyDim(expectedDim);
        }
        logDims(lut);
        return lut;
    }

    /** خواندن از یک InputStream دلخواه. */
    public static LutEmbedding fromStream(InputStream in, String source) throws IOException {
        if (in == null) {
            throw new IOException("InputStream داده نشده است.");
        }
        return fromBytes(readAll(in), source);
    }

    // ---------------------------------------------------------
    // پارس واقعی فرمت .npy
    // ---------------------------------------------------------

    /** پارس بایت‌های یک فایل npy و ساخت جدول. */
    public static LutEmbedding fromBytes(byte[] bytes, String source) throws IOException {
        if (bytes == null || bytes.length < 10) {
            throw new IOException("فایل npy خیلی کوچک است (" + source + ").");
        }
        // ۱) بررسی امضای magic
        for (int i = 0; i < MAGIC.length; i++) {
            if (bytes[i] != MAGIC[i]) {
                throw new IOException("امضای فایل npy نامعتبر است (" + source
                        + ")؛ انتظار «\\x93NUMPY» بود.");
            }
        }
        int major = bytes[6] & 0xFF;
        int minor = bytes[7] & 0xFF;

        // ۲) طول هدر: نسخهٔ ۱ → uint16، نسخهٔ ۲/۳ → uint32
        int headerLen;
        int headerStart;
        if (major == 1) {
            headerLen = (bytes[8] & 0xFF) | ((bytes[9] & 0xFF) << 8);
            headerStart = 10;
        } else if (major == 2 || major == 3) {
            if (bytes.length < 12) {
                throw new IOException("هدر npy ناقص است (" + source + ").");
            }
            headerLen = (bytes[8] & 0xFF)
                    | ((bytes[9] & 0xFF) << 8)
                    | ((bytes[10] & 0xFF) << 16)
                    | ((bytes[11] & 0xFF) << 24);
            headerStart = 12;
        } else {
            throw new IOException("نسخهٔ npy پشتیبانی نمی‌شود: " + major + "." + minor
                    + " (" + source + ").");
        }
        if (headerStart + headerLen > bytes.length) {
            throw new IOException("طول هدر npy از اندازهٔ فایل بیشتر است (" + source + ").");
        }

        // ۳) پارس دیکشنری هدر (ASCII)
        String header = new String(bytes, headerStart, headerLen, StandardCharsets.US_ASCII);
        String descr = extractString(header, "descr");
        boolean fortran = extractBool(header, "fortran_order");
        long[] shape = extractShape(header);
        if (descr == null || shape == null) {
            throw new IOException("هدر npy ناقص است (descr یا shape پیدا نشد) — " + source);
        }
        if (shape.length != 2) {
            throw new IOException("انتظار آرایهٔ دوبُعدی [vocab, dim] بود، اما shape="
                    + java.util.Arrays.toString(shape) + " (" + source + ").");
        }
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        if (rows <= 0 || cols <= 0) {
            throw new IOException("ابعاد npy نامعتبر است: " + rows + "×" + cols + " (" + source + ").");
        }

        // ۴) تعیین نوع داده و اندازهٔ هر عنصر
        DType dt = DType.parse(descr);
        if (dt == null) {
            throw new IOException("نوع دادهٔ npy پشتیبانی نمی‌شود: «" + descr + "» (" + source + ").");
        }
        long need = (long) rows * cols * dt.size;
        int dataStart = headerStart + headerLen;
        if (dataStart + need > bytes.length) {
            throw new IOException("دادهٔ npy ناقص است: انتظار " + need + " بایت، اما "
                    + (bytes.length - dataStart) + " بایت موجود است (" + source + ").");
        }

        // ۵) خواندن داده و تبدیل به float
        // نکته: آفست شروع داده (dataStart) از خود هدر محاسبه شد، نه از عدد ثابت ۱۲۸.
        ByteBuffer bb = ByteBuffer.wrap(bytes, dataStart, (int) need).order(dt.order);
        float[][] table = new float[rows][cols];
        if (!fortran) {
            // row-major: هر ردیف پشت سر هم
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    table[r][c] = dt.read(bb);
                }
            }
        } else {
            // column-major: هر ستون پشت سر هم
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    table[r][c] = dt.read(bb);
                }
            }
        }
        return new LutEmbedding(table, source, dataStart);
    }

    // ---------------------------------------------------------
    // API عمومی
    // ---------------------------------------------------------

    /** تعداد توکن‌های جدول (vocab size). */
    public int vocabSize() {
        return vocabSize;
    }

    /** بُعد هر بردار embedding. */
    public int dim() {
        return dim;
    }

    public String source() {
        return source;
    }

    /**
     * تبدیل یک شناسهٔ توکن به بردار embedding.
     *
     * @throws IllegalArgumentException اگر شناسه خارج از بازهٔ جدول باشد.
     */
    public float[] embed(int tokenId) {
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw new IllegalArgumentException("شناسهٔ توکن خارج از بازه است: " + tokenId
                    + " (بازهٔ مجاز: 0.." + (vocabSize - 1) + ")");
        }
        return table[tokenId].clone();
    }

    /**
     * نگاشت ایمن یک شناسهٔ توکن به بازهٔ جدول و بازگرداندن embedding آن.
     *
     * <p>اگر توکنایزر شناسه‌ای بزرگ‌تر از vocabSize تولید کند (مثلاً وقتی اندازهٔ vocab
     * در ابزار ساخت fixture کوچک انتخاب شده باشد)، این متد به‌جای پرتاب استثنا آن را
     * با {@code floorMod} به بازه می‌آورد تا سنتز متوقف نشود.</p>
     */
    public float[] embedToken(int tokenId) {
        int id = Math.floorMod(tokenId, vocabSize);
        return table[id].clone();
    }

    /** میانگین embedding یک دنبالهٔ توکن با نگاشت ایمن ({@link #embedToken}). */
    public float[] meanEmbeddingSafe(int[] tokens) {
        float[] out = new float[dim];
        if (tokens == null || tokens.length == 0) {
            return out;
        }
        for (int t : tokens) {
            float[] v = embedToken(t);
            for (int i = 0; i < dim; i++) {
                out[i] += v[i];
            }
        }
        for (int i = 0; i < dim; i++) {
            out[i] /= tokens.length;
        }
        return out;
    }

    /** تبدیل یک دنبالهٔ توکن به ماتریس embedding [n][dim]. */
    public float[][] embed(int[] tokens) {
        if (tokens == null) {
            return new float[0][];
        }
        float[][] out = new float[tokens.length][];
        for (int i = 0; i < tokens.length; i++) {
            out[i] = embed(tokens[i]);
        }
        return out;
    }

    /** میانگین بردارهای embedding یک دنباله (برای خلاصه‌سازی/دیباگ). */
    public float[] meanEmbedding(int[] tokens) {
        float[] out = new float[dim];
        if (tokens == null || tokens.length == 0) {
            return out;
        }
        for (int t : tokens) {
            float[] v = embed(t);
            for (int i = 0; i < dim; i++) {
                out[i] += v[i];
            }
        }
        for (int i = 0; i < dim; i++) {
            out[i] /= tokens.length;
        }
        return out;
    }

    // ---------------------------------------------------------
    // کمکی‌های پارس هدر
    // ---------------------------------------------------------

    /** استخراج مقدار رشته‌ای مثل {@code 'descr': '<f4'} از هدر. */
    private static String extractString(String header, String key) {
        int k = header.indexOf("'" + key + "'");
        if (k < 0) {
            return null;
        }
        int colon = header.indexOf(':', k);
        if (colon < 0) {
            return null;
        }
        int q1 = header.indexOf('\'', colon);
        if (q1 < 0) {
            return null;
        }
        int q2 = header.indexOf('\'', q1 + 1);
        if (q2 < 0) {
            return null;
        }
        return header.substring(q1 + 1, q2);
    }

    /** استخراج مقدار بولی مثل {@code 'fortran_order': False}. */
    private static boolean extractBool(String header, String key) {
        int k = header.indexOf("'" + key + "'");
        if (k < 0) {
            return false;
        }
        int colon = header.indexOf(':', k);
        if (colon < 0) {
            return false;
        }
        String rest = header.substring(colon + 1).trim();
        return rest.startsWith("True");
    }

    /** استخراج shape مثل {@code 'shape': (1000, 512)}. */
    private static long[] extractShape(String header) {
        int k = header.indexOf("'shape'");
        if (k < 0) {
            return null;
        }
        int open = header.indexOf('(', k);
        int close = header.indexOf(')', open);
        if (open < 0 || close < 0) {
            return null;
        }
        String inner = header.substring(open + 1, close).trim();
        if (inner.isEmpty()) {
            return new long[0];
        }
        String[] parts = inner.split(",");
        java.util.List<Long> vals = new java.util.ArrayList<>();
        for (String p : parts) {
            String s = p.trim();
            if (s.isEmpty()) {
                continue;
            }
            try {
                vals.add(Long.parseLong(s));
            } catch (NumberFormatException e) {
                return null;
            }
        }
        long[] out = new long[vals.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = vals.get(i);
        }
        return out;
    }

    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) {
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }

    // ---------------------------------------------------------
    // انواع دادهٔ npy
    // ---------------------------------------------------------

    /** توصیف‌گر نوع دادهٔ npy (مثل {@code <f4}). */
    private static final class DType {
        final ByteOrder order;
        final char kind;   // f=float, i=int, u=uint
        final int size;    // بایت

        DType(ByteOrder order, char kind, int size) {
            this.order = order;
            this.kind = kind;
            this.size = size;
        }

        static DType parse(String descr) {
            if (descr == null || descr.length() < 3) {
                return null;
            }
            ByteOrder order;
            char c0 = descr.charAt(0);
            if (c0 == '<') {
                order = ByteOrder.LITTLE_ENDIAN;
            } else if (c0 == '>') {
                order = ByteOrder.BIG_ENDIAN;
            } else if (c0 == '|') {
                order = ByteOrder.LITTLE_ENDIAN;   // تک‌بایتی؛ ترتیب مهم نیست
            } else {
                return null;
            }
            char kind = descr.charAt(1);
            int size;
            try {
                size = Integer.parseInt(descr.substring(2));
            } catch (NumberFormatException e) {
                return null;
            }
            if (kind != 'f' && kind != 'i' && kind != 'u') {
                return null;
            }
            return new DType(order, kind, size);
        }

        /** خواندن یک عنصر از بافر و تبدیل به float. */
        float read(ByteBuffer bb) {
            switch (kind) {
                case 'f':
                    if (size == 4) {
                        return bb.getFloat();
                    }
                    if (size == 8) {
                        return (float) bb.getDouble();
                    }
                    break;
                case 'i':
                    if (size == 1) {
                        return bb.get();
                    }
                    if (size == 2) {
                        return bb.getShort();
                    }
                    if (size == 4) {
                        return bb.getInt();
                    }
                    if (size == 8) {
                        return bb.getLong();
                    }
                    break;
                case 'u':
                    if (size == 1) {
                        return bb.get() & 0xFF;
                    }
                    if (size == 2) {
                        return bb.getShort() & 0xFFFF;
                    }
                    if (size == 4) {
                        return bb.getInt() & 0xFFFFFFFFL;
                    }
                    break;
                default:
                    break;
            }
            throw new IllegalStateException("نوع دادهٔ npy پشتیبانی نمی‌شود: " + kind + size);
        }
    }
}
