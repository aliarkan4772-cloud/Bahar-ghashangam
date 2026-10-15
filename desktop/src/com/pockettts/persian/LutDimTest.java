package com.pockettts.persian;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * LutDimTest — تست رفتاری LutEmbedding و Dims (برنامهٔ دسکتاپ، بدون اندروید).
 *
 * <p>سه چیز را بررسی می‌کند:</p>
 * <ol>
 *   <li>فایل با ابعاد درست ({@link Dims#TEXT_DIM}) با موفقیت خوانده شود.</li>
 *   <li>فایل با ابعاد ناهمخوان با پیام فارسی واضح رد شود.</li>
 *   <li>نگاشت ایمن توکن ({@code embedToken}) روی شناسهٔ خارج از بازه کرش نکند.</li>
 * </ol>
 *
 * <p><b>اجرا:</b> {@code java -cp out com.pockettts.persian.LutDimTest <good.npy> <bad.npy>}</p>
 */
public final class LutDimTest {

    private static int pass = 0;
    private static int fail = 0;

    public static void main(String[] args) throws Exception {
        Path good = Paths.get(args.length > 0 ? args[0] : "fixtures/lut_weight.npy");
        Path bad = Paths.get(args.length > 1 ? args[1] : "fixtures/lut_bad_dim.npy");

        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("  LutDimTest — Dims.TEXT_DIM = " + Dims.TEXT_DIM + "  (" + Dims.summary() + ")");
        System.out.println("  فایل درست: " + good);
        System.out.println("  فایل ناهمخوان: " + bad);
        System.out.println("════════════════════════════════════════════════════════════");

        // ── ۱) فایل درست ───────────────────────────────────────
        check("۱) فایل با ابعاد درست (dim=" + Dims.TEXT_DIM + ") خوانده می‌شود", () -> {
            LutEmbedding lut = LutEmbedding.fromFile(good, Dims.TEXT_DIM);
            if (lut.dim() != Dims.TEXT_DIM) {
                throw new AssertionError("بُعد خوانده‌شده غلط است: " + lut.dim());
            }
            System.out.println("      → vocab=" + lut.vocabSize() + "  dim=" + lut.dim()
                    + "  dataOffset=" + lut.dataOffset() + " بایت");
        });

        // ── ۲) فایل ناهمخوان رد می‌شود ─────────────────────────
        check("۲) فایل ناهمخوان رد می‌شود + پیام فارسی با راه‌حل", () -> {
            if (!java.nio.file.Files.exists(bad)) {
                throw new AssertionError("فایل نمونهٔ ناهمخوان موجود نیست: " + bad);
            }
            try {
                LutEmbedding.fromFile(bad, Dims.TEXT_DIM);
                throw new AssertionError("باید استثنا پرتاب می‌شد، اما فایل پذیرفته شد!");
            } catch (java.io.IOException e) {
                String m = e.getMessage();
                if (m == null || !m.contains("ناهمخوان")) {
                    throw new AssertionError("پیام خطا واضح نیست: " + m);
                }
                if (!m.contains("--dim " + Dims.TEXT_DIM)) {
                    throw new AssertionError("راه‌حل در پیام نیست: " + m);
                }
                System.out.println("      → درست رد شد (پیام شامل راه‌حل --dim " + Dims.TEXT_DIM + ")");
            }
        });

        // ── ۳) نگاشت ایمن توکن ─────────────────────────────────
        check("۳) embedToken با شناسهٔ بزرگ‌تر از vocab کرش نمی‌کند", () -> {
            LutEmbedding lut = LutEmbedding.fromFile(good, Dims.TEXT_DIM);
            float[] v = lut.embedToken(999999);
            if (v.length != Dims.TEXT_DIM) {
                throw new AssertionError("طول بردار غلط: " + v.length);
            }
            System.out.println("      → embedToken(999999) → بردار با طول " + v.length);
        });

        // ── ۴) میانگین ایمن ────────────────────────────────────
        check("۴) meanEmbeddingSafe روی توکن‌های خارج از بازه کار می‌کند", () -> {
            LutEmbedding lut = LutEmbedding.fromFile(good, Dims.TEXT_DIM);
            float[] v = lut.meanEmbeddingSafe(new int[]{1, 5000, -3});
            if (v.length != Dims.TEXT_DIM) {
                throw new AssertionError("طول میانگین غلط: " + v.length);
            }
            System.out.println("      → میانگین ۳ توکن (شامل خارج از بازه) → طول " + v.length);
        });

        System.out.println("════════════════════════════════════════════════════════════");
        System.out.println("  نتیجه: " + pass + " موفق، " + fail + " ناموفق");
        System.out.println("════════════════════════════════════════════════════════════");
        if (fail > 0) {
            System.exit(1);
        }
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    private static void check(String name, ThrowingRunnable body) {
        try {
            body.run();
            System.out.println("  ✓ " + name);
            pass++;
        } catch (Throwable t) {
            System.out.println("  ✗ " + name + "  → " + t);
            fail++;
        }
    }
}
