package com.pockettts.persian;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * DesktopMain — اجرای نسخهٔ دسکتاپ موتور TTS فارسی (بدون اندروید، بدون ONNX).
 * ============================================================
 * <p>این کلاس متن فارسی می‌گیرد، آن را به chunk تقسیم می‌کند، هر chunk را با
 * {@link DesktopTtsEngine} (که از {@link LutEmbedding} استفاده می‌کند) سنتز
 * می‌کند، قطعات را با مکث‌های plan به هم می‌چسباند و یک فایل WAV می‌نویسد.</p>
 *
 * <p><b>اجرا:</b></p>
 * <pre>
 *   javac -d out $(find app/src/main/java desktop/src -name '*.java')
 *   java -cp out com.pockettts.persian.DesktopMain \
 *        --lut app/src/main/assets/models/lut_weight.npy \
 *        --out out.wav \
 *        --text "سلام، این یک آزمایش است."
 * </pre>
 *
 * <p>در پایان، هدر WAV و تعداد نمونه‌ها اعتبارسنجی و گزارش می‌شوند.</p>
 * ============================================================
 */
public final class DesktopMain {

    public static void main(String[] args) throws Exception {
        String lutPath = "app/src/main/assets/models/lut_weight.npy";
        String outPath = "out.wav";
        String text = "سلام، این یک آزمایش موتور تبدیل متن به گفتار فارسی است.";
        int threads = 4;
        boolean parallel = true;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--lut":      lutPath = args[++i]; break;
                case "--out":      outPath = args[++i]; break;
                case "--text":     text = args[++i]; break;
                case "--threads":  threads = Integer.parseInt(args[++i]); break;
                case "--serial":   parallel = false; break;
                default:
                    System.err.println("آرگومان ناشناخته: " + args[i]);
            }
        }

        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("  Pocket-TTS فارسی — نسخهٔ دسکتاپ");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("متن ورودی: «" + text + "»");

        // ── ۱) بارگذاری جدول embedding ──
        Path lutFile = Paths.get(lutPath);
        System.out.println("\n[۱] بارگذاری LutEmbedding از: " + lutFile.toAbsolutePath());
        // بُعد مورد انتظار از Dims.TEXT_DIM (که باید با text_dim در model_contract.json یکی باشد)
        System.out.println("    بُعد مورد انتظار (Dims.TEXT_DIM) = " + Dims.TEXT_DIM);
        LutEmbedding lut = LutEmbedding.fromFile(lutFile, Dims.TEXT_DIM);
        System.out.println("    ✓ vocab=" + lut.vocabSize() + "  dim=" + lut.dim()
                + "  dataOffset=" + lut.dataOffset() + "  منبع=" + lut.source());

        // ── ۲) تقسیم متن به chunk ──
        List<String> chunks = Chunker.chunk(text);
        System.out.println("\n[۲] تقسیم متن به chunk: " + chunks.size() + " قطعه");
        for (int i = 0; i < chunks.size(); i++) {
            System.out.println("    chunk[" + i + "] = «" + chunks.get(i) + "»");
        }

        // ── ۳) ساخت jobها با gap ──
        List<PlanLoader.Job> jobs = TtsEngine.jobsFromText(text);
        System.out.println("\n[۳] ساخت " + jobs.size() + " job با مکث‌های پیش‌فرض");

        // ── ۴) سنتز ──
        DesktopTtsEngine engine = new DesktopTtsEngine(lut, TtsEngine.SAMPLE_RATE);
        float[][] voiceCache = new float[TtsEngine.VOICE_OFF][TtsEngine.LDIM];

        System.out.println("\n[۴] سنتز ...");
        long t0 = System.currentTimeMillis();
        List<float[]> audios = new java.util.ArrayList<>();
        for (int i = 0; i < jobs.size(); i++) {
            PlanLoader.Job job = jobs.get(i);
            float[] a = engine.synthesizeChunk(job.text, voiceCache, TtsEngine.seedFor(job, i));
            audios.add(a);
            System.out.println("    job[" + i + "] «" + job.text + "» → "
                    + a.length + " نمونه (" + String.format("%.2f", a.length / (double) TtsEngine.SAMPLE_RATE) + " ثانیه)");
        }
        long serialMs = System.currentTimeMillis() - t0;

        // ── ۵) gain سراسری + چسباندن با gap ──
        TtsEngine.applyGlobalGain(audios);
        float[] audio = TtsEngine.stitchWithGaps(audios, TtsEngine.gapsOf(jobs));

        // ── ۶) نوشتن WAV ──
        Path out = Paths.get(outPath);
        AudioUtils.writeWav(out, audio, TtsEngine.SAMPLE_RATE);
        System.out.println("\n[۵] نوشتن WAV: " + out.toAbsolutePath());
        System.out.println("    زمان سنتز سری: " + serialMs + " ms");

        // ── ۷) اعتبارسنجی WAV ──
        System.out.println("\n[۶] اعتبارسنجی فایل WAV ...");
        verifyWav(out, audio.length);

        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("  ✓ تمام شد. فایل خروجی: " + out.toAbsolutePath());
        System.out.println("════════════════════════════════════════════════════════");
    }

    /** اعتبارسنجی هدر WAV و تعداد نمونه‌ها. */
    private static void verifyWav(Path path, int expectedSamples) throws Exception {
        byte[] all = Files.readAllBytes(path);
        check(all.length >= 44, "فایل WAV باید حداقل ۴۴ بایت (هدر) باشد");
        check(ascii(all, 0, 4).equals("RIFF"), "امضای RIFF");
        check(ascii(all, 8, 4).equals("WAVE"), "امضای WAVE");
        check(ascii(all, 12, 4).equals("fmt "), "chunk fmt");
        check(ascii(all, 36, 4).equals("data"), "chunk data");

        int riffSize = le32(all, 4);
        int dataSize = le32(all, 40);
        int sampleRate = le32(all, 24);
        int channels = le16(all, 22);
        int bits = le16(all, 34);

        check(riffSize == all.length - 8, "اندازهٔ RIFF با طول فایل هم‌خوان باشد");
        check(dataSize == all.length - 44, "اندازهٔ data با طول فایل هم‌خوان باشد");
        check(sampleRate == TtsEngine.SAMPLE_RATE, "نرخ نمونه‌برداری = " + TtsEngine.SAMPLE_RATE);
        check(channels == 1, "مونو");
        check(bits == 16, "۱۶ بیتی");

        int samples = dataSize / 2;
        check(samples == expectedSamples,
                "تعداد نمونه‌ها = " + samples + " (انتظار " + expectedSamples + ")");

        System.out.println("    ✓ RIFF/WAVE/fmt/data معتبر");
        System.out.println("    ✓ نرخ=" + sampleRate + " Hz، کانال=" + channels + "، بیت=" + bits);
        System.out.println("    ✓ تعداد نمونه=" + samples + "  ("
                + String.format("%.2f", samples / (double) sampleRate) + " ثانیه)");
        System.out.println("    ✓ حجم فایل=" + all.length + " بایت");
    }

    private static void check(boolean cond, String what) {
        if (!cond) {
            throw new IllegalStateException("اعتبارسنجی WAV شکست خورد: " + what);
        }
    }

    private static String ascii(byte[] b, int off, int len) {
        return new String(b, off, len, java.nio.charset.StandardCharsets.US_ASCII);
    }

    private static int le32(byte[] b, int off) {
        return (b[off] & 0xFF) | ((b[off + 1] & 0xFF) << 8)
                | ((b[off + 2] & 0xFF) << 16) | ((b[off + 3] & 0xFF) << 24);
    }

    private static int le16(byte[] b, int off) {
        return (b[off] & 0xFF) | ((b[off + 1] & 0xFF) << 8);
    }
}
