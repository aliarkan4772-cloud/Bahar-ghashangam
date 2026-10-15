package com.pockettts.persian;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * PlanLoader — خواندن فایل plan.tsv (نسخهٔ اصلاح‌شده).
 *
 * <p>فرمت هر خط: {@code متن<TAB>gap} که gap مکث قبل از آن خط به ثانیه است.
 * نخستین chunk کل فایل همیشه gap=0 می‌گیرد و chunkهای میانی gap پیش‌فرض
 * ({@link #DEFAULT_INNER_GAP}) دارند.</p>
 *
 * <p>اصلاحات نسبت به نسخهٔ اصلی:</p>
 * <ul>
 *   <li>اعتبارسنجی خط: خط بدون tab باعث ArrayIndexOutOfBounds نمی‌شود.</li>
 *   <li>مدیریت خطا در parse عدد (به‌جای توقف کل برنامه).</li>
 *   <li>پشتیبانی از خطوط کامنت (شروع با #) و هدر اختیاری.</li>
 *   <li>پارامتری‌کردن gap پیش‌فرض chunkهای میانی.</li>
 *   <li>سازگاری با BOM و CRLF فایل‌های ویندوزی.</li>
 *   <li>ستون‌های اختیاری اضافه: seed و inner_gap.</li>
 * </ul>
 */
public final class PlanLoader {

    /** gap پیش‌فرض بین chunkهای میانی یک خط (ثانیه). */
    public static final double DEFAULT_INNER_GAP = 0.12;

    private PlanLoader() { }

    /** یک کار (job) تولید: متن chunk + مکث قبل از آن. */
    public static final class Job {
        public final String text;
        public final double gap;
        public final long seed;      // -1 یعنی تصادفی

        public Job(String text, double gap, long seed) {
            this.text = text;
            this.gap = gap;
            this.seed = seed;
        }
    }

    /** خواندن plan از یک InputStream (مثلاً assets). */
    public static List<Job> load(InputStream in) throws IOException {
        return load(in, DEFAULT_INNER_GAP);
    }

    /**
     * خواندن plan و تبدیل هر خط به چند Job با استفاده از {@link Chunker}.
     *
     * @param in            جریان ورودی فایل TSV.
     * @param innerGap      gap پیش‌فرض برای chunkهای میانی هر خط.
     */
    public static List<Job> load(InputStream in, double innerGap) throws IOException {
        List<Job> jobs = new ArrayList<>();
        if (in == null) {
            return jobs;
        }
        boolean first = true;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            int lineNo = 0;
            while ((line = br.readLine()) != null) {
                lineNo++;
                // حذف BOM در ابتدای فایل و CR انتهای خط (سازگاری ویندوز)
                if (lineNo == 1 && line.startsWith("\uFEFF")) {
                    line = line.substring(1);
                }
                line = line.replace("\r", "");
                String trimmed = line.trim();
                // رد کردن خطوط خالی و کامنت
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                // جدا کردن ستون‌ها با tab
                String[] cols = line.split("\t", -1);
                String text = cols[0].trim();
                if (text.isEmpty()) {
                    System.err.println("[PlanLoader] خط " + lineNo + " متن خالی دارد؛ رد شد.");
                    continue;
                }
                // ستون gap (اختیاری)
                double gap = 0.0;
                if (cols.length >= 2 && !cols[1].trim().isEmpty()) {
                    try {
                        gap = Double.parseDouble(cols[1].trim());
                    } catch (NumberFormatException e) {
                        System.err.println("[PlanLoader] خط " + lineNo
                                + " مقدار gap نامعتبر («" + cols[1].trim() + "»)؛ صفر در نظر گرفته شد.");
                        gap = 0.0;
                    }
                }
                // ستون seed (اختیاری)
                long seed = -1L;
                if (cols.length >= 3 && !cols[2].trim().isEmpty()) {
                    try {
                        seed = Long.parseLong(cols[2].trim());
                    } catch (NumberFormatException e) {
                        System.err.println("[PlanLoader] خط " + lineNo + " seed نامعتبر؛ تصادفی در نظر گرفته شد.");
                    }
                }
                // ستون inner_gap (اختیاری)
                double lineInnerGap = innerGap;
                if (cols.length >= 4 && !cols[3].trim().isEmpty()) {
                    try {
                        lineInnerGap = Double.parseDouble(cols[3].trim());
                    } catch (NumberFormatException e) {
                        System.err.println("[PlanLoader] خط " + lineNo + " inner_gap نامعتبر؛ پیش‌فرض استفاده شد.");
                    }
                }
                // تقسیم متن خط به chunkها
                List<String> chunks = Chunker.chunk(text);
                for (int ci = 0; ci < chunks.size(); ci++) {
                    double chunkGap;
                    if (first) {
                        chunkGap = 0.0;                 // نخستین chunk کل فایل
                    } else if (ci == 0) {
                        chunkGap = gap;                 // نخستین chunk این خط
                    } else {
                        chunkGap = lineInnerGap;        // chunkهای میانی
                    }
                    jobs.add(new Job(chunks.get(ci), chunkGap, seed));
                    first = false;
                }
            }
        }
        return jobs;
    }
}
