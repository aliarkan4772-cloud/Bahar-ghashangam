package com.pockettts.persian;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * AudioUtils — ابزارهای سیگنال صوتی و I/O فایل WAV.
 *
 * <p>شامل پورت‌های جاوا از توابع پایتون:</p>
 * <ul>
 *   <li>{@link #speechBounds} — معادل {@code _speech_bounds}: برش سکوت ابتدا/انتها</li>
 *   <li>{@link #compressPauses} — معادل {@code _compress_pauses}: حذف مکث‌های مرده</li>
 *   <li>{@link #solidFraction} — کسر گفتار پیوسته (برای معیار کیفیت)</li>
 *   <li>{@link #deadAirRegions} — یافتن بازه‌های سکوت طولانی</li>
 *   <li>{@link #rms} — جذر میانگین مربعات</li>
 * </ul>
 */
public final class AudioUtils {

    private AudioUtils() { }

    /** جذر میانگین مربعات (RMS) یک بازهٔ نمونه. */
    public static double rms(float[] x, int from, int to) {
        if (to <= from) {
            return 0.0;
        }
        double sum = 0.0;
        for (int i = from; i < to; i++) {
            sum += (double) x[i] * x[i];
        }
        return Math.sqrt(sum / (to - from));
    }

    public static double rms(float[] x) {
        return rms(x, 0, x.length);
    }

    /**
     * معادل {@code _speech_bounds}: نخستین و آخرین نمونه‌ای که انرژی‌اش از
     * آستانهٔ سکوت بیشتر است. سکوت ابتدا/انتها را حذف می‌کند.
     *
     * @return آرایهٔ دوعنصری {start, end} (end انحصاری)
     */
    public static int[] speechBounds(float[] x, double silenceRms) {
        int n = x.length;
        if (n == 0) {
            return new int[]{0, 0};
        }
        // پنجرهٔ ۲۰ میلی‌ثانیه‌ای برای پایداری در برابر نویز
        int win = Math.max(1, (int) (0.020 * TtsEngine.SAMPLE_RATE));
        int start = 0;
        for (int i = 0; i + win <= n; i += win) {
            if (rms(x, i, i + win) > silenceRms) {
                start = i;
                break;
            }
        }
        int end = n;
        for (int i = n - win; i >= 0; i -= win) {
            if (rms(x, i, i + win) > silenceRms) {
                end = i + win;
                break;
            }
        }
        if (end <= start) {
            return new int[]{0, n};
        }
        return new int[]{start, end};
    }

    /**
     * معادل {@code _compress_pauses}: هر بازهٔ سکوت طولانی‌تر از {@code maxPause}
     * را به {@code keep} ثانیه کوتاه می‌کند تا مکث‌های مرده حذف شوند.
     */
    public static float[] compressPauses(float[] x, double maxPause, double keep) {
        int n = x.length;
        if (n == 0) {
            return x;
        }
        int win = Math.max(1, (int) (0.010 * TtsEngine.SAMPLE_RATE));
        double silenceRms = 0.01;
        List<int[]> silences = new ArrayList<>();
        int i = 0;
        while (i + win <= n) {
            if (rms(x, i, i + win) <= silenceRms) {
                int j = i;
                while (j + win <= n && rms(x, j, j + win) <= silenceRms) {
                    j += win;
                }
                double dur = (j - i) / (double) TtsEngine.SAMPLE_RATE;
                if (dur > maxPause) {
                    silences.add(new int[]{i, j});
                }
                i = j;
            } else {
                i += win;
            }
        }
        if (silences.isEmpty()) {
            return x;
        }
        int keepSamples = (int) (keep * TtsEngine.SAMPLE_RATE);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int cursor = 0;
        for (int[] s : silences) {
            appendFloats(out, x, cursor, s[0]);
            appendFloats(out, x, s[0], Math.min(s[0] + keepSamples, s[1]));
            cursor = s[1];
        }
        appendFloats(out, x, cursor, n);
        return toFloats(out.toByteArray());
    }

    /** کسر نمونه‌هایی که «گفتار پیوسته» محسوب می‌شوند (انرژی بالای آستانه). */
    public static double solidFraction(float[] x, double threshold) {
        if (x.length == 0) {
            return 0.0;
        }
        int win = Math.max(1, (int) (0.010 * TtsEngine.SAMPLE_RATE));
        int solid = 0, total = 0;
        for (int i = 0; i + win <= x.length; i += win) {
            total++;
            if (rms(x, i, i + win) > threshold) {
                solid++;
            }
        }
        return total == 0 ? 0.0 : (double) solid / total;
    }

    /** بازه‌های سکوت مرده (طولانی‌تر از {@code minDur} ثانیه) به‌صورت {start,end}. */
    public static List<int[]> deadAirRegions(float[] x, double minDur) {
        List<int[]> regions = new ArrayList<>();
        int n = x.length;
        int win = Math.max(1, (int) (0.010 * TtsEngine.SAMPLE_RATE));
        double silenceRms = 0.01;
        int i = 0;
        while (i + win <= n) {
            if (rms(x, i, i + win) <= silenceRms) {
                int j = i;
                while (j + win <= n && rms(x, j, j + win) <= silenceRms) {
                    j += win;
                }
                if ((j - i) / (double) TtsEngine.SAMPLE_RATE >= minDur) {
                    regions.add(new int[]{i, j});
                }
                i = j;
            } else {
                i += win;
            }
        }
        return regions;
    }

    /** میانهٔ یک آرایه (برای نرمال‌سازی محافظه‌کارانه). */
    public static double median(double[] a) {
        if (a.length == 0) {
            return 0.0;
        }
        double[] c = a.clone();
        java.util.Arrays.sort(c);
        int m = c.length / 2;
        return (c.length % 2 == 1) ? c[m] : 0.5 * (c[m - 1] + c[m]);
    }

    // ---------------------------------------------------------
    // I/O فایل WAV (PCM 16-bit)
    // ---------------------------------------------------------

    /** نوشتن نمونه‌های float در یک فایل WAV با هدر صحیح. */
    public static void writeWav(Path path, float[] samples, int sampleRate) throws IOException {
        byte[] pcm = new byte[samples.length * 2];
        ByteBuffer bb = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
        for (float s : samples) {
            int v = Math.round(Math.max(-1f, Math.min(1f, s)) * 32767f);
            bb.putShort((short) v);
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        writeAscii(out, "RIFF");
        writeIntLE(out, 36 + pcm.length);
        writeAscii(out, "WAVE");
        writeAscii(out, "fmt ");
        writeIntLE(out, 16);
        writeShortLE(out, 1);                 // PCM
        writeShortLE(out, 1);                 // mono
        writeIntLE(out, sampleRate);
        writeIntLE(out, sampleRate * 2);      // byte rate
        writeShortLE(out, 2);                 // block align
        writeShortLE(out, 16);                // bits per sample
        writeAscii(out, "data");
        writeIntLE(out, pcm.length);
        out.write(pcm);
        Files.write(path, out.toByteArray());
    }

    /** خواندن یک فایل WAV مونو ۱۶ بیتی به آرایهٔ float در بازهٔ [-1, 1]. */
    public static float[] readWav(Path path) throws IOException {
        byte[] all = Files.readAllBytes(path);
        ByteBuffer bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
        if (all.length < 44) {
            throw new IOException("فایل WAV خیلی کوچک است");
        }
        // پیمایش سادهٔ chunkها برای یافتن data
        int pos = 12;
        int dataOff = -1, dataLen = 0;
        while (pos + 8 <= all.length) {
            String id = new String(all, pos, 4, java.nio.charset.StandardCharsets.US_ASCII);
            int size = bb.getInt(pos + 4);
            if ("data".equals(id)) {
                dataOff = pos + 8;
                dataLen = size;
                break;
            }
            pos += 8 + size + (size % 2);
        }
        if (dataOff < 0) {
            throw new IOException("chunk داده در WAV یافت نشد");
        }
        int count = dataLen / 2;
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            short v = bb.getShort(dataOff + i * 2);
            out[i] = v / 32768f;
        }
        return out;
    }

    // ---------------------------------------------------------
    // کمکی‌های داخلی
    // ---------------------------------------------------------

    private static void appendFloats(ByteArrayOutputStream out, float[] x, int from, int to) {
        for (int i = from; i < to; i++) {
            int bits = Float.floatToIntBits(x[i]);
            out.write(bits & 0xFF);
            out.write((bits >> 8) & 0xFF);
            out.write((bits >> 16) & 0xFF);
            out.write((bits >> 24) & 0xFF);
        }
    }

    private static float[] toFloats(byte[] b) {
        float[] out = new float[b.length / 4];
        for (int i = 0; i < out.length; i++) {
            int bits = (b[i * 4] & 0xFF)
                    | ((b[i * 4 + 1] & 0xFF) << 8)
                    | ((b[i * 4 + 2] & 0xFF) << 16)
                    | ((b[i * 4 + 3] & 0xFF) << 24);
            out[i] = Float.intBitsToFloat(bits);
        }
        return out;
    }

    private static void writeAscii(ByteArrayOutputStream o, String s) {
        for (int i = 0; i < s.length(); i++) {
            o.write(s.charAt(i));
        }
    }

    private static void writeIntLE(ByteArrayOutputStream o, int v) {
        o.write(v & 0xFF);
        o.write((v >> 8) & 0xFF);
        o.write((v >> 16) & 0xFF);
        o.write((v >> 24) & 0xFF);
    }

    private static void writeShortLE(ByteArrayOutputStream o, int v) {
        o.write(v & 0xFF);
        o.write((v >> 8) & 0xFF);
    }
}
