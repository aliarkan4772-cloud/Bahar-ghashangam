package com.pockettts.persian;

import java.util.ArrayList;
import java.util.List;

/**
 * G2P — تبدیل متن فارسی به واج (Grapheme-to-Phoneme).
 *
 * <p>پایپ‌لاین: متن فارسی → G2P → واج‌ها → LUT embedding → FlowLM → Mimi → WAV.
 * این کلاس یک پیاده‌سازی ساده و قابل‌جایگزینی است که نویسه‌های فارسی را به
 * نمایش واجی لاتین (همان قرارداد پروژهٔ پایتون) نگاشت می‌کند. برای دقت بالاتر،
 * این کلاس را با مدل G2P واقعی جایگزین کنید.</p>
 */
public final class G2P {

    private G2P() { }

    /**
     * تبدیل یک متن فارسی به رشتهٔ واجی.
     * خروجی، واژه‌ها با فاصله جدا شده‌اند و هر واژه به واج‌های لاتین شکسته شده است.
     */
    public static String toPhonemes(String text) {
        if (text == null) {
            return "";
        }
        List<String> out = new ArrayList<>();
        for (String word : text.trim().split("\\s+")) {
            if (word.isEmpty()) {
                continue;
            }
            out.add(wordToPhonemes(word));
        }
        return String.join(" ", out);
    }

    /** نگاشت سادهٔ نویسه‌به‌واج برای یک واژه. */
    private static String wordToPhonemes(String word) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            switch (c) {
                case 'ا': case 'آ': sb.append('A'); break;
                case 'ب': sb.append('b'); break;
                case 'پ': sb.append('p'); break;
                case 'ت': sb.append('t'); break;
                case 'ث': sb.append('s'); break;
                case 'ج': sb.append('j'); break;
                case 'چ': sb.append('C'); break;
                case 'ح': case 'ه': sb.append('h'); break;
                case 'خ': sb.append('x'); break;
                case 'د': sb.append('d'); break;
                case 'ذ': case 'ز': case 'ض': case 'ظ': sb.append('z'); break;
                case 'ر': sb.append('r'); break;
                case 'ژ': sb.append('Z'); break;
                case 'س': case 'ص': sb.append('s'); break;
                case 'ش': sb.append('S'); break;
                case 'ط': sb.append('t'); break;
                case 'ع': case 'ء': sb.append('?'); break;
                case 'غ': sb.append('q'); break;
                case 'ف': sb.append('f'); break;
                case 'ق': sb.append('q'); break;
                case 'ک': sb.append('k'); break;
                case 'گ': sb.append('g'); break;
                case 'ل': sb.append('l'); break;
                case 'م': sb.append('m'); break;
                case 'ن': sb.append('n'); break;
                case 'و': sb.append('v'); break;
                case 'ی': sb.append('y'); break;
                case ' ': sb.append(' '); break;
                default:
                    // نویسه‌های لاتین/عدد/نشانه‌گذاری همان‌طور عبور می‌کنند
                    sb.append(c);
            }
        }
        return sb.toString();
    }
}
