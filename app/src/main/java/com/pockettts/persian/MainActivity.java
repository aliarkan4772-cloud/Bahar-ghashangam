package com.pockettts.persian;

import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * MainActivity — رابط کاربری موتور TTS فارسی.
 * ============================================================
 * <p>امکانات:</p>
 * <ul>
 *   <li>ورودی متن فارسی + دکمهٔ «تبدیل به گفتار»</li>
 *   <li>کلید «حالت موازی» و انتخاب تعداد thread (۱ تا ۸)</li>
 *   <li>دکمهٔ «مقایسهٔ سری و موازی» برای نمایش زمان هر دو حالت</li>
 *   <li>پخش خروجی WAV با MediaPlayer</li>
 * </ul>
 *
 * <p>مدل‌ها در {@code getFilesDir()/models} قرار می‌گیرند و در
 * پس‌زمینه بارگذاری می‌شوند تا UI قفل نشود.</p>
 * ============================================================
 */
public class MainActivity extends AppCompatActivity {

    private EditText editText;
    private Button btnSpeak, btnPlay, btnStop, btnBenchmark;
    private TextView statusText, timingText, threadLabel;
    private SeekBar seekThreads;
    private Switch switchParallel;

    private TtsEngine engine;
    private MediaPlayer player;
    private File outputWav;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        editText = findViewById(R.id.editText);
        btnSpeak = findViewById(R.id.btnSpeak);
        btnPlay = findViewById(R.id.btnPlay);
        btnStop = findViewById(R.id.btnStop);
        btnBenchmark = findViewById(R.id.btnBenchmark);
        statusText = findViewById(R.id.statusText);
        timingText = findViewById(R.id.timingText);
        seekThreads = findViewById(R.id.seekThreads);
        threadLabel = findViewById(R.id.threadLabel);
        switchParallel = findViewById(R.id.switchParallel);

        // ── تنظیم تعداد thread ──
        seekThreads.setMax(ParallelSynthesizer.MAX_THREADS - 1);   // ۰..۷ → ۱..۸
        seekThreads.setProgress(ParallelSynthesizer.DEFAULT_THREADS - 1);
        updateThreadLabel();

        switchParallel.setOnCheckedChangeListener((v, checked) -> {
            seekThreads.setEnabled(checked);
            updateThreadLabel();
        });

        seekThreads.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar sb, int progress, boolean fromUser) {
                updateThreadLabel();
            }

            @Override public void onStartTrackingTouch(SeekBar sb) { }

            @Override public void onStopTrackingTouch(SeekBar sb) { }
        });

        Path modelDir = new File(getFilesDir(), "models").toPath();

        btnSpeak.setOnClickListener(v -> onSpeak(modelDir));
        btnBenchmark.setOnClickListener(v -> onBenchmark(modelDir));
        btnPlay.setOnClickListener(v -> onPlay());
        btnStop.setOnClickListener(v -> onStop());

        // ── بارگذاری مدل‌ها و LUT در پس‌زمینه ──
        setStatus(getString(R.string.status_loading));
        executor.execute(() -> {
            try {
                // ۱) استخراج فایل‌های مدل از assets به حافظهٔ داخلی (یک‌بار)
                extractModelsFromAssets(modelDir);

                // ۲) ساخت موتور و بارگذاری جدول embedding توکن‌ها (lut_weight.npy)
                engine = new TtsEngine(modelDir);
                // نکته: پیش‌تر خطای LUT فقط در console چاپ می‌شد و کاربر در اپ
                // پیام واضحی نمی‌دید. حالا خطا هم چاپ می‌شود و هم در UI نشان داده می‌شود.
                try {
                    engine.loadLutFromAssets(getAssets(), Dims.LUT_ASSET);
                } catch (Exception le) {
                    final String lutError = le.getMessage();
                    System.err.println("[MainActivity] LUT بارگذاری نشد: " + lutError);
                    ui.post(() -> setStatus(getString(R.string.status_lut_error) + "\n" + lutError));
                }

                // ۳) بارگذاری مدل‌های ONNX
                engine.load();
                ui.post(() -> setStatus(getString(R.string.status_idle)));
            } catch (Exception e) {
                ui.post(() -> setStatus(getString(R.string.status_error) + ": " + e.getMessage()));
            }
        });
    }

    /**
     * کپی فایل‌های مدل از {@code assets/models} به {@code getFilesDir()/models}.
     *
     * <p>فایل‌های داخل assets در APK فشرده‌اند و مسیر فایل واقعی ندارند؛
     * بنابراین ONNX Runtime نمی‌تواند مستقیم از آن‌ها بخواند و باید یک‌بار
     * به حافظهٔ داخلی استخراج شوند. فایل‌های موجود دوباره کپی نمی‌شوند.</p>
     */
    private void extractModelsFromAssets(Path modelDir) throws Exception {
        if (!Files.exists(modelDir)) {
            Files.createDirectories(modelDir);
        }
        String[] names = {
                "flow_lm_step.onnx", "mimi_decoder_step_kv.onnx", "mimi_encoder.onnx",
                "reference_01.wav", "lut_weight.npy", "model_contract.json"
        };
        for (String name : names) {
            Path dst = modelDir.resolve(name);
            if (Files.exists(dst)) {
                continue;   // قبلاً استخراج شده است
            }
            try (InputStream in = getAssets().open("models/" + name)) {
                Files.copy(in, dst);
            } catch (Exception e) {
                System.err.println("[MainActivity] استخراج " + name + " ناموفق: " + e.getMessage());
            }
        }
    }

    /** تعداد thread انتخاب‌شده (۱ تا ۸). */
    private int selectedThreads() {
        return ParallelSynthesizer.clampThreads(seekThreads.getProgress() + 1);
    }

    private boolean parallelEnabled() {
        return switchParallel.isChecked();
    }

    private void updateThreadLabel() {
        if (parallelEnabled()) {
            threadLabel.setText("تعداد thread: " + selectedThreads());
        } else {
            threadLabel.setText("حالت سری فعال است");
        }
    }

    // ---------------------------------------------------------
    // تبدیل به گفتار
    // ---------------------------------------------------------

    private void onSpeak(Path modelDir) {
        String text = editText.getText().toString().trim();
        if (text.isEmpty()) {
            Toast.makeText(this, R.string.msg_empty_text, Toast.LENGTH_SHORT).show();
            return;
        }
        if (engine == null || !engine.isReady()) {
            Toast.makeText(this, R.string.msg_engine_not_ready, Toast.LENGTH_SHORT).show();
            return;
        }

        final boolean parallel = parallelEnabled();
        final int threads = selectedThreads();

        setStatus(getString(R.string.status_generating));
        timingText.setText("");

        executor.execute(() -> {
            try {
                Path ref = modelDir.resolve("reference_01.wav");
                float[][] voiceCache = engine.voiceCache(ref);
                List<PlanLoader.Job> jobs = TtsEngine.jobsFromText(text);

                SynthesisResult result = parallel
                        ? synthesizeParallel(jobs, voiceCache, threads, modelDir)
                        : engine.synthesizeSerial(jobs, voiceCache);

                outputWav = new File(getCacheDir(), "output.wav");
                AudioUtils.writeWav(outputWav.toPath(), result.audio, TtsEngine.SAMPLE_RATE);

                final String summary = result.describe();
                ui.post(() -> {
                    setStatus(getString(R.string.status_done));
                    timingText.setText(summary + "\n" + formatSeconds(result));
                    onPlay();
                });
            } catch (Exception e) {
                ui.post(() -> setStatus(getString(R.string.status_error) + ": " + e.getMessage()));
            }
        });
    }

    /** اجرای حالت موازی با ساخت نمونهٔ مستقل موتور برای هر worker. */
    private SynthesisResult synthesizeParallel(List<PlanLoader.Job> jobs, float[][] voiceCache,
                                               int threads, Path modelDir) {
        ParallelSynthesizer synth = new ParallelSynthesizer(threads)
                .withVoiceCache(voiceCache);

        // هر worker یک نمونهٔ TtsEngine مستقل می‌سازد → بدون قفل، بدون دادهٔ مشترک
        ParallelSynthesizer.Outcome out = synth.synthesize(
                jobs,
                () -> {
                    TtsEngine e = new TtsEngine(modelDir);
                    e.load();
                    return e;
                },
                engine,
                threads);

        for (ParallelSynthesizer.Failure f : out.failures) {
            System.err.println("[MainActivity] " + f);
        }
        return out.stats;
    }

    // ---------------------------------------------------------
    // آزمون مقایسه‌ای سری در برابر موازی
    // ---------------------------------------------------------

    private void onBenchmark(Path modelDir) {
        String text = editText.getText().toString().trim();
        if (text.isEmpty()) {
            Toast.makeText(this, R.string.msg_empty_text, Toast.LENGTH_SHORT).show();
            return;
        }
        if (engine == null || !engine.isReady()) {
            Toast.makeText(this, R.string.msg_engine_not_ready, Toast.LENGTH_SHORT).show();
            return;
        }

        final int threads = selectedThreads();
        setStatus(getString(R.string.status_benchmark));
        timingText.setText("");

        executor.execute(() -> {
            try {
                Path ref = modelDir.resolve("reference_01.wav");
                float[][] voiceCache = engine.voiceCache(ref);
                List<PlanLoader.Job> jobs = TtsEngine.jobsFromText(text);

                // ۱) اجرای سری
                SynthesisResult serial = engine.synthesizeSerial(jobs, voiceCache);

                // ۲) اجرای موازی
                SynthesisResult par = synthesizeParallel(jobs, voiceCache, threads, modelDir);

                double speedup = par.millis > 0
                        ? (serial.millis / (double) par.millis) : 0.0;

                StringBuilder sb = new StringBuilder();
                sb.append("سری:     ").append(serial.millis).append(" ms\n");
                sb.append("موازی:  ").append(par.millis).append(" ms  (")
                        .append(par.workers).append(" thread)\n");
                sb.append("سرعت نسبی: ").append(String.format("%.2f", speedup)).append("×\n");
                sb.append(formatSeconds(par));

                // خروجی نهایی = نسخهٔ موازی
                outputWav = new File(getCacheDir(), "output.wav");
                AudioUtils.writeWav(outputWav.toPath(), par.audio, TtsEngine.SAMPLE_RATE);

                final String report = sb.toString();
                ui.post(() -> {
                    setStatus(getString(R.string.status_done));
                    timingText.setText(report);
                });
            } catch (Exception e) {
                ui.post(() -> setStatus(getString(R.string.status_error) + ": " + e.getMessage()));
            }
        });
    }

    private static String formatSeconds(SynthesisResult r) {
        double sec = r.audio.length / (double) TtsEngine.SAMPLE_RATE;
        return String.format("طول صدا: %.2f ثانیه · %d نمونه", sec, r.audio.length);
    }

    // ---------------------------------------------------------
    // پخش
    // ---------------------------------------------------------

    private void onPlay() {
        if (outputWav == null || !outputWav.exists()) {
            Toast.makeText(this, R.string.msg_generate_first, Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            onStop();
            player = new MediaPlayer();
            player.setDataSource(outputWav.getAbsolutePath());
            player.prepare();
            player.start();
        } catch (Exception e) {
            Toast.makeText(this, "خطا در پخش: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    protected void onStop() {
        if (player != null) {
            try {
                player.stop();
            } catch (Exception ignored) { }
            player.release();
            player = null;
        }
    }

    private void setStatus(String s) {
        statusText.setText(s);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        onStop();
        executor.shutdownNow();
        if (engine != null) {
            engine.close();
        }
    }
}
