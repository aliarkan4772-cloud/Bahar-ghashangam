[1mdiff --git a/app/src/main/AndroidManifest.xml b/app/src/main/AndroidManifest.xml[m
[1mindex c808866..4b67edb 100644[m
[1m--- a/app/src/main/AndroidManifest.xml[m
[1m+++ b/app/src/main/AndroidManifest.xml[m
[36m@@ -1,8 +1,12 @@[m
 <?xml version="1.0" encoding="utf-8"?>[m
 <!-- AndroidManifest.xml — تعریف اپلیکیشن و Activity اصلی -->[m
[31m-<manifest xmlns:android="http://schemas.android.com/apk/res/android">[m
[32m+[m[32m<manifest xmlns:android="http://schemas.android.com/apk/res/android"[m
[32m+[m[32m    xmlns:tools="http://schemas.android.com/tools">[m
 [m
     <!-- برای ذخیرهٔ فایل WAV خروجی در حافظهٔ خارجی (اندروید ۹ و پایین‌تر) -->[m
[32m+[m[32m    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />[m
[32m+[m[32m    <uses-permission android:name="android.permission.MANAGE_EXTERNAL_STORAGE"[m
[32m+[m[32m        tools:ignore="ScopedStorage" />[m
     <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"[m
         android:maxSdkVersion="28" />[m
 [m
[1mdiff --git a/app/src/main/java/com/pockettts/persian/MainActivity.java b/app/src/main/java/com/pockettts/persian/MainActivity.java[m
[1mindex 09bff02..303ab32 100644[m
[1m--- a/app/src/main/java/com/pockettts/persian/MainActivity.java[m
[1m+++ b/app/src/main/java/com/pockettts/persian/MainActivity.java[m
[36m@@ -3,7 +3,12 @@[m [mpackage com.pockettts.persian;[m
 import android.media.MediaPlayer;[m
 import android.os.Bundle;[m
 import android.os.Handler;[m
[32m+[m[32mimport android.content.Intent;[m
[32m+[m[32mimport android.net.Uri;[m
[32m+[m[32mimport android.os.Build;[m
[32m+[m[32mimport android.os.Environment;[m
 import android.os.Looper;[m
[32m+[m[32mimport android.provider.Settings;[m
 import android.widget.Button;[m
 import android.widget.EditText;[m
 import android.widget.SeekBar;[m
[36m@@ -88,7 +93,8 @@[m [mpublic class MainActivity extends AppCompatActivity {[m
             @Override public void onStopTrackingTouch(SeekBar sb) { }[m
         });[m
 [m
[31m-        Path modelDir = new File(getFilesDir(), "models").toPath();[m
[32m+[m[32m        Path modelDir = new File(Environment.getExternalStoragePublicDirectory([m
[32m+[m[32m                Environment.DIRECTORY_DOWNLOADS), "models").toPath();[m
 [m
         btnSpeak.setOnClickListener(v -> onSpeak(modelDir));[m
         btnBenchmark.setOnClickListener(v -> onBenchmark(modelDir));[m
[36m@@ -99,7 +105,23 @@[m [mpublic class MainActivity extends AppCompatActivity {[m
         setStatus(getString(R.string.status_loading));[m
         executor.execute(() -> {[m
             try {[m
[31m-                // ۱) استخراج فایل‌های مدل از assets به حافظهٔ داخلی (یک‌بار)[m
[32m+[m[32m                // ۰) بررسی دسترسی به حافظهٔ خارجی (اندروید ۱۱+)[m
[32m+[m[32m                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R[m
[32m+[m[32m                        && !Environment.isExternalStorageManager()) {[m
[32m+[m[32m                    ui.post(() -> {[m
[32m+[m[32m                        setStatus("برای خواندن مدل‌ها از Downloads، دسترسی «همه فایل‌ها» لازم است. در صفحهٔ باز شده این دسترسی را فعال کنید و اپ را دوباره باز کنید.");[m
[32m+[m[32m                        try {[m
[32m+[m[32m                            Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);[m
[32m+[m[32m                            intent.setData(Uri.parse("package:" + getPackageName()));[m
[32m+[m[32m                            startActivity(intent);[m
[32m+[m[32m                        } catch (Exception ex) {[m
[32m+[m[32m                            System.err.println("cannot open settings: " + ex.getMessage());[m
[32m+[m[32m                        }[m
[32m+[m[32m                    });[m
[32m+[m[32m                    return;[m
[32m+[m[32m                }[m
[32m+[m
[32m+[m[32m                // ۱) استخراج reference_01.wav از assets[m
                 extractModelsFromAssets(modelDir);[m
 [m
                 // ۲) ساخت موتور و بارگذاری جدول embedding توکن‌ها (lut_weight.npy)[m
[36m@@ -107,7 +129,7 @@[m [mpublic class MainActivity extends AppCompatActivity {[m
                 // نکته: پیش‌تر خطای LUT فقط در console چاپ می‌شد و کاربر در اپ[m
                 // پیام واضحی نمی‌دید. حالا خطا هم چاپ می‌شود و هم در UI نشان داده می‌شود.[m
                 try {[m
[31m-                    engine.loadLutFromAssets(getAssets(), Dims.LUT_ASSET);[m
[32m+[m[32m                    engine.loadLut(modelDir.resolve("lut_weight.npy"));[m
                 } catch (Exception le) {[m
                     final String lutError = le.getMessage();[m
                     System.err.println("[MainActivity] LUT بارگذاری نشد: " + lutError);[m
[36m@@ -134,10 +156,8 @@[m [mpublic class MainActivity extends AppCompatActivity {[m
         if (!Files.exists(modelDir)) {[m
             Files.createDirectories(modelDir);[m
         }[m
[31m-        String[] names = {[m
[31m-                "flow_lm_step.onnx", "mimi_decoder_step_kv.onnx", "mimi_encoder.onnx",[m
[31m-                "reference_01.wav", "lut_weight.npy", "model_contract.json"[m
[31m-        };[m
[32m+[m[32m        // بقیهٔ فایل‌های مدل از قبل در Downloads/models قرار دارند؛ فقط reference_01.wav را استخراج می‌کنیم.[m
[32m+[m[32m        String[] names = { "reference_01.wav" };[m
         for (String name : names) {[m
             Path dst = modelDir.resolve(name);[m
             if (Files.exists(dst)) {[m
