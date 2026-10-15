# proguard-rules.pro — قواعد نگه‌داشتن کلاس‌های ONNX Runtime
# ONNX Runtime از JNI استفاده می‌کند؛ نام کلاس‌ها نباید مبهم شوند.
-keep class ai.onnxruntime.** { *; }
-keep class com.pockettts.persian.** { *; }
-dontwarn ai.onnxruntime.**
