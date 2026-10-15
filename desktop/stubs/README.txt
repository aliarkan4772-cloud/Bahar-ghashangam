# استاب‌های ONNX (فقط برای کامپایل نسخهٔ دسکتاپ)

این کلاس‌ها صرفاً «پوستهٔ خالی» (stub) از API کتابخانهٔ ONNX Runtime هستند تا
کلاس `TtsEngine` (که `import ai.onnxruntime.*` دارد) روی دسکتاپ **کامپایل** شود،
بدون نیاز به فایل JAR واقعی ONNX.

⚠️ این استاب‌ها در زمان اجرا هیچ کاری نمی‌کنند. نسخهٔ دسکتاپ (`DesktopMain`)
از `DesktopTtsEngine` استفاده می‌کند که به ONNX وابسته نیست؛ بنابراین این
استاب‌ها فقط برای کامپایل لازم‌اند.

اگر ONNX Runtime واقعی را دارید، می‌توانید به‌جای این پوشه از JAR آن استفاده کنید:

    javac -cp libs/onnxruntime.jar -d out $(find app/src/main/java desktop/src -name '*.java')
