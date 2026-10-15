package ai.onnxruntime;
/** استاب فقط-کامپایل برای نسخهٔ دسکتاپ (بدون ONNX واقعی). */
public class OrtEnvironment implements AutoCloseable {
  public static OrtEnvironment getEnvironment(){return new OrtEnvironment();}
  public OrtSession createSession(String p, OrtSession.SessionOptions o){return null;}
  public void close(){} }
