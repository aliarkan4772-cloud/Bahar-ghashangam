package ai.onnxruntime;
/** استاب فقط-کامپایل برای نسخهٔ دسکتاپ (بدون ONNX واقعی). */
public class OrtSession implements AutoCloseable {
  public static class SessionOptions implements AutoCloseable {
    public void setIntraOpNumThreads(int n){} public void close(){} }
  public static class Result implements AutoCloseable {
    public OnnxValue get(int i){return null;}
    public java.util.Optional<OnnxValue> get(String n){return java.util.Optional.empty();}
    public void close(){} }
  public Result run(java.util.Map<String,OnnxTensor> f){return null;}
  public void close(){}
}
