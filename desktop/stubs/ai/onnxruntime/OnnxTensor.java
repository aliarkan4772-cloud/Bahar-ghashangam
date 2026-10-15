package ai.onnxruntime;
/** استاب فقط-کامپایل برای نسخهٔ دسکتاپ (بدون ONNX واقعی). */
public class OnnxTensor implements OnnxValue, AutoCloseable {
  public static OnnxTensor createTensor(OrtEnvironment e, java.nio.Buffer b, long[] s){return new OnnxTensor();}
  public static OnnxTensor createTensor(OrtEnvironment e, long[] s){return new OnnxTensor();}
  public Object getValue(){return null;}
  public void close(){}
}
