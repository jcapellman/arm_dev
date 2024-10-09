using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace npu_test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var options = new SessionOptions();
            options.AppendExecutionProvider("QNNExecutionProvider");
            options.ProfileOutputPathPrefix = Path.Combine(AppContext.BaseDirectory, "QnnHtp.dll");
            
            using var session = new InferenceSession("model.onnx", options);
            var inputTensor = new DenseTensor<float>(new float[] { /* input data */ }, new int[] { /* dimensions */ });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>().ToArray();
        }
    }
}