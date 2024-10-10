using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace npu_test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Create Session options
            using var _sessionOptions = new SessionOptions();
            Dictionary<string, string> config = new Dictionary<string, string> {
                { "backend_path", "QnnHtp.dll"},
                { "enable_htp_fp16_precision", "1"}
            };

            _sessionOptions.AppendExecutionProvider("QNN", config);

            using var _encoderSession = new InferenceSession(args[0], _sessionOptions);
            var inputTensor = new DenseTensor<float>(new float[] { /* input data */ }, new int[] { /* dimensions */ });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>().ToArray();
        }
    }
}