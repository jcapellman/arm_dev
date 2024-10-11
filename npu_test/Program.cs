using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace npu_test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Model Name and Input Data are required inputs");

                return;
            }

            var modelName = args[0];
            var inputFileName = args[1];

            // Create Session options
            using var _sessionOptions = new SessionOptions();
            Dictionary<string, string> config = new()
            {
                { "backend_path", "QnnHtp.dll"},
                { "enable_htp_fp16_precision", "1"}
            };

            _sessionOptions.AppendExecutionProvider("QNN", config);

            using var _encoderSession = new InferenceSession(modelName, _sessionOptions);
            var inputTensor = new DenseTensor<float>(new float[] { /* input data */ }, new int[] { /* dimensions */ });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

            using var results = _encoderSession.Run(inputs);
            var output = results[0].AsTensor<float>().ToArray();
        }
    }
}