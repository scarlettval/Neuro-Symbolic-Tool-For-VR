using System;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class MLIntegration : MonoBehaviour
{
    private InferenceSession session;
    public string modelPath = "Assets/Models/VRActionModel.onnx"; // Adjust path as needed

    void Start()
    {
        try
        {
            session = new InferenceSession(modelPath);
            Debug.Log("✅ ONNX Model Loaded Successfully!");
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ Error loading ONNX model: {e.Message}");
        }
    }

    public string PredictAction(float[] inputFeatures)
    {
        if (session == null)
        {
            Debug.LogError("❌ Inference session is null. Make sure the model is loaded.");
            return "unknown";
        }

        // Create input tensor
        var inputTensor = new DenseTensor<float>(inputFeatures, new int[] { 1, inputFeatures.Length });

        // Run inference
        var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) };
        using (var results = session.Run(inputs))
        {
            var output = results[0].AsTensor<float>();
            int predictedIndex = ArgMax(output);
            string predictedAction = GetActionLabel(predictedIndex);
            
            Debug.Log($"🤖 ML Predicted Action: {predictedAction}");
            return predictedAction;
        }
    }

    private int ArgMax(Tensor<float> tensor)
    {
        int maxIndex = 0;
        float maxValue = tensor[0];
        for (int i = 1; i < tensor.Length; i++)
        {
            if (tensor[i] > maxValue)
            {
                maxValue = tensor[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private string GetActionLabel(int index)
    {
        string[] actionLabels = { "move_forward", "jump", "wave_hand", "grab_object", "rotate_left", "rotate_right" };
        return (index >= 0 && index < actionLabels.Length) ? actionLabels[index] : "unknown";
    }

    void OnDestroy()
    {
        session?.Dispose();
    }
}
