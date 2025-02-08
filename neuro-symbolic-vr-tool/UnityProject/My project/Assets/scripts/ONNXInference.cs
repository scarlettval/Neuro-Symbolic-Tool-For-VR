using UnityEngine;
using Unity.Barracuda;

public class ONNXInference : MonoBehaviour
{
    public NNModel modelAsset;
    private Model runtimeModel;
    private IWorker worker;

    void Start()
    {
        // Load and prepare model
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
    }

    public float[] RunInference(float[] inputData)
    {
        Tensor inputTensor = new Tensor(1, 7, inputData);
        worker.Execute(inputTensor);
        Tensor outputTensor = worker.PeekOutput();
        
        float[] result = outputTensor.AsFloats();
        inputTensor.Dispose();
        outputTensor.Dispose();
        return result;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
