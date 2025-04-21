using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class PythonSocketListener : MonoBehaviour
{
    private TcpListener server;
    public ScreenshotVR screenshotScript;  // Drag your ScreenshotVR script here via Inspector

    void Start()
    {
        server = new TcpListener(IPAddress.Any, 5050);
        server.Start();
        Debug.Log("✅ Unity TCP server started on port 5050.");
        InvokeRepeating(nameof(Listen), 1f, 0.5f);
    }

    void Listen()
    {
        if (!server.Pending()) return;

        TcpClient client = server.AcceptTcpClient();
        NetworkStream stream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];
        int bytesRead = stream.Read(buffer, 0, buffer.Length);
        string msg = Encoding.UTF8.GetString(buffer, 0, bytesRead);
        Debug.Log($"📩 Received from Python: {msg}");
        HandleMessage(msg);
        client.Close();
    }

    void HandleMessage(string json)
    {
        try
        {
            JObject data = JObject.Parse(json);

            if (data.ContainsKey("screenshot") && (bool)data["screenshot"] == true)
            {
                var screenshotComponent = FindObjectOfType<ScreenshotVR>();
                if (screenshotComponent != null)
                {
                    screenshotComponent.TakeScreenshot();
                    Debug.Log("✅ Screenshot triggered by Python command.");
                }
                else
                {
                    Debug.LogError("❌ ScreenshotVR component not found in scene.");
                }
                return;
            }

            // [Your existing movement code here]
        }
        catch (Exception e)
        {
            Debug.LogError($"JSON Parse error: {e.Message}");
        }
    }
    void OnApplicationQuit()
    {
        server.Stop();
    }
}
