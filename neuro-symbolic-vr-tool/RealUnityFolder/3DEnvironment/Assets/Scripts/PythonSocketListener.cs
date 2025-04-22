using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class PythonSocketListener : MonoBehaviour
{
    private TcpListener server;
    private string screenshotPath;

    void Start()
    {
        server = new TcpListener(IPAddress.Any, 5050);
        server.Start();
        screenshotPath = Path.Combine(Application.dataPath, "Snapshots/vr_snapshot.png");
        Debug.Log("✅ Unity TCP server started on port 5050.");
        InvokeRepeating(nameof(Listen), 1f, 0.5f);
    }

    void Listen()
    {
        if (!server.Pending())
            return;

        TcpClient client = server.AcceptTcpClient();
        NetworkStream stream = client.GetStream();

        byte[] buffer = new byte[client.ReceiveBufferSize];
        int bytesRead = stream.Read(buffer, 0, buffer.Length);
        string json = Encoding.UTF8.GetString(buffer, 0, bytesRead);

        Debug.Log("📩 Received from Python: " + json);

        try
        {
            JObject parsed = JObject.Parse(json);

            // === Handle screenshot request ===
            if (parsed["screenshot"] != null && parsed["screenshot"].ToObject<bool>())
            {
                Debug.Log("📸 Screenshot command received");
                Directory.CreateDirectory(Path.GetDirectoryName(screenshotPath));
                ScreenCapture.CaptureScreenshot(screenshotPath);
                Debug.Log($"✅ Screenshot saved to: {screenshotPath}");
                return;
            }

            // === Handle object actions ===
            string action = parsed["action"]?.ToString();
            string objectName = parsed["object"]?.ToString();

            GameObject obj = GameObject.Find(objectName);
            if (obj == null)
            {
                Debug.LogWarning("🟥 GameObject NOT found: " + objectName);
                foreach (GameObject go in GameObject.FindObjectsOfType<GameObject>())
                {
                    Debug.Log("Scene object: " + go.name);
                }
                return;
            }

            if (action == "move")
            {
                JArray dir = (JArray)parsed["direction"];
                float dx = dir[0].ToObject<float>();
                float dy = dir[1].ToObject<float>();
                float dz = dir[2].ToObject<float>();

                Vector3 moveVector = new Vector3(dx, dy, dz);
                obj.transform.position += moveVector;
                Debug.Log($"🚚 Moved {objectName} by {moveVector}");
            }
            else if (action == "delete")
            {
                Destroy(obj);
                Debug.Log($"🗑️ Deleted object: {objectName}");
            }
            else
            {
                Debug.LogWarning("⚠️ Unknown action: " + action);
            }
        }
        catch (Exception e)
        {
            Debug.LogError("❌ Failed to parse or execute JSON: " + e.Message);
        }

        stream.Close();
        client.Close();
    }

    void OnApplicationQuit()
    {
        server?.Stop();
    }
}
