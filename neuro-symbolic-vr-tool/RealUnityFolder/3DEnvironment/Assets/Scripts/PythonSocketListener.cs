using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class PythonSocketListener : MonoBehaviour
{
    private TcpListener server;

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

            // Screenshot signal
            if (data.ContainsKey("screenshot") && data["screenshot"]?.ToObject<bool>() == true)
            {
                Debug.Log("📸 Screenshot request received.");
                var screenshotScript = GameObject.FindObjectOfType<ScreenshotVR>();
                if (screenshotScript != null)
                {
                    screenshotScript.TakeScreenshot();
                }
                else
                {
                    Debug.LogWarning("❗ ScreenshotVR component not found.");
                }
                return;
            }

            string objName = data["object"]?.ToString();
            string direction = data["direction"]?.ToString();

            Debug.Log($"🔎 Looking for object: {objName}");
            GameObject obj = GameObject.Find(objName);

            if (obj != null)
            {
                Debug.Log($"✅ Found GameObject: {obj.name}");

                Vector3 move = direction switch
                {
                    "left" => Vector3.left,
                    "right" => Vector3.right,
                    "up" => Vector3.up,
                    "down" => Vector3.down,
                    _ => Vector3.zero
                };

                obj.transform.Translate(move * 1f);
            }
            else
            {
                Debug.LogWarning($"❗ GameObject '{objName}' not found in scene.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ JSON Parse or logic error: {e.Message}");
        }
    }

    void OnApplicationQuit()
    {
        server.Stop();
        Debug.Log("🛑 Unity TCP server stopped.");
    }
}
