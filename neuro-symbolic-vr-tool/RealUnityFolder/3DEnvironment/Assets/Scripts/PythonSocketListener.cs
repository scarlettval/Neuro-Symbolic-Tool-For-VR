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
    private bool shouldTakeScreenshot = false;

    void Start()
    {
        server = new TcpListener(IPAddress.Any, 5050);
        server.Start();
        screenshotPath = Path.Combine(Application.dataPath, "Snapshots/vr_snapshot.png");
        Debug.Log("✅ Unity TCP server started on port 5050.");
        InvokeRepeating(nameof(Listen), 1f, 0.5f);
    }

    void Update()
    {
        if (shouldTakeScreenshot)
        {
            Debug.Log("📸 Taking deferred screenshot...");
            Directory.CreateDirectory(Path.GetDirectoryName(screenshotPath));
            ScreenCapture.CaptureScreenshot(screenshotPath);
            shouldTakeScreenshot = false;
            Debug.Log($"✅ Screenshot saved to: {screenshotPath}");
        }
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

            // === Screenshot Handling ===
            if (parsed["screenshot"] != null && parsed["screenshot"].ToObject<bool>())
            {
                Debug.Log("📸 Screenshot command received. Deferring to next frame.");
                shouldTakeScreenshot = true;
                return;
            }

            // === Common Fields ===
            string action = parsed["action"]?.ToString();
            string objectName = parsed["object"]?.ToString();

            if (action == "move")
            {
                GameObject obj = GameObject.Find(objectName);
                if (obj == null)
                {
                    Debug.LogWarning("🟥 GameObject NOT found: " + objectName);
                    LogAllSceneObjects();
                    return;
                }

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
                GameObject obj = GameObject.Find(objectName);
                if (obj == null)
                {
                    Debug.LogWarning("🟥 GameObject NOT found for delete: " + objectName);
                    LogAllSceneObjects();
                    return;
                }

                Destroy(obj);
                Debug.Log($"🗑️ Deleted object: {objectName}");
            }
            else if (action == "add")
            {
                JObject props = (JObject)parsed["properties"];
                string color = props["color"]?.ToString();
                string shape = props["shape"]?.ToString();
                string material = props["material"]?.ToString();
                string size = props["size"]?.ToString();

                GameObject newObj = CreateObjectFromShape(shape);
                if (newObj == null)
                {
                    Debug.LogError($"❌ Unsupported shape: {shape}");
                    return;
                }

                newObj.name = objectName;
                newObj.transform.position = Vector3.zero;

                // Apply color
                Renderer renderer = newObj.GetComponent<Renderer>();
                if (renderer != null)
                    renderer.material.color = ColorFromName(color);

                Debug.Log($"✨ Added object: {objectName} | {color}, {shape}, {material}, {size}");
            }
            else
            {
                Debug.LogWarning("⚠️ Unknown action: " + action);
            }
        }
        catch (Exception e)
        {
            Debug.LogError("❌ JSON handling error: " + e.Message);
        }

        stream.Close();
        client.Close();
    }

    GameObject CreateObjectFromShape(string shape)
    {
        switch (shape.ToLower())
        {
            case "cube": return GameObject.CreatePrimitive(PrimitiveType.Cube);
            case "sphere": return GameObject.CreatePrimitive(PrimitiveType.Sphere);
            case "cylinder": return GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            case "capsule": return GameObject.CreatePrimitive(PrimitiveType.Capsule);
            default: return null;
        }
    }

    Color ColorFromName(string name)
    {
        switch (name.ToLower())
        {
            case "red": return Color.red;
            case "blue": return Color.blue;
            case "green": return Color.green;
            case "yellow": return Color.yellow;
            case "gray": return Color.gray;
            case "white": return Color.white;
            case "black": return Color.black;
            default: return Color.magenta;
        }
    }

    void LogAllSceneObjects()
    {
        foreach (GameObject go in GameObject.FindObjectsOfType<GameObject>())
        {
            Debug.Log("📦 Scene object: " + go.name);
        }
    }

    void OnApplicationQuit()
    {
        server?.Stop();
    }
}
