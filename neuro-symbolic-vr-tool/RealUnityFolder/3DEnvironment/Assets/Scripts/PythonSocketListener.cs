using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class PythonSocketListener : MonoBehaviour
{
    private TcpListener server;
    public string prefabPath = "Prefabs/";

    void Start()
    {
        server = new TcpListener(IPAddress.Any, 5050);
        server.Start();
        Debug.Log("✅ Unity TCP server started on port 5050.");
        InvokeRepeating(nameof(Listen), 1f, 0.2f);
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

            if (data.ContainsKey("screenshot") && data["screenshot"]?.ToObject<bool>() == true)
            {
                var screenshotObj = GameObject.Find("Main Camera");
                if (screenshotObj != null && screenshotObj.GetComponent<ScreenshotVR>() != null)
                {
                    screenshotObj.GetComponent<ScreenshotVR>().TakeScreenshot();
                }
                return;
            }

            string action = data["action"]?.ToString();
            string objName = data["object"]?.ToString();

            if (action == "move")
            {
                string direction = data["direction"]?.ToString();
                GameObject obj = GameObject.Find(objName);
                if (obj != null)
                {
                    Vector3 move = direction switch
                    {
                        "left" => Vector3.left,
                        "right" => Vector3.right,
                        "up" => Vector3.up,
                        "down" => Vector3.down,
                        "forward" => Vector3.forward,
                        "backward" => Vector3.back,
                        _ => Vector3.zero
                    };
                    obj.transform.Translate(move * 1f);
                    Debug.Log($"➡️ Moved {objName} to the {direction}");
                }
                else Debug.LogWarning($"❗ GameObject '{objName}' not found.");
            }
            else if (action == "delete")
            {
                GameObject obj = GameObject.Find(objName);
                if (obj != null)
                {
                    Destroy(obj);
                    Debug.Log($"🗑️ Deleted GameObject: {objName}");
                }
                else Debug.LogWarning($"❗ GameObject '{objName}' not found for deletion.");
            }
            else if (action == "add")
            {
                JObject props = (JObject)data["properties"];
                string color = props["color"]?.ToString();
                string shape = props["shape"]?.ToString();
                string size = props["size"]?.ToString();
                string prefabName = $"{shape}_{color}_{size}";

                GameObject prefab = Resources.Load<GameObject>($"{prefabPath}{prefabName}");
                if (prefab != null)
                {
                    GameObject newObj = Instantiate(prefab, Vector3.zero, Quaternion.identity);
                    newObj.name = objName;
                    Debug.Log($"✨ Created object: {objName}");
                }
                else
                {
                    Debug.LogWarning($"❗ Prefab not found: {prefabPath}{prefabName}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ JSON Parse error: {e.Message}");
        }
    }

    void OnApplicationQuit()
    {
        server.Stop();
    }
}
