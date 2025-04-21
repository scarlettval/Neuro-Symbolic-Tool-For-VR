using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class PythonSocketListener : MonoBehaviour
{
    private TcpListener server;
    private bool isListening = false;

    void Start()
    {
        try
        {
            server = new TcpListener(IPAddress.Any, 5050);
            server.Start();
            isListening = true;
            Debug.Log("[Socket] Unity TCP server started on port 5050.");
        }
        catch (Exception e)
        {
            Debug.LogError("[Socket] Failed to start TCP server: " + e.Message);
        }
    }

    void Update()
    {
        if (!isListening || server == null || !server.Pending()) return;

        try
        {
            TcpClient client = server.AcceptTcpClient();
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[client.ReceiveBufferSize];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            string msg = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            Debug.Log("[Socket] Received from Python: " + msg);
            HandleMessage(msg);
            client.Close();
        }
        catch (Exception e)
        {
            Debug.LogError("[Socket] Error handling client: " + e.Message);
        }
    }

    void HandleMessage(string json)
    {
        try
        {
            JObject data = JObject.Parse(json);
            string objName = data["object"]?.ToString();
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
                    _ => Vector3.zero
                };
                obj.transform.Translate(move);
                Debug.Log($"[Socket] Moved {objName} to the {direction}");
            }
            else
            {
                Debug.LogWarning($"[Socket] GameObject '{objName}' not found in scene.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError("[Socket] JSON parse error: " + e.Message);
        }
    }

    void OnApplicationQuit()
    {
        if (server != null)
        {
            server.Stop();
            isListening = false;
        }
    }
}
