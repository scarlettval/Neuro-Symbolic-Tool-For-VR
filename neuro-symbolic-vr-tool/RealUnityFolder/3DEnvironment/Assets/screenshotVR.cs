using UnityEngine;
using System.IO;
using UnityEngine.InputSystem;

public class ScreenshotVR : MonoBehaviour
{
    public Camera vrCamera;

    void Update()
    {
        if (Keyboard.current.enterKey.wasPressedThisFrame)
        {
            TakeScreenshot();
        }
    }

    public void TakeScreenshot()
    {
        if (vrCamera == null)
        {
            Debug.LogError("❌ ScreenshotVR: No VR Camera assigned.");
            return;
        }

        string folderPath = Path.Combine(Application.dataPath, "Snapshots");
        string filePath = Path.Combine(folderPath, "vr_snapshot.png");

        Directory.CreateDirectory(folderPath);

        int width = Screen.width;
        int height = Screen.height;

        RenderTexture rt = new RenderTexture(width, height, 24);
        vrCamera.targetTexture = rt;
        Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);

        vrCamera.Render();
        RenderTexture.active = rt;
        screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenshot.Apply();

        vrCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        byte[] bytes = screenshot.EncodeToPNG();
        File.WriteAllBytes(filePath, bytes);

        Debug.Log("✅ Screenshot saved to: " + filePath);
    }
}
