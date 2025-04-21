using UnityEngine;
using System.IO;
using UnityEngine.InputSystem;

public class ScreenshotVR : MonoBehaviour
{
    public Camera vrCamera;
    private string screenshotPath;

    void Start()
    {
        screenshotPath = Path.Combine(Application.dataPath, "python", "vr_snapshot.png");
        Directory.CreateDirectory(Path.GetDirectoryName(screenshotPath));
    }

    public void TakeScreenshot()
    {
        if (vrCamera == null)
        {
            Debug.LogError("❌ VR Camera not assigned.");
            return;
        }

        RenderTexture rt = new RenderTexture(Screen.width, Screen.height, 24);
        vrCamera.targetTexture = rt;
        Texture2D screenshot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
        vrCamera.Render();
        RenderTexture.active = rt;
        screenshot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenshot.Apply();
        vrCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        File.WriteAllBytes(screenshotPath, screenshot.EncodeToPNG());
        Debug.Log("✅ Screenshot saved at: " + screenshotPath);
    }

    void Update()
    {
        if (Keyboard.current.enterKey.wasPressedThisFrame)
        {
            TakeScreenshot();
        }
    }
}
