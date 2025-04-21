using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

public class ScreenshotVR : MonoBehaviour
{
    public Camera vrCamera;
    private string screenshotPath;

    void Start()
    {
        string folder = Application.dataPath + "/../output/";
        if (!Directory.Exists(folder))
        {
            Directory.CreateDirectory(folder);
        }

        screenshotPath = folder + "vr_snapshot.png";
        Debug.Log("📸 Screenshot path: " + screenshotPath);
    }

    public void TakeScreenshot()
    {
        if (vrCamera == null)
        {
            Debug.LogError("❌ VR Camera not assigned!");
            return;
        }

        RenderTexture rt = new RenderTexture(Screen.width, Screen.height, 24);
        vrCamera.targetTexture = rt;
        Texture2D screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
        vrCamera.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenShot.Apply();

        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(screenshotPath, bytes);
        Debug.Log("✅ Screenshot saved to: " + screenshotPath);

        vrCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
    }

    void Update()
    {
        if (Keyboard.current.enterKey.wasPressedThisFrame)
        {
            TakeScreenshot();
        }
    }
}
