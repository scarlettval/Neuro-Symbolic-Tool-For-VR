using UnityEngine;
using UnityEngine.InputSystem;
using System.IO;

public class ScreenshotVR : MonoBehaviour
{
     public Camera vrCamera;
    private string screenshotPath;

    void Start()
    {
        string scriptFolder = Application.dataPath + "/python/";
        if (!Directory.Exists(scriptFolder))
        {
            Directory.CreateDirectory(scriptFolder);
        }

        screenshotPath = scriptFolder + "vr_screenshot.png";
    }

    public void TakeScreenshot()
    {
        if (vrCamera != null)
        {
            RenderTexture renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
            vrCamera.targetTexture = renderTexture;
            Texture2D screenshot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
            vrCamera.Render();
            RenderTexture.active = renderTexture;
            screenshot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
            screenshot.Apply();
            byte[] bytes = screenshot.EncodeToPNG();
            System.IO.File.WriteAllBytes(screenshotPath, bytes);
            vrCamera.targetTexture = null;
            RenderTexture.active = null;
            Destroy(renderTexture);
            Debug.Log("Screenshot saved to: " + screenshotPath);
        }
        else
        {
            Debug.LogError("VR Camera is not assigned!");
        }
    }

    void Update()
    {
        if (Keyboard.current.enterKey.wasPressedThisFrame)
        {
            TakeScreenshot();
        }
    }
}