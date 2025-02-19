using UnityEngine; //core unity functions
using UnityEngine.UI; // handels ui like text and buttons
using UnityEngine.SceneManagement;// allows changes in unity


public class StartScreenUI : MonoBehaviour // Reference to a UI text element that displays instructions
{
    public Text descriptionText;

    void Start()
    {
        descriptionText.text = "Say 'Start' to begin"; // Set the instruction text when the scene starts
    }

    // This function will be called when voice command  is detected and recognized 
    public void OnVoiceCommandRecognized(string command)
    {
        if (command.ToLower() == "start")
        {
            SceneManager.LoadScene("MainScene");  // Load the main scene when "start" is detected
        }
    }
}