using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.XR;

public class VRDataLogger : MonoBehaviour
{
    private string filePath;

    void Start()
    {
        filePath = Application.persistentDataPath + "/vr_training_data.csv";

        if (!File.Exists(filePath))
        {
            File.WriteAllText(filePath, "HandTracking,Controller,ObjectNearby,Posture,HandX,HandY,HandZ,Action\n");
        }
    }

    void Update()
    {
        bool handTracking = XRSettings.loadedDeviceName == "Oculus" ? OVRInput.IsControllerConnected(OVRInput.Controller.Hands) : false;
        bool controllerConnected = OVRInput.IsControllerConnected(OVRInput.Controller.RTouch);
        bool objectNearby = Physics.Raycast(transform.position, transform.forward, 1.5f);
        Vector3 handPos = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);
        string action = DetectUserAction();

        string logEntry = $"{(handTracking ? 1 : 0)},{(controllerConnected ? 1 : 0)},{(objectNearby ? 1 : 0)},{GetUserPosture()},{handPos.x},{handPos.y},{handPos.z},{action}\n";
        File.AppendAllText(filePath, logEntry);
    }

    string GetUserPosture()
    {
        if (OVRInput.Get(OVRInput.Button.One))
            return "crouching";
        if (OVRInput.Get(OVRInput.Button.Two))
            return "jumping";
        return "standing";
    }

    string DetectUserAction()
    {
        if (OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger)) return "grab_object";
        if (OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick).y > 0.5f) return "move_forward";
        if (OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick).y < -0.5f) return "move_backward";
        if (OVRInput.Get(OVRInput.Button.Four)) return "wave_hand";
        return "idle";
    }
}
