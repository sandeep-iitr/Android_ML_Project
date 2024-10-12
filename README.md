# SampleMLApp
The **SampleMLApp** demonstrates the power of on-device Machine Learning (ML) by running a **TensorFlow Lite (TFLite)** model directly on an Android device. With on-device ML, all computations happen locally on the phone, ensuring **faster predictions**, **enhanced privacy**, and the ability to work **offline** without relying on internet access. This app makes it easy to explore how ML models can be integrated into mobile applications for tasks like **image classification**, showcasing how accessible and exciting AI has become for mobile developers. 

<img src="https://github.com/sandeep-iitr/Android_ML_Project/blob/main/MainScreen.png" alt="Main Screen" width="400"/>


## Understanding the Code/Concepts

This section explains the **MainActivity** in the `SampleMLApp`. The app provides an interactive way to classify images using a TensorFlow Lite (TFLite) model. Below is a step-by-step breakdown of the key components and functionality.

---

## Overview

The **SampleMLApp** allows users to:
- Select an image from storage or take a photo using the camera.
- Classify the image using a TFLite model running on the **CPU**.
- Display the predicted label in the UI.

---

## Features

- **Select Image**: Pick an image from storage.
- **Capture Image**: Take a photo using the camera.
- **AI-based Classification**: Classify the image using the MobileNet V3 TFLite model.
- **CPU-based Inference**: Runs inference on the CPU to avoid GPU dependencies.
- **Modern UI with Jetpack Compose**: Uses Jetpack Compose for building the user interface.

---

## How It Works

1. The app loads the **TFLite model** and its **labels** from the assets folder.
2. Users can either **select** an image or **take a photo**.
3. The selected image is **preprocessed** to the required size and format.
4. The TFLite model classifies the image, and the **prediction result** is displayed in the UI.

---

## Key Components

1. **TFLite Model Loading**  
   Loads the model and labels from the assets folder using the TensorFlow Lite interpreter.

2. **Image Capture and Selection**  
   Users can select images from storage or take photos using the camera.

3. **Preprocessing**  
   Resizes the image to 224x224 pixels and normalizes the pixel values for the model.

4. **Background Inference**  
   Runs the model inference in a separate thread to keep the UI responsive.

5. **Jetpack Compose UI**  
   Displays the selected image and the classification result dynamically using Jetpack Compose.

---


# How to Compile and Run `SampleMLApp`

This guide explains how to compile and run the **SampleMLApp** project for image classification using TensorFlow Lite on Android devices. Follow these steps to set up, build, and run the project.

---

## Prerequisites

1. **Android Studio**: Download and install the latest version of [Android Studio](https://developer.android.com/studio).
2. **Java Development Kit (JDK)**: Ensure you have **JDK 11 or later** installed.
3. **Android Device or Emulator**:  
   - A physical Android device (with **USB debugging enabled**).  
   - OR set up an **Android emulator** via Android Studio.
4. **Git**: Make sure Git is installed. You can install it from [here](https://git-scm.com/downloads).

---

##  Step 1: Clone the Repository

Use SSH to clone the repository to your local machine:

```bash
git clone git@github.com:sandeep-iitr/Android_ML_Project.git
```

## Step 2: Open the Project in Android Studio

1. **Launch Android Studio** on your computer.
2. On the **Welcome screen**, select **Open an Existing Project**.
3. Navigate to the **cloned project directory**:
4. Select the project folder and click **OK**.
5. **Wait for Gradle Sync** to complete. Android Studio will automatically download any required dependencies.
6. If prompted, click **Sync Project with Gradle Files** (available from the **File menu**).
7. Ensure that the **Project view** is visible by selecting **View > Tool Windows > Project** or pressing:
8. Once the project is open and synced, you are ready to **build and run the app**.



