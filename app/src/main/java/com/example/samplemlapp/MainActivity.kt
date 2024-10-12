package com.example.samplemlapp

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.contract.ActivityResultContracts.RequestPermission
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.samplemlapp.ui.theme.SampleMLAppTheme
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.io.BufferedReader
import java.io.InputStreamReader
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.ui.draw.alpha

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var labels: List<String>

    // Mutable state to hold the most recent image and classification result
    private var mostRecentImage: MutableState<Bitmap?> = mutableStateOf(null)
    private var classificationResult: MutableState<String> = mutableStateOf("")

    // File picker to select images from storage
    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            bitmap?.let {
                mostRecentImage.value = it  // Update most recent image
                classifyImageInBackground(it)
            }
        } ?: run {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
        }
    }

    // Camera launcher to capture image
    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        bitmap?.let {
            mostRecentImage.value = it  // Update most recent image
            classifyImageInBackground(it)
        } ?: run {
            Toast.makeText(this, "No photo captured", Toast.LENGTH_SHORT).show()
        }
    }

    // Permission launcher to request camera permission
    private val requestCameraPermissionLauncher = registerForActivityResult(RequestPermission()) { isGranted ->
        if (isGranted) {
            takePictureLauncher.launch(null)  // Launch the camera after permission is granted
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            Log.d("MainActivity", "Loading TensorFlow Lite model and labels...")
            interpreter = Interpreter(loadModelFile(), getInterpreterOptions())
            interpreter.allocateTensors()  // Ensures that memory for tensors is allocated
            labels = loadLabels()  // Load labels from assets
            Log.d("MainActivity", "Model and labels loaded successfully.")
        } catch (e: Exception) {
            Log.e("MainActivity", "Error loading TensorFlow Lite model: ${e.message}")
        }

        setContent {
            SampleMLAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ImageClassifierUI(
                        onSelectImage = { pickImageLauncher.launch("image/*") },
                        onTakePhoto = {
                            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        },
                        image = mostRecentImage.value,
                        result = classificationResult.value
                    )
                }
            }
        }
    }

    // Create interpreter options without the GPU delegate (runs on CPU)
    private fun getInterpreterOptions(): Interpreter.Options {
        return Interpreter.Options()
    }

    // Load TFLite model from assets
    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("mobilenet_v3.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val modelByteArray = inputStream.readBytes()
        val buffer = ByteBuffer.allocateDirect(modelByteArray.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(modelByteArray)
        return buffer
    }

    // Load labels from the assets folder
    private fun loadLabels(): List<String> {
        val labels = mutableListOf<String>()
        val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
        reader.forEachLine {
            labels.add(it.trim())  // Each line is a separate label
        }
        reader.close()
        return labels
    }

    // Preprocess the image into a ByteBuffer expected by the model
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        Log.d("MainActivity", "Preprocessing image...")
        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3) // 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder()) // Use native order

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = scaledBitmap.getPixel(x, y)

                // Convert pixel to float and normalize
                val r = ((pixel shr 16 and 0xFF) / 255.0f)
                val g = ((pixel shr 8 and 0xFF) / 255.0f)
                val b = ((pixel and 0xFF) / 255.0f)

                // Put the normalized values into the ByteBuffer
                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }
        }

        Log.d("MainActivity", "Image preprocessing completed.")
        return inputBuffer
    }

    // Run classification in the background
    private fun classifyImageInBackground(bitmap: Bitmap) {
        executor.execute {
            try {
                Log.d("MainActivity", "Starting image classification...")
                val input = preprocessImage(bitmap)
                val result = runModel(input)
                runOnUiThread {
                    classificationResult.value = result  // Update classification result
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error during classification: ${e.message}")
                runOnUiThread {
                    classificationResult.value = "Failed to classify image"  // Update classification result
                }
            }
        }
    }

    // Run the model on the preprocessed input
    private fun runModel(inputBuffer: ByteBuffer): String {
        try {
            Log.d("MainActivity", "Running model inference...")

            // Output buffer to store the results (MobileNet usually has 1001 classes)
            val output = Array(1) { FloatArray(1001) }

            // Run inference
            interpreter.run(inputBuffer, output)
            Log.d("MainActivity", "Model inference completed.")

            // Find the index of the maximum value in the output
            val classId = output[0].withIndex().maxByOrNull { it.value }?.index ?: -1
            Log.d("MainActivity", "Predicted Class ID: $classId")

            // Map class ID to human-readable label
            val label = if (classId in labels.indices) labels[classId] else "Unknown"
            Log.d("MainActivity", "Predicted Label: $label")

            return "AI Results: $label"
        } catch (e: Exception) {
            Log.e("MainActivity", "Error during model inference: ${e.message}")
            return "Error"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
    }
}

@Composable
fun ImageClassifierUI(
    onSelectImage: () -> Unit,
    onTakePhoto: () -> Unit,
    image: Bitmap?,
    result: String
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Welcome message with animation
        var alphaState by remember { mutableStateOf(0f) }
        val animatedAlpha = animateFloatAsState(targetValue = alphaState)

        LaunchedEffect(Unit) {
            alphaState = 1f  // Trigger the fade-in effect
        }

        Text(
            text = "Welcome to Mobile-AI",
            fontWeight = FontWeight.Bold,
            fontSize = 24.sp,
            color = Color.Blue,
            modifier = Modifier
                .padding(16.dp)
                .alpha(animatedAlpha.value)
        )

        Button(onClick = onSelectImage, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Select Image")
        }
        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = onTakePhoto, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Take Photo")
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Display the most recent image
        image?.let {
            Image(bitmap = it.asImageBitmap(), contentDescription = null, modifier = Modifier.size(224.dp))
            Spacer(modifier = Modifier.height(16.dp))
        }

        // Display the classification result
        Text(text = result)
    }
}

@Preview(showBackground = true)
@Composable
fun DefaultPreview() {
    SampleMLAppTheme {
        ImageClassifierUI(
            onSelectImage = {},
            onTakePhoto = {},
            image = null,
            result = "No Classification Yet"
        )
    }
}


//package com.example.samplemlapp
//
//import android.Manifest
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import android.os.Bundle
//import android.util.Log
//import android.widget.Toast
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.activity.result.contract.ActivityResultContracts
//import androidx.activity.result.contract.ActivityResultContracts.RequestPermission
//import androidx.compose.foundation.Image
//import androidx.compose.foundation.layout.*
//import androidx.compose.material3.*f
//import androidx.compose.runtime.*
//import androidx.compose.ui.Alignment
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.asImageBitmap
//import androidx.compose.ui.tooling.preview.Preview
//import androidx.compose.ui.unit.dp
//import com.example.samplemlapp.ui.theme.SampleMLAppTheme
//import org.tensorflow.lite.Interpreter
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//import java.util.concurrent.Executors
//import java.io.BufferedReader
//import java.io.InputStreamReader
//
//class MainActivity : ComponentActivity() {
//
//    private lateinit var interpreter: Interpreter
//    private val executor = Executors.newSingleThreadExecutor()
//    private lateinit var labels: List<String>
//
//    // Mutable state to hold the most recent image and classification result
//    private var mostRecentImage: MutableState<Bitmap?> = mutableStateOf(null)
//    private var classificationResult: MutableState<String> = mutableStateOf("")
//
//    // File picker to select images from storage
//    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
//        uri?.let {
//            val inputStream = contentResolver.openInputStream(uri)
//            val bitmap = BitmapFactory.decodeStream(inputStream)
//            bitmap?.let {
//                mostRecentImage.value = it  // Update most recent image
//                classifyImageInBackground(it)
//            }
//        } ?: run {
//            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    // Camera launcher to capture image
//    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
//        bitmap?.let {
//            mostRecentImage.value = it  // Update most recent image
//            classifyImageInBackground(it)
//        } ?: run {
//            Toast.makeText(this, "No photo captured", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    // Permission launcher to request camera permission
//    private val requestCameraPermissionLauncher = registerForActivityResult(RequestPermission()) { isGranted ->
//        if (isGranted) {
//            takePictureLauncher.launch(null)  // Launch the camera after permission is granted
//        } else {
//            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        try {
//            Log.d("MainActivity", "Loading TensorFlow Lite model and labels...")
//            interpreter = Interpreter(loadModelFile(), getInterpreterOptions())
//            interpreter.allocateTensors()  // Ensures that memory for tensors is allocated
//            labels = loadLabels()  // Load labels from assets
//            Log.d("MainActivity", "Model and labels loaded successfully.")
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error loading TensorFlow Lite model: ${e.message}")
//        }
//
//        setContent {
//            SampleMLAppTheme {
//                Surface(
//                    modifier = Modifier.fillMaxSize(),
//                    color = MaterialTheme.colorScheme.background
//                ) {
//                    ImageClassifierUI(
//                        onSelectImage = { pickImageLauncher.launch("image/*") },
//                        onTakePhoto = {
//                            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
//                        },
//                        image = mostRecentImage.value,
//                        result = classificationResult.value
//                    )
//                }
//            }
//        }
//    }
//
//    // Create interpreter options without the GPU delegate (runs on CPU)
//    private fun getInterpreterOptions(): Interpreter.Options {
//        return Interpreter.Options()
//    }
//
//    // Load TFLite model from assets
//    private fun loadModelFile(): ByteBuffer {
//        val fileDescriptor = assets.openFd("mobilenet_v3.tflite")
//        val inputStream = fileDescriptor.createInputStream()
//        val modelByteArray = inputStream.readBytes()
//        val buffer = ByteBuffer.allocateDirect(modelByteArray.size)
//        buffer.order(ByteOrder.nativeOrder())
//        buffer.put(modelByteArray)
//        return buffer
//    }
//
//    // Load labels from the assets folder
//    private fun loadLabels(): List<String> {
//        val labels = mutableListOf<String>()
//        val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
//        reader.forEachLine {
//            labels.add(it.trim())  // Each line is a separate label
//        }
//        reader.close()
//        return labels
//    }
//
//
//    // Preprocess the image into a ByteBuffer expected by the model
//    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
//        Log.d("MainActivity", "Preprocessing image...")
//        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3) // 4 bytes per float
//        inputBuffer.order(ByteOrder.nativeOrder()) // Use native order
//
//        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
//
//        for (y in 0 until 224) {
//            for (x in 0 until 224) {
//                val pixel = scaledBitmap.getPixel(x, y)
//
//                // Convert pixel to float and normalize
//                val r = ((pixel shr 16 and 0xFF) / 255.0f)
//                val g = ((pixel shr 8 and 0xFF) / 255.0f)
//                val b = ((pixel and 0xFF) / 255.0f)
//
//                // Put the normalized values into the ByteBuffer
//                inputBuffer.putFloat(r)
//                inputBuffer.putFloat(g)
//                inputBuffer.putFloat(b)
//            }
//        }
//
//        Log.d("MainActivity", "Image preprocessing completed.")
//        return inputBuffer
//    }
//
//    // Run classification in the background
//    private fun classifyImageInBackground(bitmap: Bitmap) {
//        executor.execute {
//            try {
//                Log.d("MainActivity", "Starting image classification...")
//                val input = preprocessImage(bitmap)
//                val result = runModel(input)
//                runOnUiThread {
//                    classificationResult.value = result  // Update classification result
//                }
//            } catch (e: Exception) {
//                Log.e("MainActivity", "Error during classification: ${e.message}")
//                runOnUiThread {
//                    classificationResult.value = "Failed to classify image"  // Update classification result
//                }
//            }
//        }
//    }
//
//    // Run the model on the preprocessed input
//    private fun runModel(inputBuffer: ByteBuffer): String {
//        try {
//            Log.d("MainActivity", "Running model inference...")
//
//            // Output buffer to store the results (MobileNet usually has 1001 classes)
//            val output = Array(1) { FloatArray(1001) }
//
//            // Run inference
//            interpreter.run(inputBuffer, output)
//            Log.d("MainActivity", "Model inference completed.")
//
//            // Find the index of the maximum value in the output
//            val classId = output[0].withIndex().maxByOrNull { it.value }?.index ?: -1
//            Log.d("MainActivity", "Predicted Class ID: $classId")
//
//            // Map class ID to human-readable label
//            val label = if (classId in labels.indices) labels[classId] else "Unknown"
//            Log.d("MainActivity", "Predicted Label: $label")
//
//            return "Class: $label (ID: $classId)"
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error during model inference: ${e.message}")
//            return "Error"
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        interpreter.close()
//    }
//}
//
//@Composable
//fun ImageClassifierUI(
//    onSelectImage: () -> Unit,
//    onTakePhoto: () -> Unit,
//    image: Bitmap?,
//    result: String
//) {
//    Column(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(16.dp),
//        verticalArrangement = Arrangement.Center,
//        horizontalAlignment = Alignment.CenterHorizontally
//    ) {
//        Button(onClick = onSelectImage, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Select Image")
//        }
//        Spacer(modifier = Modifier.height(16.dp))
//        Button(onClick = onTakePhoto, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Take Photo")
//        }
//
//        Spacer(modifier = Modifier.height(32.dp))
//
//        // Display the most recent image
//        image?.let {
//            Image(bitmap = it.asImageBitmap(), contentDescription = null, modifier = Modifier.size(224.dp))
//            Spacer(modifier = Modifier.height(16.dp))
//        }
//
//        // Display the classification result
//        Text(text = result)
//    }
//}
//
//@Preview(showBackground = true)
//@Composable
//fun DefaultPreview() {
//    SampleMLAppTheme {
//        ImageClassifierUI(
//            onSelectImage = {},
//            onTakePhoto = {},
//            image = null,
//            result = "No Classification Yet"
//        )
//    }
//}


//package com.example.samplemlapp
//import android.Manifest
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import android.os.Bundle
//import android.util.Log
//import android.widget.Toast
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.activity.result.contract.ActivityResultContracts
//import androidx.activity.result.contract.ActivityResultContracts.RequestPermission
//import androidx.compose.foundation.layout.*
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.tooling.preview.Preview
//import androidx.compose.ui.unit.dp
//import com.example.samplemlapp.ui.theme.SampleMLAppTheme
//import org.tensorflow.lite.Interpreter
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//import java.util.concurrent.Executors
//import java.io.BufferedReader
//import java.io.InputStreamReader
//
//class MainActivity : ComponentActivity() {
//
//    private lateinit var interpreter: Interpreter
//    private val executor = Executors.newSingleThreadExecutor()
//    private lateinit var labels: List<String>
//
//    // File picker to select images from storage
//    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
//        uri?.let {
//            val inputStream = contentResolver.openInputStream(uri)
//            val bitmap = BitmapFactory.decodeStream(inputStream)
//            bitmap?.let {
//                classifyImageInBackground(it)
//            }
//        } ?: run {
//            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    // Camera launcher to capture image
//    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
//        bitmap?.let {
//            classifyImageInBackground(it)
//        } ?: run {
//            Toast.makeText(this, "No photo captured", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    // Permission launcher to request camera permission
//    private val requestCameraPermissionLauncher = registerForActivityResult(RequestPermission()) { isGranted ->
//        if (isGranted) {
//            takePictureLauncher.launch(null)  // Launch the camera after permission is granted
//        } else {
//            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        try {
//            Log.d("MainActivity", "Loading TensorFlow Lite model and labels...")
//            interpreter = Interpreter(loadModelFile(), getInterpreterOptions())
//            interpreter.allocateTensors()  // Ensures that memory for tensors is allocated
//            labels = loadLabels()  // Load labels from assets
//            Log.d("MainActivity", "Model and labels loaded successfully.")
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error loading TensorFlow Lite model: ${e.message}")
//        }
//
//        setContent {
//            SampleMLAppTheme {
//                Surface(
//                    modifier = Modifier.fillMaxSize(),
//                    color = MaterialTheme.colorScheme.background
//                ) {
//                    ImageClassifierUI(
//                        onSelectImage = { pickImageLauncher.launch("image/*") },
//                        onTakePhoto = {
//                            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
//                        }
//                    )
//                }
//            }
//        }
//    }
//
//    // Create interpreter options without the GPU delegate (runs on CPU)
//    private fun getInterpreterOptions(): Interpreter.Options {
//        return Interpreter.Options()
//    }
//
//    // Load TFLite model from assets
//    private fun loadModelFile(): ByteBuffer {
//        val fileDescriptor = assets.openFd("mobilenet_v3.tflite")
//        val inputStream = fileDescriptor.createInputStream()
//        val modelByteArray = inputStream.readBytes()
//        val buffer = ByteBuffer.allocateDirect(modelByteArray.size)
//        buffer.order(ByteOrder.nativeOrder())
//        buffer.put(modelByteArray)
//        return buffer
//    }
//
//    // Load labels from the assets folder
//    private fun loadLabels(): List<String> {
//        val labels = mutableListOf<String>()
//        val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
//        reader.forEachLine {
//            val label = it.split(":")[1].trim()  // Extract the label part after the colon
//            labels.add(label)
//        }
//        reader.close()
//        return labels
//    }
//
//    // Preprocess the image into a ByteBuffer expected by the model
//    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
//        Log.d("MainActivity", "Preprocessing image...")
//        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3) // 4 bytes per float
//        inputBuffer.order(ByteOrder.nativeOrder()) // Use native order
//
//        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
//
//        for (y in 0 until 224) {
//            for (x in 0 until 224) {
//                val pixel = scaledBitmap.getPixel(x, y)
//
//                // Convert pixel to float and normalize
//                val r = ((pixel shr 16 and 0xFF) / 255.0f)
//                val g = ((pixel shr 8 and 0xFF) / 255.0f)
//                val b = ((pixel and 0xFF) / 255.0f)
//
//                // Put the normalized values into the ByteBuffer
//                inputBuffer.putFloat(r)
//                inputBuffer.putFloat(g)
//                inputBuffer.putFloat(b)
//            }
//        }
//
//        Log.d("MainActivity", "Image preprocessing completed.")
//        return inputBuffer
//    }
//
//    // Run classification in the background
//    private fun classifyImageInBackground(bitmap: Bitmap) {
//        executor.execute {
//            try {
//                Log.d("MainActivity", "Starting image classification...")
//                val input = preprocessImage(bitmap)
//                val result = runModel(input)
//                runOnUiThread {
//                    Toast.makeText(this, "Classification Result: $result", Toast.LENGTH_SHORT).show()
//                }
//            } catch (e: Exception) {
//                Log.e("MainActivity", "Error during classification: ${e.message}")
//                runOnUiThread {
//                    Toast.makeText(this, "Failed to classify image", Toast.LENGTH_SHORT).show()
//                }
//            }
//        }
//    }
//
//    // Run the model on the preprocessed input
//    private fun runModel(inputBuffer: ByteBuffer): String {
//        try {
//            Log.d("MainActivity", "Running model inference...")
//
//            // Output buffer to store the results (MobileNet usually has 1001 classes)
//            val output = Array(1) { FloatArray(1001) }
//
//            // Run inference
//            interpreter.run(inputBuffer, output)
//            Log.d("MainActivity", "Model inference completed.")
//
//            // Find the index of the maximum value in the output
//            val classId = output[0].withIndex().maxByOrNull { it.value }?.index ?: -1
//            Log.d("MainActivity", "Predicted Class ID: $classId")
//
//            // Map class ID to human-readable label
//            val label = if (classId in labels.indices) labels[classId] else "Unknown"
//            Log.d("MainActivity", "Predicted Label: $label")
//
//            return "Class: $label (ID: $classId)"
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error during model inference: ${e.message}")
//            return "Error"
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        interpreter.close()
//    }
//}
//
//@Composable
//fun ImageClassifierUI(
//    onSelectImage: () -> Unit,
//    onTakePhoto: () -> Unit
//) {
//    Column(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(16.dp),
//        verticalArrangement = Arrangement.Center
//    ) {
//        Button(onClick = onSelectImage, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Select Image")
//        }
//        Spacer(modifier = Modifier.height(16.dp))
//        Button(onClick = onTakePhoto, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Take Photo")
//        }
//    }
//}
//
//@Preview(showBackground = true)
//@Composable
//fun DefaultPreview() {
//    SampleMLAppTheme {
//        ImageClassifierUI(
//            onSelectImage = {},
//            onTakePhoto = {}
//        )
//    }
//}


//package com.example.samplemlapp
//
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import android.os.Bundle
//import android.util.Log
//import android.widget.Toast
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.activity.result.contract.ActivityResultContracts
//import androidx.compose.foundation.layout.*
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.tooling.preview.Preview
//import androidx.compose.ui.unit.dp
//import com.example.samplemlapp.ui.theme.SampleMLAppTheme
//import org.tensorflow.lite.Interpreter
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//import java.util.concurrent.Executors
//import java.io.BufferedReader
//import java.io.InputStreamReader
//
//class MainActivity : ComponentActivity() {
//
//    private lateinit var interpreter: Interpreter
//    private val executor = Executors.newSingleThreadExecutor()
//    private lateinit var labels: List<String>
//
//    // File picker to select images from storage
//    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
//        uri?.let {
//            val inputStream = contentResolver.openInputStream(uri)
//            val bitmap = BitmapFactory.decodeStream(inputStream)
//            bitmap?.let {
//                classifyImageInBackground(it)
//            }
//        } ?: run {
//            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        try {
//            Log.d("MainActivity", "Loading TensorFlow Lite model and labels...")
//            interpreter = Interpreter(loadModelFile(), getInterpreterOptions())
//            interpreter.allocateTensors()  // Ensures that memory for tensors is allocated
//            labels = loadLabels()  // Load labels from assets
//            Log.d("MainActivity", "Model and labels loaded successfully.")
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error loading TensorFlow Lite model: ${e.message}")
//        }
//
//        setContent {
//            SampleMLAppTheme {
//                Surface(
//                    modifier = Modifier.fillMaxSize(),
//                    color = MaterialTheme.colorScheme.background
//                ) {
//                    ImageClassifierUI(
//                        onSelectImage = { pickImageLauncher.launch("image/*") }
//                    )
//                }
//            }
//        }
//    }
//
//    // Create interpreter options without the GPU delegate (runs on CPU)
//    private fun getInterpreterOptions(): Interpreter.Options {
//        return Interpreter.Options()
//    }
//
//    // Load TFLite model from assets
//    private fun loadModelFile(): ByteBuffer {
//        val fileDescriptor = assets.openFd("mobilenet_v3.tflite")
//        val inputStream = fileDescriptor.createInputStream()
//        val modelByteArray = inputStream.readBytes()
//        val buffer = ByteBuffer.allocateDirect(modelByteArray.size)
//        buffer.order(ByteOrder.nativeOrder())
//        buffer.put(modelByteArray)
//        return buffer
//    }
//
//    // Load labels from the assets folder
//    private fun loadLabels(): List<String> {
//        val labels = mutableListOf<String>()
//        val reader = BufferedReader(InputStreamReader(assets.open("labels.txt")))
//        reader.forEachLine {
//            val label = it.split(":")[1].trim()  // Extract the label part after the colon
//            labels.add(label)
//        }
//        reader.close()
//        return labels
//    }
//
//    // Preprocess the image into a ByteBuffer expected by the model
//    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
//        Log.d("MainActivity", "Preprocessing image...")
//        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3) // 4 bytes per float
//        inputBuffer.order(ByteOrder.nativeOrder()) // Use native order
//
//        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
//
//        for (y in 0 until 224) {
//            for (x in 0 until 224) {
//                val pixel = scaledBitmap.getPixel(x, y)
//
//                // Convert pixel to float and normalize
//                val r = ((pixel shr 16 and 0xFF) / 255.0f)
//                val g = ((pixel shr 8 and 0xFF) / 255.0f)
//                val b = ((pixel and 0xFF) / 255.0f)
//
//                // Put the normalized values into the ByteBuffer
//                inputBuffer.putFloat(r)
//                inputBuffer.putFloat(g)
//                inputBuffer.putFloat(b)
//            }
//        }
//
//        Log.d("MainActivity", "Image preprocessing completed.")
//        return inputBuffer
//    }
//
//    // Run classification in the background
//    private fun classifyImageInBackground(bitmap: Bitmap) {
//        executor.execute {
//            try {
//                Log.d("MainActivity", "Starting image classification...")
//                val input = preprocessImage(bitmap)
//                val result = runModel(input)
//                runOnUiThread {
//                    Toast.makeText(this, "Classification Result: $result", Toast.LENGTH_SHORT).show()
//                }
//            } catch (e: Exception) {
//                Log.e("MainActivity", "Error during classification: ${e.message}")
//                runOnUiThread {
//                    Toast.makeText(this, "Failed to classify image", Toast.LENGTH_SHORT).show()
//                }
//            }
//        }
//    }
//
//    // Run the model on the preprocessed input
//    private fun runModel(inputBuffer: ByteBuffer): String {
//        try {
//            Log.d("MainActivity", "Running model inference...")
//
//            // Output buffer to store the results (MobileNet usually has 1001 classes)
//            val output = Array(1) { FloatArray(1001) }
//
//            // Run inference
//            interpreter.run(inputBuffer, output)
//            Log.d("MainActivity", "Model inference completed.")
//
//            // Find the index of the maximum value in the output
//            val classId = output[0].withIndex().maxByOrNull { it.value }?.index ?: -1
//            Log.d("MainActivity", "Predicted Class ID: $classId")
//
//            // Map class ID to human-readable label
//            val label = if (classId in labels.indices) labels[classId] else "Unknown"
//            Log.d("MainActivity", "Predicted Label: $label")
//
//            return "Class: $label (ID: $classId)"
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error during model inference: ${e.message}")
//            return "Error"
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        interpreter.close()
//    }
//}
//
//@Composable
//fun ImageClassifierUI(
//    onSelectImage: () -> Unit
//) {
//    Column(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(16.dp),
//        verticalArrangement = Arrangement.Center
//    ) {
//        Button(onClick = onSelectImage, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Select Image")
//        }
//    }
//}
//
//@Preview(showBackground = true)
//@Composable
//fun DefaultPreview() {
//    SampleMLAppTheme {
//        ImageClassifierUI(
//            onSelectImage = {}
//        )
//    }
//}




//package com.example.samplemlapp
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import android.os.Bundle
//import android.util.Log
//import android.widget.Toast
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.activity.result.contract.ActivityResultContracts
//import androidx.compose.foundation.layout.*
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.tooling.preview.Preview
//import androidx.compose.ui.unit.dp
//import com.example.samplemlapp.ui.theme.SampleMLAppTheme
//import org.tensorflow.lite.Interpreter
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//import java.util.concurrent.Executors
//
//class MainActivity : ComponentActivity() {
//
//    private lateinit var interpreter: Interpreter
//    private val executor = Executors.newSingleThreadExecutor()
//
//    // File picker to select images from storage
//    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
//        uri?.let {
//            val inputStream = contentResolver.openInputStream(uri)
//            val bitmap = BitmapFactory.decodeStream(inputStream)
//            bitmap?.let {
//                classifyImageInBackground(it)
//            }
//        } ?: run {
//            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
//        }
//    }
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        try {
//            Log.d("MainActivity", "Loading TensorFlow Lite model...")
//            interpreter = Interpreter(loadModelFile(), getInterpreterOptions())
//            interpreter.allocateTensors()  // Ensures that memory for tensors is allocated
//            Log.d("MainActivity", "Model loaded and tensors allocated successfully.")
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error loading TensorFlow Lite model: ${e.message}")
//        }
//
//        setContent {
//            SampleMLAppTheme {
//                Surface(
//                    modifier = Modifier.fillMaxSize(),
//                    color = MaterialTheme.colorScheme.background
//                ) {
//                    ImageClassifierUI(
//                        onSelectImage = { pickImageLauncher.launch("image/*") }
//                    )
//                }
//            }
//        }
//    }
//
//    // Create interpreter options without the GPU delegate (runs on CPU)
//    private fun getInterpreterOptions(): Interpreter.Options {
//        return Interpreter.Options()
//    }
//
//    // Load TFLite model from assets
//    private fun loadModelFile(): ByteBuffer {
//        val fileDescriptor = assets.openFd("mobilenet_v3.tflite")
//        val inputStream = fileDescriptor.createInputStream()
//        val modelByteArray = inputStream.readBytes()
//        val buffer = ByteBuffer.allocateDirect(modelByteArray.size)
//        buffer.order(ByteOrder.nativeOrder())
//        buffer.put(modelByteArray)
//        return buffer
//    }
//
//    // Preprocess the image into a ByteBuffer expected by the model
//    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
//        Log.d("MainActivity", "Preprocessing image...")
//        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3) // 4 bytes per float
//        inputBuffer.order(ByteOrder.nativeOrder()) // Use native order
//
//        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
//
//        for (y in 0 until 224) {
//            for (x in 0 until 224) {
//                val pixel = scaledBitmap.getPixel(x, y)
//
//                // Convert pixel to float and normalize
//                val r = ((pixel shr 16 and 0xFF) / 255.0f)
//                val g = ((pixel shr 8 and 0xFF) / 255.0f)
//                val b = ((pixel and 0xFF) / 255.0f)
//
//                // Put the normalized values into the ByteBuffer
//                inputBuffer.putFloat(r)
//                inputBuffer.putFloat(g)
//                inputBuffer.putFloat(b)
//            }
//        }
//
//        Log.d("MainActivity", "Image preprocessing completed.")
//        return inputBuffer
//    }
//
//    // Run classification in the background
//    private fun classifyImageInBackground(bitmap: Bitmap) {
//        executor.execute {
//            try {
//                Log.d("MainActivity", "Starting image classification...")
//                val input = preprocessImage(bitmap)
//                val result = runModel(input)
//                runOnUiThread {
//                    Toast.makeText(this, "Classification Result: $result", Toast.LENGTH_SHORT).show()
//                }
//            } catch (e: Exception) {
//                Log.e("MainActivity", "Error during classification: ${e.message}")
//                runOnUiThread {
//                    Toast.makeText(this, "Failed to classify image", Toast.LENGTH_SHORT).show()
//                }
//            }
//        }
//    }
//
//    // Run the model on the preprocessed input
//    private fun runModel(inputBuffer: ByteBuffer): String {
//        try {
//            Log.d("MainActivity", "Running model inference...")
//
//            // Output buffer to store the results (MobileNet usually has 1001 classes)
//            val output = Array(1) { FloatArray(1001) }
//
//            // Run inference
//            interpreter.run(inputBuffer, output)
//            Log.d("MainActivity", "Model inference completed.")
//
//            // Find the index of the maximum value in the output
//            val classId = output[0].withIndex().maxByOrNull { it.value }?.index ?: -1
//            Log.d("MainActivity", "Predicted Class ID: $classId")
//
//            return "Class ID: $classId"
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Error during model inference: ${e.message}")
//            return "Error"
//        }
//    }
//
//
//    override fun onDestroy() {
//        super.onDestroy()
//        interpreter.close()
//    }
//}
//
//@Composable
//fun ImageClassifierUI(
//    onSelectImage: () -> Unit
//) {
//    Column(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(16.dp),
//        verticalArrangement = Arrangement.Center
//    ) {
//        Button(onClick = onSelectImage, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Select Image")
//        }
//    }
//}
//
//@Preview(showBackground = true)
//@Composable
//fun DefaultPreview() {
//    SampleMLAppTheme {
//        ImageClassifierUI(
//            onSelectImage = {}
//        )
//    }
//}
//


//package com.example.samplemlapp
//
//import android.Manifest
//import android.media.MediaPlayer
//import android.media.MediaRecorder
//import android.os.Bundle
//import android.widget.Toast
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.activity.result.contract.ActivityResultContracts
//import androidx.compose.foundation.layout.*
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.tooling.preview.Preview
//import androidx.compose.ui.unit.dp
//import com.example.samplemlapp.ui.theme.SampleMLAppTheme
//import java.io.File
//
//class MainActivity : ComponentActivity() {
//    private var mediaRecorder: MediaRecorder? = null
//    private var mediaPlayer: MediaPlayer? = null
//    private var audioFilePath: String = ""
//
//    // Permission launcher to request microphone permission
//    private val requestPermissionLauncher =
//        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
//            if (isGranted) {
//                Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show()
//            } else {
//                Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
//            }
//        }
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        // Path to save the recorded audio
//        audioFilePath = "${externalCacheDir?.absolutePath}/audio_record.3gp"
//
//        // Request microphone permission on app start
//        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
//
//        setContent {
//            SampleMLAppTheme {
//                Surface(
//                    modifier = Modifier.fillMaxSize(),
//                    color = MaterialTheme.colorScheme.background
//                ) {
//                    AudioRecorderUI(
//                        onStartRecording = { startRecording() },
//                        onStopRecording = { stopRecording() },
//                        onPlayRecording = { playRecording() }
//                    )
//                }
//            }
//        }
//    }
//
//    private fun startRecording() {
//        mediaRecorder = MediaRecorder().apply {
//            setAudioSource(MediaRecorder.AudioSource.MIC)
//            setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
//            setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
//            setOutputFile(audioFilePath)
//            prepare()
//            start()
//        }
//        Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show()
//    }
//
//    private fun stopRecording() {
//        mediaRecorder?.apply {
//            stop()
//            release()
//        }
//        mediaRecorder = null
//        Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
//    }
//
//    private fun playRecording() {
//        mediaPlayer = MediaPlayer().apply {
//            setDataSource(audioFilePath)
//            prepare()
//            start()
//        }
//        Toast.makeText(this, "Playing audio", Toast.LENGTH_SHORT).show()
//        mediaPlayer?.setOnCompletionListener {
//            Toast.makeText(this@MainActivity, "Playback finished", Toast.LENGTH_SHORT).show()
//        }
//    }
//}
//
//@Composable
//fun AudioRecorderUI(
//    onStartRecording: () -> Unit,
//    onStopRecording: () -> Unit,
//    onPlayRecording: () -> Unit
//) {
//    Column(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(16.dp),
//        verticalArrangement = Arrangement.Center
//    ) {
//        Button(onClick = onStartRecording, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Start Recording")
//        }
//        Spacer(modifier = Modifier.height(16.dp))
//        Button(onClick = onStopRecording, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Stop Recording")
//        }
//        Spacer(modifier = Modifier.height(16.dp))
//        Button(onClick = onPlayRecording, modifier = Modifier.fillMaxWidth()) {
//            Text(text = "Play Recording")
//        }
//    }
//}
//
//@Preview(showBackground = true)
//@Composable
//fun DefaultPreview() {
//    SampleMLAppTheme {
//        AudioRecorderUI(
//            onStartRecording = {},
//            onStopRecording = {},
//            onPlayRecording = {}
//        )
//    }
//}
