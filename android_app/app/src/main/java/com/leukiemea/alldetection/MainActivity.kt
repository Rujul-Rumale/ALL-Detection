package com.leukiemea.alldetection

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.leukiemea.alldetection.ui.theme.ALLDetectionTheme
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Color
import androidx.compose.ui.graphics.Color as ComposeColor
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {

    private var tfliteInterpreter: Interpreter? = null
    private val MODEL_FILE = "all_nano_33_ble_sense.tflite"
    private val INPUT_SIZE = 100
    
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 100
    }
    
    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (!allGranted) {
            Toast.makeText(
                this,
                "Some permissions denied. App features may be limited.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private val manualImportLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            lifecycleScope.launch(Dispatchers.IO) {
                val success = SummaryGenerator.importModel(it)
                withContext(Dispatchers.Main) {
                    if (success) {
                        Toast.makeText(this@MainActivity, "Model imported successfully!", Toast.LENGTH_LONG).show()
                    } else {
                        Toast.makeText(this@MainActivity, "Import failed. Please select a valid model file.", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Request permissions at launch
        requestPermissions()
        
        // Load TFLite Model
        try {
            tfliteInterpreter = Interpreter(loadModelFile())
        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread {
                Toast.makeText(
                    this,
                    "Failed to load AI model: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

        setContent {
            ALLDetectionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
        
        // Initialize Gemma LLM asynchronously in background
        lifecycleScope.launch(Dispatchers.IO) {
            SummaryGenerator.initialize(this@MainActivity) {
                // onImportRequest callback execution moves to Main thread to launch activity
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Please select 'gemma-2b-it-gpu-int4.bin'", Toast.LENGTH_LONG).show()
                    manualImportLauncher.launch("*/*")
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clean up TFLite interpreter to prevent memory leaks
        tfliteInterpreter?.close()
        tfliteInterpreter = null
        
        // Clean up Gemma LLM resources
        SummaryGenerator.cleanup()
    }
    
    private fun requestPermissions() {
        val permissionsToRequest = mutableListOf<String>()
        
        // Camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
            != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(Manifest.permission.CAMERA)
        }
        
        // Storage permission based on Android version
        val storagePermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }
        
        if (ContextCompat.checkSelfPermission(this, storagePermission) 
            != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(storagePermission)
        }
        
        if (permissionsToRequest.isNotEmpty()) {
            permissionsLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }

    @OptIn(ExperimentalMaterial3Api::class, ExperimentalAnimationApi::class)
    @Composable
    fun MainScreen() {
        var capturedBitmap by remember { mutableStateOf<Bitmap?>(null) }
        var isProcessing by remember { mutableStateOf(false) }
        var resultText by remember { mutableStateOf("Ready for analysis") }
        var confidenceText by remember { mutableStateOf("") }
        var isHealthy by remember { mutableStateOf(true) }
        // Store CellData (bitmap + bounds) instead of just Bitmaps
        var segmentedCells by remember { mutableStateOf<List<CellData>>(emptyList()) }
        var cellCount by remember { mutableStateOf(0) }
        var aiSummary by remember { mutableStateOf("") }
        val scope = rememberCoroutineScope()

        // Permission handling
        val cameraPermissionLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            if (!isGranted) {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }

        // Camera launcher
        val cameraLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == RESULT_OK) {
                val imageBitmap = result.data?.extras?.get("data") as? Bitmap
                imageBitmap?.let {
                    capturedBitmap = it
                    isProcessing = true
                    scope.launch {
                        runInference(it) { healthy, healthyScore, allScore, conf ->
                            isHealthy = healthy
                            resultText = if (healthy) "✓ HEALTHY" else "⚠ SUSPECTED ALL"
                            confidenceText = conf
                            
                            // Offload heavy segmentation and summary generation to background
                            scope.launch(Dispatchers.Default) {
                                val segmenter = WBCSegmenter()
                                val segResult = segmenter.segment(it)
                                
                                // Create a mutable copy to draw on
                                val mutableBitmap = it.copy(Bitmap.Config.ARGB_8888, true)
                                val canvas = Canvas(mutableBitmap)
                                val paint = Paint().apply {
                                    style = Paint.Style.STROKE
                                    strokeWidth = 5f
                                }

                                // Classify each cell and draw box
                                var blastCount = 0
                                val blastCells = mutableListOf<CellData>()
                                segResult.cells.forEach { cell ->
                                    val features = runSingleCellInference(cell.bitmap)
                                    cell.isBlast = features.isBlast
                                    cell.nucArea = features.nucArea
                                    cell.perimeter = features.perimeter
                                    cell.circularity = features.circularity
                                    cell.eccentricity = features.eccentricity
                                    cell.homogeneity = features.homogeneity
                                    cell.score = features.score
                                    
                                    if (features.isBlast) {
                                        blastCount++
                                        blastCells.add(cell)
                                    }
                                    
                                    paint.color = if (features.isBlast) Color.RED else Color.GREEN
                                    canvas.drawRect(cell.bounds, paint)
                                }
                                
                                // Generate summary asynchronously with Gemma LLM
                                lifecycleScope.launch {
                                    val summary = SummaryGenerator.generateSummary(
                                        healthy,
                                        healthyScore,
                                        allScore,
                                        segResult.count,
                                        blastCells
                                    )
                                    aiSummary = summary
                                }
                                
                                withContext(Dispatchers.Main) {
                                    capturedBitmap = mutableBitmap // Update with annotated image
                                    segmentedCells = segResult.cells
                                    cellCount = segResult.count
                                    isProcessing = false
                                }
                            }
                        }
                    }
                }
            }
        }

        // Gallery launcher
        val galleryLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == RESULT_OK) {
                val uri: Uri? = result.data?.data
                uri?.let { imageUri ->
                    try {
                        val imageBitmap = contentResolver.openInputStream(imageUri)?.use { inputStream ->
                            BitmapFactory.decodeStream(inputStream)
                        }
                        
                        if (imageBitmap != null) {
                            Log.d("MainActivity", "Image loaded successfully, starting inference")
                            capturedBitmap = imageBitmap
                            isProcessing = true
                            scope.launch {
                                Log.d("MainActivity", "Coroutine launched for inference")
                                runInference(imageBitmap) { healthy, healthyScore, allScore, conf ->
                                    Log.d("MainActivity", "Inference complete: $conf")
                                    isHealthy = healthy
                                    resultText = if (healthy) "✓ HEALTHY" else "⚠ SUSPECTED ALL"
                                    confidenceText = conf
                                    
                                    // Offload heavy segmentation and summary generation to background
                                    scope.launch(Dispatchers.Default) {
                                        val segmenter = WBCSegmenter()
                                        val segResult = segmenter.segment(imageBitmap)
                                        
                                        // Create a mutable copy to draw on
                                        val mutableBitmap = imageBitmap.copy(Bitmap.Config.ARGB_8888, true)
                                        val canvas = Canvas(mutableBitmap)
                                        val paint = Paint().apply {
                                            style = Paint.Style.STROKE
                                            strokeWidth = 5f
                                        }

                                        // Classify each cell and draw box
                                        var blastCount = 0
                                        val blastCells = mutableListOf<CellData>()
                                        segResult.cells.forEach { cell ->
                                            val features = runSingleCellInference(cell.bitmap)
                                            cell.isBlast = features.isBlast
                                            cell.nucArea = features.nucArea
                                            cell.perimeter = features.perimeter
                                            cell.circularity = features.circularity
                                            cell.eccentricity = features.eccentricity
                                            cell.homogeneity = features.homogeneity
                                            cell.score = features.score
                                            
                                            if (features.isBlast) {
                                                blastCount++
                                                blastCells.add(cell)
                                            }
                                            
                                            paint.color = if (features.isBlast) Color.RED else Color.GREEN
                                            canvas.drawRect(cell.bounds, paint)
                                        }

                                        // Update summary logic based on individual cell counts if needed
                                        // or keep using global scores (mixed approach is safer for now)
                                        // Generate summary asynchronously with Gemma LLM
                                        lifecycleScope.launch {
                                            val summary = SummaryGenerator.generateSummary(
                                                healthy,
                                                healthyScore,
                                                allScore,
                                                segResult.count,
                                                blastCells
                                            )
                                            aiSummary = summary
                                        }
                                        
                                        withContext(Dispatchers.Main) {
                                            capturedBitmap = mutableBitmap // Update with annotated image
                                            segmentedCells = segResult.cells
                                            cellCount = segResult.count
                                            isProcessing = false
                                        }
                                    }
                                }
                            }
                        } else {
                            Log.e("MainActivity", "Failed to decode bitmap")
                            Toast.makeText(
                                this@MainActivity,
                                "Failed to load image",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Error loading image", e)
                        e.printStackTrace()
                        Toast.makeText(
                            this@MainActivity,
                            "Error loading image: ${e.message}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        }
        
        val storagePermissionLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            if (isGranted) {
                val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                galleryLauncher.launch(intent)
            } else {
                Toast.makeText(this, "Storage permission denied", Toast.LENGTH_SHORT).show()
            }
        }


        Scaffold(
            topBar = {
                CenterAlignedTopAppBar(
                    title = {
                        Text(
                            "ALL Detection",
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold
                        )
                    },
                    colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                        titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                )
            }
        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Top
            ) {
                // Image Preview Card with Animation
                AnimatedVisibility(
                    visible = capturedBitmap != null,
                    enter = fadeIn() + scaleIn(),
                    exit = fadeOut() + scaleOut()
                ) {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(400.dp)
                            .padding(bottom = 16.dp),
                        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        capturedBitmap?.let { bitmap ->
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = "Captured Image",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Fit
                            )
                        }
                    }
                }

                // Result Card with Animated Color Transition
                val backgroundColor by animateColorAsState(
                    targetValue = when {
                        isProcessing -> MaterialTheme.colorScheme.surfaceVariant
                        isHealthy -> MaterialTheme.colorScheme.primaryContainer
                        else -> MaterialTheme.colorScheme.errorContainer
                    },
                    animationSpec = tween(durationMillis = 500),
                    label = "bgColor"
                )

                val contentColor by animateColorAsState(
                    targetValue = when {
                        isProcessing -> MaterialTheme.colorScheme.onSurfaceVariant
                        isHealthy -> MaterialTheme.colorScheme.onPrimaryContainer
                        else -> MaterialTheme.colorScheme.onErrorContainer
                    },
                    animationSpec = tween(durationMillis = 500),
                    label = "contentColor"
                )

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = backgroundColor,
                        contentColor = contentColor
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
                    shape = RoundedCornerShape(20.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        if (isProcessing) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(48.dp),
                                color = MaterialTheme.colorScheme.primary
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                        }

                        AnimatedContent(
                            targetState = resultText,
                            transitionSpec = {
                                fadeIn() + slideInVertically() with fadeOut() + slideOutVertically()
                            },
                            label = "resultText"
                        ) { text ->
                            Text(
                                text = text,
                                style = MaterialTheme.typography.titleLarge,
                                fontWeight = FontWeight.Bold
                            )
                        }


                        if (confidenceText.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = confidenceText,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }

                // Segmented Cells Section (smooth animation)
                AnimatedVisibility(
                    visible = segmentedCells.isNotEmpty(),
                    enter = fadeIn(animationSpec = tween(600)) + expandVertically(),
                    exit = fadeOut() + shrinkVertically()
                ) {
                    Column(modifier = Modifier.padding(bottom = 16.dp)) {
                        Text(
                            text = "Detected Cells ($cellCount)",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(bottom = 12.dp)
                        )
                        
                        androidx.compose.foundation.lazy.LazyRow(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            itemsIndexed(segmentedCells) { index, cellData ->
                                Card(
                                    modifier = Modifier.size(90.dp),
                                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                                    shape = RoundedCornerShape(12.dp),
                                    border = BorderStroke(
                                        width = 3.dp,
                                        color = if (cellData.isBlast) ComposeColor.Red else ComposeColor.Green
                                    )
                                ) {
                                    Image(
                                        bitmap = cellData.bitmap.asImageBitmap(),
                                        contentDescription = "Cell ${index + 1}",
                                        modifier = Modifier.fillMaxSize(),
                                        contentScale = ContentScale.Crop
                                    )
                                }
                            }
                        }
                    }
                }

                // AI Summary Card (smooth animation)
                AnimatedVisibility(
                    visible = aiSummary.isNotEmpty(),
                    enter = fadeIn(animationSpec = tween(800)) + expandVertically(),
                    exit = fadeOut() + shrinkVertically()
                ) {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                            contentColor = MaterialTheme.colorScheme.onTertiaryContainer
                        ),
                        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(20.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(bottom = 12.dp)
                            ) {
                                Text(
                                    text = "🤖",
                                    style = MaterialTheme.typography.headlineSmall,
                                    modifier = Modifier.padding(end = 10.dp)
                                )
                                Text(
                                    text = "AI Analysis",
                                    style = MaterialTheme.typography.titleLarge,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            Text(
                                text = aiSummary,
                                style = MaterialTheme.typography.bodyMedium,
                                lineHeight = 22.sp
                            )
                        }
                    }
                }

                // Action Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    FilledTonalButton(
                        onClick = {
                            val hasCameraPermission = ContextCompat.checkSelfPermission(
                                this@MainActivity,
                                Manifest.permission.CAMERA
                            ) == PackageManager.PERMISSION_GRANTED

                            if (hasCameraPermission) {
                                val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                                cameraLauncher.launch(intent)
                            } else {
                                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                            }
                        },
                        modifier = Modifier.weight(1f),
                        enabled = !isProcessing
                    ) {
                        Icon(Icons.Default.CameraAlt, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Camera")
                    }

                    FilledTonalButton(
                        onClick = {
                            // Check storage permission based on Android version
                            val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                Manifest.permission.READ_MEDIA_IMAGES
                            } else {
                                Manifest.permission.READ_EXTERNAL_STORAGE
                            }
                            
                            val hasPermission = ContextCompat.checkSelfPermission(
                                this@MainActivity,
                                permission
                            ) == PackageManager.PERMISSION_GRANTED
                            
                            if (hasPermission) {
                                val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                                galleryLauncher.launch(intent)
                            } else {
                                storagePermissionLauncher.launch(permission)
                            }
                        },
                        modifier = Modifier.weight(1f),
                        enabled = !isProcessing
                    ) {
                        Icon(Icons.Default.PhotoLibrary, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Gallery")
                    }
                }
            }
        }
    }

    // Define CellData here, assuming it's a nested class or top-level class
    // CellData is now in WBCSegmenter class

    private suspend fun runInference(
        bitmap: Bitmap,
        onResult: (healthy: Boolean, healthyScore: Int, allScore: Int, confidenceText: String) -> Unit
    ) = withContext(Dispatchers.IO) {
        try {
            if (tfliteInterpreter == null) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Model not loaded. Please restart the app.",
                        Toast.LENGTH_LONG
                    ).show()
                    onResult(true, 0, 0, "Error: Model not loaded")
                }
                return@withContext
            }


            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val byteBuffer = convertBitmapToByteBuffer(resizedBitmap)
            val output = Array(1) { ByteArray(2) }
            
            // Normalize: (Value - 128) to center around 0 for INT8 input
            // byteBuffer.put((pixelVal - 128).toByte())
            // Implementation in convertBitmapToByteBuffer
            
            tfliteInterpreter?.run(byteBuffer, output)

            // INT8 quantized output: Properly dequantize using model parameters
            // Python training: CLASS_NAMES = ['hem', 'all'] -> index 0 = Healthy, index 1 = ALL
            // Python comment: "Model output is already probabilities (quantized 0.0-1.0)"
            val interpreter = tfliteInterpreter!!
            val outputTensor = interpreter.getOutputTensor(0)
            val scale = outputTensor.quantizationParams().scale
            val zeroPoint = outputTensor.quantizationParams().zeroPoint
            
            // Dequantize: real_value = (quantized_value - zero_point) * scale
            // Output is already probabilities after dequantization - NO softmax needed
            val healthyProb = (output[0][0].toInt() - zeroPoint) * scale
            val allProb = (output[0][1].toInt() - zeroPoint) * scale
            
            Log.d("MainActivity", "Dequantized probabilities - Healthy: $healthyProb, ALL: $allProb (scale=$scale, zp=$zeroPoint)")
            Log.d("MainActivity", "Raw INT8 bytes - [0]=${output[0][0]}, [1]=${output[0][1]}")
            
            // Convert to percentages (values should already be in 0.0-1.0 range)
            val healthyPercent = (healthyProb * 100).toInt().coerceIn(0, 100)
            val allPercent = (allProb * 100).toInt().coerceIn(0, 100)
            val isHealthy = healthyProb > allProb
            
            Log.d("MainActivity", "Final probabilities - Healthy: $healthyPercent%, ALL: $allPercent%")

            runOnUiThread {
                Log.d("MainActivity", "Updating UI with results")
                onResult(isHealthy, healthyPercent, allPercent, "Healthy: $healthyPercent% | ALL: $allPercent%")
            }
        } catch (e: Exception) {
            e.printStackTrace()
            withContext(Dispatchers.Main) {
                Toast.makeText(
                    this@MainActivity,
                    "Inference error: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                onResult(true, 0, 0, "Error: ${e.message}")
            }
        }
    }

    data class CellFeatures(
        val isBlast: Boolean,
        val nucArea: Int,
        val perimeter: Int,
        val circularity: Float,
        val eccentricity: Float,
        val homogeneity: Float,
        val score: Float
    )
    
    /**
     * Feature-based blast detection - PROPER PORT from Python blast_detector_v5.py
     * Segments nucleus within cell crop, then calculates features on nucleus mask
     */
    private fun runSingleCellInference(bitmap: Bitmap): CellFeatures {
        try {
            val width = bitmap.width
            val height = bitmap.height
            
            // **1. Convert to grayscale**
            val gray = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(gray)
            val paint = android.graphics.Paint()
            val colorMatrix = android.graphics.ColorMatrix()
            colorMatrix.setSaturation(0f)
            paint.colorFilter = android.graphics.ColorMatrixColorFilter(colorMatrix)
            canvas.drawBitmap(bitmap, 0f, 0f, paint)
            
            val grayPixels = IntArray(width * height)
            gray.getPixels(grayPixels, 0, width, 0, 0, width, height)
            val grayValues = grayPixels.map { (it and 0xFF) }
            
            // **2. Otsu thresholding to segment nucleus (dark region)**
            // Calculate histogram
            val histogram = IntArray(256)
            grayValues.forEach { histogram[it]++ }
            
            // Otsu's method
            val total = width * height
            var sum = 0.0
            for (i in 0..255) sum += i * histogram[i]
            
            var sumB = 0.0
            var wB = 0
            var wF: Int
            var maxVariance = 0.0
            var threshold = 0
            
            for (t in 0..255) {
                wB += histogram[t]
                if (wB == 0) continue
                wF = total - wB
                if (wF == 0) break
                
                sumB += t * histogram[t]
                val mB = sumB / wB
                val mF = (sum - sumB) / wF
                val variance = wB.toDouble() * wF.toDouble() * (mB - mF) * (mB - mF)
                
                if (variance > maxVariance) {
                    maxVariance = variance
                    threshold = t
                }
            }
            
            // Create binary mask: nucleus pixels are BELOW threshold (dark)
            val nucleusMask = BooleanArray(width * height)
            for (i in grayValues.indices) {
                nucleusMask[i] = grayValues[i] < threshold
            }
            
            // **3. Find largest connected component (nucleus)**
            val visited = BooleanArray(width * height)
            var largestArea = 0
            var largestPerimeter = 0
            var largestEccentricity = 0f
            
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val idx = y * width + x
                    if (nucleusMask[idx] && !visited[idx]) {
                        // Flood fill to find component
                        val (area, perimeter) = floodFillNucleus(nucleusMask, visited, x, y, width, height)
                        
                        if (area > largestArea) {
                            largestArea = area
                            largestPerimeter = perimeter
                        }
                    }
                }
            }
            
            // Reject if nucleus too small (Python MIN_NUC_AREA = 500)
            if (largestArea < 500) {
                Log.d("MainActivity", "Cell rejected - nucleus too small (area = $largestArea)")
                return CellFeatures(false, 0, 0, 0f, 0f, 0f, 0f)
            }
            
            // **4. Calculate features on nucleus**
            // Circularity (Python formula)
            val circularity = if (largestPerimeter > 0) {
                ((4.0 * Math.PI * largestArea) / (largestPerimeter * largestPerimeter)).toFloat()
            } else 0f
            
            // Eccentricity approximation from aspect ratio of crop
            val aspectRatio = width.toFloat() / height.toFloat()
            val eccentricity = if (aspectRatio > 1) 1.0f - (1.0f / aspectRatio) else 1.0f - aspectRatio
            
            if (eccentricity > 0.85f) {
                Log.d("MainActivity", "Cell rejected as debris (ecc = $eccentricity)")
                return CellFeatures(false, 0, 0, 0f, 0f, 0f, 0f)
            }
            
            // Homogeneity from variance (simplified GLCM)
            val mean = grayValues.average().toFloat()
            val variance = grayValues.map { (it - mean) * (it - mean) }.average().toFloat()
            val stdDev = kotlin.math.sqrt(variance)
            val homogeneity = 1.0f - (stdDev / 128.0f).coerceIn(0f, 1f)
            
            // **5. Scoring (PYTHON FORMULA with REAL nucleus area)**
            val s_area = (largestArea / 1500.0f).coerceAtMost(1.2f)
            val s_circ = circularity
            val s_tex = homogeneity
            
            val totalScore = (s_area * 1.0f) + (s_circ * 1.5f) + (s_tex * 1.2f)
            
            // Python cutoff = 3.2
            val isBlast = totalScore > 3.2f
            
            Log.d("MainActivity", "Cell - NucArea:$largestArea, Perim:$largestPerimeter, Circ:$circularity, Ecc:$eccentricity, Hom:$homogeneity, Score:$totalScore -> ${if (isBlast) "BLAST" else "Normal"}")
            
            return CellFeatures(
                isBlast = isBlast,
                nucArea = largestArea,
                perimeter = largestPerimeter,
                circularity = circularity,
                eccentricity = eccentricity,
                homogeneity = homogeneity,
                score = totalScore
            )
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("MainActivity", "Feature extraction failed", e)
            return CellFeatures(false, 0, 0, 0f, 0f, 0f, 0f)
        }
    }
    
    /**
     * Flood fill to find connected component area and perimeter
     * Returns Pair(area, perimeter)
     */
    private fun floodFillNucleus(
        mask: BooleanArray,
        visited: BooleanArray,
        startX: Int,
        startY: Int,
        width: Int,
        height: Int
    ): Pair<Int, Int> {
        val qx = IntArray(width * height)
        val qy = IntArray(width * height)
        var head = 0
        var tail = 0
        
        qx[tail] = startX
        qy[tail] = startY
        tail++
        
        visited[startY * width + startX] = true
        val componentPixels = mutableListOf<Pair<Int, Int>>()
        
        // First pass: collect all pixels in component
        while (head < tail) {
            val cx = qx[head]
            val cy = qy[head]
            head++
            componentPixels.add(Pair(cx, cy))
            
            // Check 4 neighbors for flood fill
            val neighbors = listOf(
                Pair(cx + 1, cy), Pair(cx - 1, cy),
                Pair(cx, cy + 1), Pair(cx, cy - 1)
            )
            
            for ((nx, ny) in neighbors) {
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    val idx = ny * width + nx
                    if (mask[idx] && !visited[idx]) {
                        visited[idx] = true
                        qx[tail] = nx
                        qy[tail] = ny
                        tail++
                    }
                }
            }
        }
        
        val area = componentPixels.size
        
        // Second pass: count perimeter (pixels with at least one non-nucleus neighbor)
        var perimeter = 0
        for ((px, py) in componentPixels) {
            val neighbors = listOf(
                Pair(px + 1, py), Pair(px - 1, py),
                Pair(px, py + 1), Pair(px, py - 1)
            )
            
            var isBoundary = false
            for ((nx, ny) in neighbors) {
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    isBoundary = true
                    break
                }
                val idx = ny * width + nx
                if (!mask[idx]) {
                    isBoundary = true
                    break
                }
            }
            
            if (isBoundary) perimeter++
        }
        
        return Pair(area, perimeter)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 1)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // 1. Calculate Y channel for Histogram Equalization
        val Y = IntArray(intValues.size)
        val R = IntArray(intValues.size)
        val G = IntArray(intValues.size)
        val B = IntArray(intValues.size)

        for (i in intValues.indices) {
            val pixel = intValues[i]
            R[i] = (pixel shr 16) and 0xFF
            G[i] = (pixel shr 8) and 0xFF
            B[i] = pixel and 0xFF
            // BT.601 conversion
            Y[i] = ((0.299 * R[i]) + (0.587 * G[i]) + (0.114 * B[i])).toInt().coerceIn(0, 255)
        }

        // 2. Compute Histogram
        val histogram = IntArray(256)
        for (y in Y) histogram[y]++

        // 3. Compute CDF
        val cdf = IntArray(256)
        cdf[0] = histogram[0]
        for (i in 1..255) cdf[i] = cdf[i - 1] + histogram[i]

        // 4. Compute Equalized Y map
        val cdfMin = cdf.find { it > 0 } ?: 0
        val totalPixels = intValues.size
        val equalizedY = IntArray(256)
        
        if (totalPixels - cdfMin > 0) {
            for (i in 0..255) {
                equalizedY[i] = ((((cdf[i] - cdfMin).toFloat() / (totalPixels - cdfMin)) * 255) + 0.5f).toInt().coerceIn(0, 255)
            }
        } else {
            for (i in 0..255) equalizedY[i] = i // Fallback
        }

        // 5. Apply EQ and Fill ByteBuffer (INT8 Quantized with model params)
        // Get input quantization parameters from model
        val inputTensor = tfliteInterpreter!!.getInputTensor(0)
        val inputScale = inputTensor.quantizationParams().scale
        val inputZeroPoint = inputTensor.quantizationParams().zeroPoint
        
        Log.d("MainActivity", "Input quantization - scale=$inputScale, zeroPoint=$inputZeroPoint")
        
        for (i in intValues.indices) {
            val oldY = Y[i]
            val newY = equalizedY[oldY]
            
            // Scale RGB to match new Luminance
            // Avoid division by zero
            val scale = if (oldY > 0) newY.toFloat() / oldY else 1.0f

            val r = (R[i] * scale).toInt().coerceIn(0, 255)
            val g = (G[i] * scale).toInt().coerceIn(0, 255)
            val b = (B[i] * scale).toInt().coerceIn(0, 255)

            // Quantize using model's input parameters (matches Python)
            // Python: image = (image / 255.0 / input_scale) + input_zero_point
            val rQuantized = ((r / 255.0f / inputScale) + inputZeroPoint).toInt().coerceIn(-128, 127).toByte()
            val gQuantized = ((g / 255.0f / inputScale) + inputZeroPoint).toInt().coerceIn(-128, 127).toByte()
            val bQuantized = ((b / 255.0f / inputScale) + inputZeroPoint).toInt().coerceIn(-128, 127).toByte()
            
            byteBuffer.put(rQuantized)
            byteBuffer.put(gQuantized)
            byteBuffer.put(bQuantized)
        }
        
        
        return byteBuffer
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd(MODEL_FILE)
        FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }
}
