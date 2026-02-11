package com.leukiemea.alldetection

import android.app.AlertDialog
import android.content.Context
import android.util.Log
import android.widget.ProgressBar
import android.widget.TextView
import androidx.compose.runtime.mutableStateOf
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Helper class for on-device LLM inference using MediaPipe + Gemma 2B
 * Includes automatic model download with progress tracking
 */
class GemmaLLMHelper(
    private val context: Context,
    private val onImportRequest: () -> Unit
) {
    
    private var llmInference: LlmInference? = null
    private var isInitialized = false
    
    // Download progress tracking
    var downloadProgress = mutableStateOf(0f)
    var isDownloading = mutableStateOf(false)
    var downloadError = mutableStateOf<String?>(null)
    
    companion object {
        private const val TAG = "GemmaLLM"
        // Using Qwen instead of Gemma because Gemma is gated/auth-protected.
        // Qwen 2.5 1.5B is public, comparable quality, and auto-downloads.
        private const val MODEL_PATH = "qwen2.5-1.5b-instruct.bin"
        private const val MODEL_SIZE_MB = 1524L // ~1.5GB
        
        // Direct download URL for Qwen 2.5 1.5B Instruct (Public LiteRT Community)
        private const val MODEL_DOWNLOAD_URL = 
            "https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/19edb84c69a0212f29a6ef17ba0d6f278b6a1614/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm?download=true"
    }
    
    /**
     * Initialize the LLM model
     * If model not found, prompts user and downloads automatically
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (isInitialized) return@withContext true
        
        try {
            val modelFile = File(context.filesDir, MODEL_PATH)
            
            // Download model if not present
            if (!modelFile.exists()) {
                Log.i(TAG, "Gemma model not found locally")
                
                // Check if user wants to download
                val shouldDownload = promptUserForDownload()
                if (!shouldDownload) {
                    Log.i(TAG, "User declined model download")
                    return@withContext false
                }
                
                // Download the model
                // Note: downloadModel handles show/hide of progress dialog
                val downloadSuccess = downloadModel(modelFile)
                if (!downloadSuccess) {
                    Log.e(TAG, "Model download failed")
                    return@withContext false
                }
            }
            
            // Initialize LLM with the model
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(512)
                .build()
            
            llmInference = LlmInference.createFromOptions(context, options)
            isInitialized = true
            
            Log.d(TAG, "✅ Gemma 2B model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize Gemma model", e)
            downloadError.value = e.message
            false
        }
    }
    
    /**
     * Prompt user to download the model
     */
    private suspend fun promptUserForDownload(): Boolean = withContext(Dispatchers.Main) {
        var userConsent = false
        val semaphore = java.util.concurrent.Semaphore(0)
        
        val dialog = AlertDialog.Builder(context)
            .setTitle("AI Summary Enhancement")
            .setMessage("Download Gemma AI model (~1.5GB) for natural language clinical summaries?\n\n" +
                    "• Generates human-like reports\n" +
                    "• Works offline after download\n" +
                    "• One-time ~1.5GB download\n\n" +
                    "Without this, template-based summaries will be used.")
            .setPositiveButton("Download") { _, _ ->
                userConsent = true
                semaphore.release()
            }
            .setNeutralButton("Import File") { _, _ ->
                userConsent = false // Don't download
                semaphore.release()
                onImportRequest()
            }
            .setNegativeButton("Skip") { _, _ ->
                userConsent = false
                semaphore.release()
            }
            .setCancelable(false)
            .create()
        
        dialog.show()
        
        // Wait for user response
        withContext(Dispatchers.IO) {
            semaphore.acquire()
        }
        
        userConsent
    }
    
    /**
     * Download the Gemma model with progress tracking UI using OkHttp
     */
    private suspend fun downloadModel(targetFile: File): Boolean = withContext(Dispatchers.Main) {
        isDownloading.value = true
        downloadProgress.value = 0f
        downloadError.value = null
        
        // create a layout for progress dialog
        val layout = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(50, 40, 50, 40)
        }
        
        val tvMessage = TextView(context).apply {
            text = "Downloading Gemma AI Model...\n(This ensures offline capability)"
            textSize = 16f
            setPadding(0, 0, 0, 30)
        }
        
        val progressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = true // Start indeterminate until we get content length
            max = 100
        }
        
        val tvProgress = TextView(context).apply {
            text = "Starting..."
            gravity = android.view.Gravity.END
            setPadding(0, 10, 0, 0)
        }
        
        layout.addView(tvMessage)
        layout.addView(progressBar)
        layout.addView(tvProgress)
        
        val progressDialog = AlertDialog.Builder(context)
            .setView(layout)
            .setCancelable(false) // Prevent cancelling by tapping outside
            .create()
            
        progressDialog.show()
        
        // Run download in IO context but update UI in Main context
        val result = withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Starting model download from: $MODEL_DOWNLOAD_URL")
                
                val client = OkHttpClient.Builder()
                    .connectTimeout(60, TimeUnit.SECONDS) // 1 minute connect
                    .readTimeout(10, TimeUnit.MINUTES)    // 10 minutes read
                    .callTimeout(15, TimeUnit.MINUTES)    // 15 minutes total
                    .followRedirects(true)
                    .followSslRedirects(true)
                    .build()
                
                val request = Request.Builder()
                    .url(MODEL_DOWNLOAD_URL)
                    .build()
                
                val response = client.newCall(request).execute()
                
                if (!response.isSuccessful) {
                    throw IOException("Failed to download model: ${response.code}")
                }
                
                val body = response.body ?: throw IOException("Empty response body")
                val contentLength = body.contentLength()
                
                Log.d(TAG, "Content Length: $contentLength")
                
                // If content length is known, switch to determinate progress
                if (contentLength > 0) {
                    withContext(Dispatchers.Main) {
                        progressBar.isIndeterminate = false
                    }
                }
                
                body.byteStream().use { input ->
                    FileOutputStream(targetFile).use { output ->
                        val buffer = ByteArray(8192) // 8KB buffer
                        var bytesRead: Int
                        var totalBytesRead = 0L
                        var lastUiUpdate = 0L
                        
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            totalBytesRead += bytesRead
                            
                            // Update progress logic
                            val currentTime = System.currentTimeMillis()
                            // Update UI max every 100ms to avoid main thread spam
                            if (currentTime - lastUiUpdate > 100) {
                                val progress = if (contentLength > 0) {
                                    (totalBytesRead.toFloat() / contentLength.toFloat())
                                } else {
                                    // Estimate based on expected size (1.5GB)
                                    (totalBytesRead.toFloat() / (MODEL_SIZE_MB * 1024 * 1024)).coerceAtMost(0.99f)
                                }
                                
                                val progressInt = (progress * 100).toInt()
                                val downloadedMb = totalBytesRead / (1024 * 1024)
                                val totalMb = if (contentLength > 0) contentLength / (1024 * 1024) else MODEL_SIZE_MB
                                
                                withContext(Dispatchers.Main) {
                                    progressBar.progress = progressInt
                                    tvProgress.text = "$progressInt% ($downloadedMb/${totalMb} MB)"
                                }
                                lastUiUpdate = currentTime
                            }
                        }
                    }
                }
                
                Log.i(TAG, "✅ Model download completed successfully")
                true
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ Model download failed", e)
                withContext(Dispatchers.Main) {
                    downloadError.value = e.message ?: "Download failed"
                }
                
                // Clean up partial download
                if (targetFile.exists()) {
                    targetFile.delete()
                }
                
                false
            }
        }
        
        progressDialog.dismiss()
        isDownloading.value = false
        result
    }
    
    /**
     * Generate clinical summary using Gemma
     */
    suspend fun generateSummary(
        isHealthy: Boolean,
        healthyScore: Int,
        allScore: Int,
        cellCount: Int,
        blastCells: List<CellData>
    ): String? = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            Log.w(TAG, "LLM not initialized, cannot generate summary")
            return@withContext null
        }
        
        try {
            val prompt = buildPrompt(isHealthy, healthyScore, allScore, cellCount, blastCells)
            Log.d(TAG, "Generating summary with Gemma...")
            
            val response = llmInference?.generateResponse(prompt)
            
            Log.d(TAG, "✅ Summary generated successfully")
            response
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to generate summary", e)
            null
        }
    }
    
    /**
     * Build clinical prompt for Gemma (similar to Python version)
     */
    private fun buildPrompt(
        isHealthy: Boolean,
        healthyScore: Int,
        allScore: Int,
        cellCount: Int,
        blastCells: List<CellData>
    ): String {
        val sb = StringBuilder()
        
        sb.appendLine("<|im_start|>system")
        sb.appendLine("You are a clinical AI assistant analyzing peripheral blood smear microscopy.")
        sb.appendLine("Generate a concise, professional clinical summary (2-3 sentences) based on the inputs provided.")
        sb.appendLine("<|im_end|>")
        sb.appendLine("<|im_start|>user")
        sb.appendLine("**Analysis Results:**")
        sb.appendLine("- Classification: ${if (isHealthy) "NORMAL" else "SUSPECTED ACUTE LYMPHOBLASTIC LEUKEMIA (ALL)"}")
        sb.appendLine("- Healthy Score: $healthyScore%")
        sb.appendLine("- ALL Score: $allScore%")
        sb.appendLine("- Total WBCs Detected: $cellCount")
        
        if (blastCells.isNotEmpty()) {
            sb.appendLine("- Blast Cells Identified: ${blastCells.size}")
            sb.appendLine()
            sb.appendLine("**Blast Cell Morphology:**")
            sb.appendLine("The following cells exhibit abnormal characteristics:")
            blastCells.take(3).forEachIndexed { idx, cell ->
                sb.appendLine("  • Cell ${idx + 1}: Nucleus Area ${cell.nucArea}px, Circularity ${(cell.circularity * 100).toInt()}%, Homogeneity ${(cell.homogeneity * 100).toInt()}%, Confidence Score ${String.format("%.1f", cell.score)}")
            }
            if (blastCells.size > 3) {
                sb.appendLine("  • ...and ${blastCells.size - 3} more")
            }
        }
        
        sb.appendLine()
        if (isHealthy) {
            sb.appendLine("Provide a brief summary indicating normal morphology.")
        } else {
            sb.appendLine("Provide a clinical summary that:")
            sb.appendLine("1. Notes the number of blast cells with abnormal morphology")
            sb.appendLine("2. Mentions key morphological features (large nuclei, irregular shape)")
            sb.appendLine("3. Recommends urgent hematologist consultation")
            sb.appendLine("4. Start with ⚠️ warning symbol")
        }
        sb.appendLine("<|im_end|>")
        sb.appendLine("<|im_start|>assistant")
        
        return sb.toString()
    }
    
    /**
     * Import model manually from a Uri (picked by user)
     */
    suspend fun importModelFromUri(uri: android.net.Uri): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Importing model from URI: $uri")
            val modelFile = File(context.filesDir, MODEL_PATH)
            
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }
            
            // Validate size roughly
            if (modelFile.length() < 100 * 1024 * 1024) { // < 100MB
                 Log.e(TAG, "Imported file too small")
                 modelFile.delete()
                 return@withContext false
            }
            
            Log.d(TAG, "✅ Model imported successfully")
            
            // Re-initialize logic will be handled by calling initialize() again from MainActivity if needed
            // But since this is a helper, we can just return true and let the caller handle
            return@withContext true
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to import model", e)
            false
        }
    }

    /**
     * Cleanup resources
     */
    fun close() {
        llmInference?.close()
        llmInference = null
        isInitialized = false
        Log.d(TAG, "LLM resources released")
    }
}
