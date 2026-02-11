package com.leukiemea.alldetection

import java.text.SimpleDateFormat
import java.util.*

/**
 * Generates clinical summaries based on ALL detection results
 * Mirrors the Python llm_utils.py implementation
 */
class SummaryGenerator {
    companion object {
        private var gemmaHelper: GemmaLLMHelper? = null
        private var isLlmInitialized = false
        
        /**
         * Initialize Gemma LLM (call once on app start)
         */
        suspend fun initialize(context: android.content.Context, onImportRequest: () -> Unit) {
            gemmaHelper = GemmaLLMHelper(context, onImportRequest)
            isLlmInitialized = gemmaHelper!!.initialize()
        }
        
        suspend fun importModel(uri: android.net.Uri): Boolean {
            return gemmaHelper?.importModelFromUri(uri) ?: false
        }
        
        /**
         * Generate summary - tries Gemma LLM first, falls back to templates
         */
        suspend fun generateSummary(
            isHealthy: Boolean,
            healthyScore: Int,
            allScore: Int,
            cellCount: Int,
            blastCells: List<CellData> = emptyList()
        ): String {
            // Try Gemma LLM if initialized
            if (isLlmInitialized && gemmaHelper != null) {
                val llmSummary = gemmaHelper!!.generateSummary(
                    isHealthy, healthyScore, allScore, cellCount, blastCells
                )
                if (llmSummary != null && llmSummary.isNotBlank()) {
                    val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
                    return "[$timestamp Gemma-2B] $llmSummary"
                }
            }
            
            // Fallback to template-based generation
            val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
            
            return if (isHealthy) {
                buildHealthySummary(timestamp, healthyScore, allScore, cellCount)
            } else {
                buildALLSummary(timestamp, healthyScore, allScore, cellCount, blastCells)
            }
        }
        
        /**
         * Cleanup LLM resources
         */
        fun cleanup() {
            gemmaHelper?.close()
        }
        
        private fun buildHealthySummary(
            timestamp: String,
            healthyScore: Int,
            allScore: Int,
            cellCount: Int
        ): String {
            val confidence = healthyScore - allScore
            val parts = mutableListOf<String>()
            
            parts.add("Peripheral blood smear appears normal.")
            parts.add("Healthy cell score ($healthyScore%) is significantly higher than ALL score ($allScore%).")
            
            if (cellCount > 0) {
                parts.add("$cellCount cell(s) analyzed with no blast characteristics detected.")
            }
            
            if (confidence > 30) {
                parts.add("High confidence in normal classification.")
            }
            
            return "[$timestamp AI] " + parts.joinToString(" ")
        }
        
        private fun buildALLSummary(
            timestamp: String,
            healthyScore: Int,
            allScore: Int,
            cellCount: Int,
            blastCells: List<CellData>
        ): String {
            val parts = mutableListOf<String>()
           val confidence = allScore - healthyScore
            
            parts.add("⚠️ SUSPECTED ACUTE LYMPHOBLASTIC LEUKEMIA DETECTED")
            parts.add("ALL score ($allScore%) exceeds healthy score ($healthyScore%).")
            
            if (blastCells.isNotEmpty()) {
                parts.add("${blastCells.size} blast cell(s) identified out of $cellCount total WBCs.")
                
                // Add morphological details for blast cells
                val morphDetails = mutableListOf<String>()
                blastCells.forEachIndexed { idx, cell ->
                    val circPercent = (cell.circularity * 100).toInt()
                    val homPercent = (cell.homogeneity * 100).toInt()
                    morphDetails.add("Cell ${idx + 1}: nucleus ${cell.nucArea}px (circ: $circPercent%, hom: $homPercent%, score: ${"%.1f".format(cell.score)})")
                }
                parts.add("Blast morphology: ${morphDetails.joinToString("; ")}.")
            } else if (cellCount > 0) {
                parts.add("Analysis of $cellCount cell(s) shows characteristics consistent with lymphoblasts:")
            }
            
            // Score-based interpretations (mirror Python logic)
            if (confidence > 30) {
                parts.add("High confidence classification - cells show typical blast morphology with round nuclear contours and fine chromatin pattern.")
            } else if (confidence > 15) {
                parts.add("Moderate confidence - cells show some blast-like features. Further examination recommended.")
            } else {
                parts.add("Low confidence - borderline classification. Requires professional review and confirmation.")
            }
            
            parts.add("⚠️ RECOMMENDATION: Immediate consultation with hematologist/oncologist required for professional diagnosis.")
            
            return "[$timestamp AI] " + parts.joinToString(" ")
        }
    }
}
