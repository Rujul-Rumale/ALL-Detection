package com.leukiemea.alldetection

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlin.math.min
import kotlin.math.max

/**
 * Simple WBC Segmenter using native Android Bitmap APIs
 * Lightweight alternative to OpenCV for mobile
 */
class WBCSegmenter(
    private val minCellSize: Int = 2500, // Drastically increased for High-Res images (approx 50x50px)
    private val padding: Int = 10,
    private val darknessThreshold: Int = 130
) {
    fun segment(bitmap: Bitmap): SegmentationResult {
        try {
            val width = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            
            val mask = BooleanArray(pixels.size)
            // Color filtering: Dark AND NOT Red-dominant (RBCs are red/pink)
            for (i in pixels.indices) {
                mask[i] = isPotentialWBC(pixels[i])
            }
            
            val visited = BooleanArray(pixels.size)
            val cells = mutableListOf<CellData>()
            
            // Reusable queue arrays to avoid object allocation
            val qx = IntArray(width * height)
            val qy = IntArray(width * height)
            
            for (y in 0 until height step 2) { // optimization: skip every other row for seed finding
                for (x in 0 until width step 2) {
                    val idx = y * width + x
                    if (mask[idx] && !visited[idx]) {
                        val (minX, minY, maxX, maxY, pixelCount) = floodFill(
                            mask, visited, x, y, width, height, qx, qy
                        )
                        
                        // Filter by size
                        // RBCs are small (approx 5-7 microns). WBCs are larger (10-20 microns).
                        // Blast cells are often larger.
                        // Increasing threshold to skip RBCs.
                        if (pixelCount >= minCellSize) {
                            // Extract cell
                            val cellW = maxX - minX + 1
                            val cellH = maxY - minY + 1
                            
                            // Aspect ratio check (cells are roughly circular/oval)
                            val ratio = cellW.toFloat() / cellH.toFloat()
                            if (ratio in 0.5f..2.0f) {
                                // Add proportional padding (context) for ML model
                                // Python code uses 50% padding
                                val padW = (cellW * 0.5).toInt()
                                val padH = (cellH * 0.5).toInt()
                                
                                val paddedMinX = max(0, minX - padW)
                                val paddedMinY = max(0, minY - padH)
                                val paddedMaxX = min(width - 1, maxX + padW)
                                val paddedMaxY = min(height - 1, maxY + padH)
                                
                                val finalW = paddedMaxX - paddedMinX
                                val finalH = paddedMaxY - paddedMinY

                                val cellBitmap = Bitmap.createBitmap(
                                    bitmap, paddedMinX, paddedMinY, finalW, finalH
                                )
                                cells.add(CellData(
                                    bitmap = cellBitmap,
                                    bounds = android.graphics.Rect(paddedMinX, paddedMinY, paddedMaxX, paddedMaxY)
                                ))
                            }
                        }
                    }
                }
            }
            
            Log.d("WBCSegmenter", "Segmented ${cells.size} cells")
            return SegmentationResult(cells, cells.size)
            
        } catch (e: Exception) {
            Log.e("WBCSegmenter", "Segmentation failed", e)
            return SegmentationResult(emptyList(), 0)
        }
    }

    private fun isPotentialWBC(color: Int): Boolean {
        val r = (color shr 16) and 0xFF
        val g = (color shr 8) and 0xFF
        val b = color and 0xFF
        
        // 1. Luminance check
        // RBCs are generally lighter than WBC nuclei.
        // Lower threshold to pick only the dark nuclei.
        val luminance = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
        if (luminance > 100) return false // Stricter threshold (was 110/130)
        
        // 2. Color Hue Check for Purple/Blue (WBC Nucleus)
        // Stained WBC nuclei are deep purple/blue (High B, High R, Low G)
        // RBCs are pink/red/orange (High R, Medium G, Low B) or greyish
        
        // Key differentiator: In purple/blue, Green is usually the lowest component.
        // In many RBCs, Green is higher than Blue (making it red/orange/grey).
        
        val maxRB = max(r, b)
        if (g > maxRB * 0.8) return false // Stricter: Green must be low
        
        // RBC Rejection:
        // If it's mostly Red and very little Blue, it's an RBC.
        // WBCs have significant Blue component.
        // Tightening: If Red is even 10% higher than Blue, reject.
        if (r > b * 1.2) return false // Was 1.5
        
        return true
    }
    
    // Returns [minX, minY, maxX, maxY, pixelCount]
    private fun floodFill(
        mask: BooleanArray,
        visited: BooleanArray,
        startX: Int,
        startY: Int,
        width: Int,
        height: Int,
        qx: IntArray,
        qy: IntArray
    ): IntArray {
        var head = 0
        var tail = 0
        
        qx[tail] = startX
        qy[tail] = startY
        tail++
        
        visited[startY * width + startX] = true
        
        var minX = startX
        var maxX = startX
        var minY = startY
        var maxY = startY
        var count = 0
        
        while (head < tail) {
            val cx = qx[head]
            val cy = qy[head]
            head++
            count++
            
            if (cx < minX) minX = cx
            if (cx > maxX) maxX = cx
            if (cy < minY) minY = cy
            if (cy > maxY) maxY = cy
            
            // Check 4 neighbors
            // Right
            if (cx + 1 < width) {
                val idx = cy * width + (cx + 1)
                if (mask[idx] && !visited[idx]) {
                    visited[idx] = true
                    qx[tail] = cx + 1
                    qy[tail] = cy
                    tail++
                }
            }
            // Left
            if (cx - 1 >= 0) {
                val idx = cy * width + (cx - 1)
                if (mask[idx] && !visited[idx]) {
                    visited[idx] = true
                    qx[tail] = cx - 1
                    qy[tail] = cy
                    tail++
                }
            }
            // Down
            if (cy + 1 < height) {
                val idx = (cy + 1) * width + cx
                if (mask[idx] && !visited[idx]) {
                    visited[idx] = true
                    qx[tail] = cx
                    qy[tail] = cy + 1
                    tail++
                }
            }
            // Up
            if (cy - 1 >= 0) {
                val idx = (cy - 1) * width + cx
                if (mask[idx] && !visited[idx]) {
                    visited[idx] = true
                    qx[tail] = cx
                    qy[tail] = cy - 1
                    tail++
                }
            }
        }
        
        return intArrayOf(minX, minY, maxX, maxY, count)
    }
}

data class SegmentationResult(
    val cells: List<CellData>,
    val count: Int
)

data class CellData(
    val bitmap: Bitmap,
    val bounds: android.graphics.Rect,
    var isBlast: Boolean = false,
    var nucArea: Int = 0,
    var perimeter: Int = 0,
    var circularity: Float = 0f,
    var eccentricity: Float = 0f,
    var homogeneity: Float = 0f,
    var score: Float = 0f
)
