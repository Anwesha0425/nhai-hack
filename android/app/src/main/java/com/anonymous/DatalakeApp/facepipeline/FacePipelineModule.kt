package com.anonymous.DatalakeApp.facepipeline

import android.content.Context
import com.facebook.react.bridge.*
import java.io.File
import java.io.FileOutputStream

class FacePipelineModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val NAME = "FacePipelineModule"
        init {
            System.loadLibrary("face_pipeline")
        }
    }

    override fun getName(): String = NAME

    private var isInitialized = false

    @ReactMethod
    fun initialize(promise: Promise) {
        try {
            val detectorPath = copyAssetToInternal("models/face_detector.tflite", "face_detector.tflite")
            val landmarkPath = copyAssetToInternal("models/face_landmark.tflite", "face_landmark.tflite")
            val recognizerPath = copyAssetToInternal("models/face_recognition.tflite", "face_recognition.tflite")

            val success = nativeInitialize(detectorPath, landmarkPath, recognizerPath, 640, 480)
            isInitialized = success
            if (success) {
                promise.resolve(true)
            } else {
                promise.reject("INIT_FAILED", "Failed to initialize native face pipeline")
            }
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun processFrame(rgbBase64: String, width: Int, height: Int, promise: Promise) {
        if (!isInitialized) {
            promise.reject("NOT_INITIALIZED", "Pipeline not initialized")
            return
        }
        try {
            val rgbBytes = android.util.Base64.decode(rgbBase64, android.util.Base64.DEFAULT)
            val result = nativeProcessFrame(rgbBytes, width, height)
            if (result != null) {
                promise.resolve(result)
            } else {
                promise.reject("PROCESS_FAILED", "Native processing returned null")
            }
        } catch (e: Exception) {
            promise.reject("PROCESS_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun processFrameBytes(rgbBytes: ReadableArray, width: Int, height: Int, promise: Promise) {
        if (!isInitialized) {
            promise.reject("NOT_INITIALIZED", "Pipeline not initialized")
            return
        }
        try {
            val bytes = ByteArray(rgbBytes.size())
            for (i in 0 until rgbBytes.size()) {
                bytes[i] = rgbBytes.getInt(i).toByte()
            }
            val result = nativeProcessFrame(bytes, width, height)
            if (result != null) {
                promise.resolve(result)
            } else {
                promise.reject("PROCESS_FAILED", "Native processing returned null")
            }
        } catch (e: Exception) {
            promise.reject("PROCESS_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun resetLiveness(promise: Promise) {
        nativeResetLiveness()
        promise.resolve(true)
    }

    @ReactMethod
    fun getEmbeddingDim(promise: Promise) {
        promise.resolve(nativeGetEmbeddingDim())
    }

    @ReactMethod
    fun setTargetEmbeddings(embeddings: ReadableArray, promise: Promise) {
        if (!isInitialized) {
            promise.reject("NOT_INITIALIZED", "Pipeline not initialized")
            return
        }
        try {
            // Flatten the array of embeddings (each embedding is an array of numbers)
            // Or pass an array of float arrays. ReadableArray of ReadableArrays
            val floatArrays = ArrayList<FloatArray>()
            for (i in 0 until embeddings.size()) {
                val emb = embeddings.getArray(i)
                if (emb != null) {
                    val floatArr = FloatArray(emb.size())
                    for (j in 0 until emb.size()) {
                        floatArr[j] = emb.getDouble(j).toFloat()
                    }
                    floatArrays.add(floatArr)
                }
            }
            nativeSetTargetEmbeddings(floatArrays.toTypedArray())
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("PROCESS_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun processImageFile(filePath: String, promise: Promise) {
        if (!isInitialized) {
            promise.reject("NOT_INITIALIZED", "Pipeline not initialized")
            return
        }
        try {
            val path = filePath.replace("file://", "")
            
            // First decode with inJustDecodeBounds=true to check dimensions
            val options = android.graphics.BitmapFactory.Options()
            options.inJustDecodeBounds = true
            android.graphics.BitmapFactory.decodeFile(path, options)
            
            // Calculate inSampleSize to avoid OutOfMemoryError
            var inSampleSize = 1
            if (options.outHeight > 640 || options.outWidth > 640) {
                val halfHeight: Int = options.outHeight / 2
                val halfWidth: Int = options.outWidth / 2
                while (halfHeight / inSampleSize >= 640 && halfWidth / inSampleSize >= 640) {
                    inSampleSize *= 2
                }
            }
            
            // Decode bitmap with inSampleSize set
            options.inJustDecodeBounds = false
            options.inSampleSize = inSampleSize
            val bitmap = android.graphics.BitmapFactory.decodeFile(path, options)
            
            if (bitmap == null) {
                promise.reject("DECODE_FAILED", "Failed to decode image file")
                return
            }
            
            // Final exact scaling if needed
            var scaled = bitmap
            if (bitmap.width > 640 || bitmap.height > 640) {
                val scale = Math.min(640f / bitmap.width, 640f / bitmap.height)
                scaled = android.graphics.Bitmap.createScaledBitmap(bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true)
            }
            
            val width = scaled.width
            val height = scaled.height
            val pixels = IntArray(width * height)
            scaled.getPixels(pixels, 0, width, 0, 0, width, height)
            
            val rgbBytes = ByteArray(width * height * 3)
            var index = 0
            for (pixel in pixels) {
                rgbBytes[index++] = ((pixel shr 16) and 0xFF).toByte() // R
                rgbBytes[index++] = ((pixel shr 8) and 0xFF).toByte()  // G
                rgbBytes[index++] = (pixel and 0xFF).toByte()         // B
            }
            
            val result = nativeProcessFrame(rgbBytes, width, height)
            if (result != null) {
                promise.resolve(result)
            } else {
                promise.reject("PROCESS_FAILED", "Native processing returned null")
            }
        } catch (e: Throwable) {
            promise.reject("PROCESS_ERROR", e.message, e)
        }
    }

    private fun copyAssetToInternal(assetName: String, outputName: String): String {
        val outFile = File(reactContext.filesDir, outputName)
        if (outFile.exists()) return outFile.absolutePath

        try {
            reactContext.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {
            val altNames = listOf(assetName, assetName.replace("models/", ""))
            for (alt in altNames) {
                try {
                    reactContext.assets.open(alt).use { input ->
                        FileOutputStream(outFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    return outFile.absolutePath
                } catch (_: Exception) {}
            }
            throw e
        }
        return outFile.absolutePath
    }

    private external fun nativeInitialize(
        detectorPath: String,
        landmarkPath: String,
        recognizerPath: String,
        width: Int,
        height: Int
    ): Boolean

    private external fun nativeProcessFrame(
        rgbData: ByteArray,
        width: Int,
        height: Int
    ): WritableMap?

    private external fun nativeResetLiveness()
    private external fun nativeGetEmbeddingDim(): Int
    private external fun nativeSetTargetEmbeddings(embeddings: Array<FloatArray>)
}
