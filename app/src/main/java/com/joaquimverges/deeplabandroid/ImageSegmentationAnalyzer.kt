package com.joaquimverges.deeplabandroid

import android.content.res.AssetManager
import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import io.reactivex.BackpressureStrategy
import io.reactivex.Flowable
import io.reactivex.subjects.PublishSubject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.system.measureTimeMillis

/**
 * Image Analyzer that segments the camera input using Deeplab v3 model
 */
class ImageSegmentationAnalyzer(
    assets: AssetManager
) : ImageAnalysis.Analyzer {

    companion object {
        private const val IMAGE_MEAN: Int = 128
        private const val IMAGE_STD: Float = 128f
        private const val DIM_BATCH_SIZE: Int = 1
        private const val DIM_IMG_SIZE_Y: Int = 257
        private const val DIM_IMG_SIZE_X: Int = 257
        private const val DIM_PIXEL_SIZE: Int = 3
        private const val BYTES_PER_POINT: Int = 4
        private const val OUTPUT_LABELS: Int = 21
        private const val MODEL_FILENAME = "deeplabv3_257_mv_gpu.tflite"

        /** Memory-map the model file in Assets.  */
        @Throws(IOException::class)
        fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
            val fileDescriptor = assets.openFd(modelFilename)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }

    data class SegmentationResults(val bitmapMask: Bitmap?, val seenObjects: String)

    private val imgData: ByteBuffer
    private val output: ByteBuffer
    private var segmentBits: Array<IntArray>
    private var segmentColors: IntArray
    private val random = Random(System.currentTimeMillis())
    private val modelFile = loadModelFile(assets, MODEL_FILENAME)
    private val options = Interpreter.Options().addDelegate(GpuDelegate())
    private val interpreter = Interpreter(modelFile, options)
    private val resultNotifier = PublishSubject.create<SegmentationResults>()

    // TODO static colors
    private val labelMap = mapOf(
        1 to "Plane",
        2 to "Bycicle",
        3 to "Bird",
        4 to "Boat",
        5 to "Bottle",
        6 to "Bus",
        7 to "Car",
        8 to "Cat",
        9 to "Chair",
        10 to "Cow",
        11 to "Table",
        12 to "Dog",
        13 to "Horse",
        14 to "Bike",
        15 to "Person",
        16 to "Plant",
        17 to "Sheep",
        18 to "Sofa",
        19 to "Train",
        20 to "TV"
    )

    init {
        // input shape: { 1 x 257 (width) x 257 (height) x 3 (rgb) x 4 (float32) }
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE * BYTES_PER_POINT)
        imgData.order(ByteOrder.nativeOrder())

        // output shape: { 1 x 257 (width) x 257 (height) x 21 (output labels) x 4 (float32) }
        output = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * OUTPUT_LABELS * BYTES_PER_POINT)
        output.order(ByteOrder.nativeOrder())

        // stores the highest scored label index for each input pixel
        segmentBits = Array(DIM_IMG_SIZE_X) { IntArray(DIM_IMG_SIZE_X) }
        // stores the colors mapped to each label
        segmentColors = IntArray(OUTPUT_LABELS)
        for (i in 0 until OUTPUT_LABELS) {
            when (i) {
                0 -> segmentColors[i] = Color.TRANSPARENT
                15 -> // person: Red
                    segmentColors[i] = Color.argb(120, 255, 0, 0)
                8 -> // cat: Blue
                    segmentColors[i] = Color.argb(120, 0, 0, 255)
                else -> // other: random color
                    segmentColors[i] = Color.argb(
                        120,
                        (255 * random.nextFloat()).toInt(),
                        (255 * random.nextFloat()).toInt(),
                        (255 * random.nextFloat()).toInt()
                    )
            }
        }
    }

    fun resultsObserver(): Flowable<SegmentationResults> = resultNotifier.toFlowable(BackpressureStrategy.LATEST)

    override fun analyze(image: ImageProxy?, rotationDegrees: Int) {
        try {
            image?.let {
                val ogWidth = image.width
                val ogHeight = image.height

                // transform camera input into 257x257 bitmap
                val imgJpg = NV21toJPEG(
                    ImageUtils.YUV420toNV21(image.image),
                    image.width,
                    image.height
                )
                val inputBitmap = tfResizeBilinear(
                    BitmapFactory.decodeByteArray(imgJpg, 0, imgJpg.size),
                    DIM_IMG_SIZE_X,
                    DIM_IMG_SIZE_Y,
                    rotationDegrees
                )

                if (inputBitmap == null) {
                    Log.e(TAG, "Input bitmap is null")
                    return
                }

                val w = inputBitmap.width
                val h = inputBitmap.height
                if (w > DIM_IMG_SIZE_X || h > DIM_IMG_SIZE_Y) {
                    Log.e(TAG, String.format("invalid bitmap size: %d x %d [should be: %d x %d]", w, h, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y))
                }

                output.rewind()
                imgData.rewind()

                // normalize input from pixels to 0-1 float values
                val inputPixelValues = IntArray(w * h)
                inputBitmap.getPixels(inputPixelValues, 0, w, 0, 0, w, h)
                var pixel = 0
                for (i in 0 until DIM_IMG_SIZE_X) {
                    for (j in 0 until DIM_IMG_SIZE_Y) {
                        if (pixel >= inputPixelValues.size) {
                            break
                        }
                        val `val` = inputPixelValues[pixel++]
                        imgData.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        imgData.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                        imgData.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    }
                }
                inputBitmap.recycle()

                // run inference
                val time = measureTimeMillis {
                    interpreter.run(imgData, output)
                }
                Log.i(TAG, "Segmentation maps calculated in $time ms")

                // transform output back into a bitmap that we can visualize
                val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                fillZeroes(segmentBits)
                val stringBuffer = StringBuffer()
                var maxScore = 0f
                var score: Float
                val seenLabels = mutableMapOf<Int, Int>()
                for (y in 0 until h) {
                    for (x in 0 until w) {
                        segmentBits[x][y] = 0
                        // find the highest output value from all the labels
                        for (c in 0 until OUTPUT_LABELS) {
                            score = output.getFloat((y * w * OUTPUT_LABELS + x * OUTPUT_LABELS + c) * BYTES_PER_POINT)
                            if (c == 0 || score > maxScore) {
                                maxScore = score
                                segmentBits[x][y] = c
                            }
                        }
                        // keep track of all seen labels, counting how many pixels they cover
                        val labelIndex = segmentBits[x][y]
                        if (labelIndex != 0) {
                            val labelPixelCount = seenLabels[labelIndex] ?: 0
                            seenLabels[labelIndex] = labelPixelCount.inc()
                        }
                        // finally, get the color value for that label and set it in the mask bitmap
                        val pixelColor = segmentColors[labelIndex]
                        maskBitmap.setPixel(x, y, pixelColor)
                    }
                }

                // create ordered list of seen labels as comma separated string
                stringBuffer.append(
                    seenLabels.entries
                        .asSequence()
                        .filter { it.value > 20 }
                        .sortedByDescending { it.value }
                        .map { labelMap[it.key] ?: it.key }
                        .joinToString(", ")
                )

                // finally, notify results
                resultNotifier.onNext(SegmentationResults(tfResizeBilinear(maskBitmap, ogHeight, ogWidth, 0), stringBuffer.toString()))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Analyzer Error", e)
        }
    }

    private fun fillZeroes(array: Array<IntArray>?) {
        if (array == null) {
            return
        }
        var r = 0
        while (r < array.size) {
            Arrays.fill(array[r], 0)
            r++
        }
    }

    private fun tfResizeBilinear(bitmap: Bitmap?, w: Int, h: Int, rotationDegrees: Int): Bitmap? {
        if (bitmap == null) {
            return null
        }
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        val resized = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        val canvas = Canvas(resized)
        canvas.drawBitmap(
            rotated,
            Rect(0, 0, rotated.width, rotated.height),
            Rect(0, 0, w, h),
            null
        )
        rotated.recycle()
        return resized
    }


    private fun NV21toJPEG(nv21: ByteArray, width: Int, height: Int): ByteArray {
        val out = ByteArrayOutputStream()
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        yuv.compressToJpeg(Rect(0, 0, width, height), 100, out)
        return out.toByteArray()
    }
}
