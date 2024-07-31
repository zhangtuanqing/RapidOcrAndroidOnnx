package com.benjaminwan.ocrlibrary

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap

class OcrEngine(context: Context) {
    companion object {
        const val numThread: Int = 4
    }

    init {
        System.loadLibrary("RapidOcr")
        val ret = init(
            context.assets, numThread,
            "ch_PP-OCRv3_det_infer.onnx",
            "ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "ch_PP-OCRv3_rec_infer.onnx",
            "ppocr_keys_v1.txt"
        )
        if (!ret) throw IllegalArgumentException()
    }

    var padding: Int = 50
    var boxScoreThresh: Float = 0.5f
    var boxThresh: Float = 0.3f
    var unClipRatio: Float = 1.6f
    var doAngle: Boolean = true
    var mostAngle: Boolean = true

    private fun convertNativeOcrResult(srcOcrResult: OcrResult, width: Int, height: Int, requestId: String): ImageOcrResponse? {
        val ocrRes = mutableListOf<ImageOcrResponse.OcrRecognize>()
        srcOcrResult.textBlocks.forEach { textBlock ->
            if (!textBlock.text.isNullOrEmpty()) {
                val textBox = textBlock.boxPoint.flatMap { listOf(it.x.toFloat(), it.y.toFloat()) }
                val textBoundingBox = ImageOcrResponse.OcrRecognize.BoundingBox(
                    left = textBlock.boundingPoint[0].x.toFloat(),
                    top = textBlock.boundingPoint[0].y.toFloat(),
                    width = (textBlock.boundingPoint[1].x - textBlock.boundingPoint[0].x).toFloat(),
                    height = (textBlock.boundingPoint[3].y - textBlock.boundingPoint[0].y).toFloat()
                )

                val charInfo = mutableListOf<ImageOcrResponse.OcrRecognize.CharBoundingInfo>()
                textBlock.charPoint.chunked(4).forEachIndexed() { index, it ->
                    val charBoundingPoint = it.flatMap { listOf(it.x.toFloat(), it.y.toFloat()) }
                    val word = textBlock.text[index]
                    charInfo.add(
                        ImageOcrResponse.OcrRecognize.CharBoundingInfo(
                            charBoundingPoint,
                            word.toString()
                        )
                    )
                }
                ocrRes.add(
                    ImageOcrResponse.OcrRecognize(
                        textBlock.text, textBoundingBox, textBox, charInfo
                    )
                )
            }
        }
        return ImageOcrResponse(ocrRes, requestId, width, height)
    }

    fun detect(input: Bitmap, requestId: String): ImageOcrResponse? {
        val dstBitmap = Bitmap.createBitmap(
            input.width, input.height, Bitmap.Config.ARGB_8888
        )
        var maxSideLen = kotlin.math.max(input.width, input.height)
        if (maxSideLen >= 960) {
            maxSideLen = (maxSideLen * 0.6f).toInt()
        }
        val ocrResult = detect(input, dstBitmap, 0, maxSideLen, 0.35f, 0.85f, 1.5f,
            doAngle = true,
            mostAngle = true
        )
        val ocrResponse = convertNativeOcrResult(ocrResult, input.width, input.height, requestId)
        return ocrResponse
    }

    fun detect(input: Bitmap, output: Bitmap, maxSideLen: Int) =
        detect(
            input, output, padding, maxSideLen,
            boxScoreThresh, boxThresh,
            unClipRatio, doAngle, mostAngle
        )

    external fun init(
        assetManager: AssetManager,
        numThread: Int, detName: String,
        clsName: String, recName: String, keysName: String
    ): Boolean

    external fun detect(
        input: Bitmap, output: Bitmap, padding: Int, maxSideLen: Int,
        boxScoreThresh: Float, boxThresh: Float,
        unClipRatio: Float, doAngle: Boolean, mostAngle: Boolean
    ): OcrResult

    external fun benchmark(input: Bitmap, loop: Int): Double

}