package com.benjaminwan.ocrlibrary

import java.io.Serializable

data class ImageOcrResponse(val ocrRes: List<OcrRecognize?>? = null,
                            val ocrRequestId: String? = null,
                            val imageWidth: Int? = null,
                            val imageHeight: Int? = null):Serializable {
    data class OcrRecognize(val lineText: String?,
                            val lineBoundingBox: BoundingBox?,
                            val lineBoundingPolygon: List<Float>?,
                            val charset: List<CharBoundingInfo>?):Serializable {
        data class BoundingBox(val left: Float?, val top: Float?, val width: Float?, val height: Float?):Serializable
        data class CharBoundingInfo(val charBoundingPolygon: List<Float>?, val word: String?):Serializable
    }
}
