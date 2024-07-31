#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "OcrLite.h"
#include "OcrUtils.h"
#include <opencv2/imgproc.hpp>
#include <codecvt>
#include <locale>

OcrLite::OcrLite() {}

OcrLite::~OcrLite() {}

void OcrLite::init(JNIEnv *jniEnv, jobject assetManager, int numThread, std::string detName,
                   std::string clsName, std::string recName, std::string keysName) {
    AAssetManager *mgr = AAssetManager_fromJava(jniEnv, assetManager);
    if (mgr == NULL) {
        LOGE(" %s", "AAssetManager==NULL");
    }

    Logger("--- Init DbNet ---\n");
    dbNet.setNumThread(numThread);
    dbNet.initModel(mgr, detName);

    Logger("--- Init AngleNet ---\n");
    angleNet.setNumThread(numThread);
    angleNet.initModel(mgr, clsName);

    Logger("--- Init CrnnNet ---\n");
    crnnNet.setNumThread(numThread);
    crnnNet.initModel(mgr, recName, keysName);

    LOGI("初始化完成!");
}

/*void OcrLite::initLogger(bool isDebug) {
    isLOG = isDebug;
}

void OcrLite::Logger(const char *format, ...) {
    if (!isLOG) return;
    char *buffer = (char *) malloc(8192);
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
    if (isLOG) LOGI("%s", buffer);
    free(buffer);
}*/

std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes) {
    std::vector<cv::Mat> partImages;
    for (int i = 0; i < textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
        partImages.emplace_back(partImg);
    }
    return partImages;
}

std::wstring convertStringToWString(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(str);
}

bool textBoxCompare(const TextBox &a, const TextBox &b) {
    auto p0 = a.boxPoint[0];
    auto p1 = b.boxPoint[0];
    if (p0.x == p1.x) {
        return p0.y < p1.y;
    } else if (p0.y == p1.y) {
        return p0.x < p1.x;
    } else {
        return p0.x < p1.x;
    }
}

bool pointYCompare(const cv::Point &a, const cv::Point  &b) {
    return a.y < b.y;
}

bool pointXCompare(const cv::Point &a, const cv::Point  &b) {
    return a.x < b.x;
}

OcrResult OcrLite::detect(cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                          float boxScoreThresh, float boxThresh,
                          float unClipRatio, bool doAngle, bool mostAngle) {

    cv::Mat textBoxPaddingImg = src.clone();
    int thickness = getThickness(src);

    Logger("=====Start detect=====");
    Logger("ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)", scale.srcWidth, scale.srcHeight,
           scale.dstWidth, scale.dstHeight,
           scale.ratioWidth, scale.ratioHeight);

    Logger("---------- step: dbNet getTextBoxes ----------");
    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes = dbNet.getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
    Logger("TextBoxesSize(%ld)", textBoxes.size());
    double endDbNetTime = getCurrentTime();
    double dbNetTime = endDbNetTime - startTime;
    Logger("dbNetTime(%fms)", dbNetTime);

    for (int i = 0; i < textBoxes.size(); ++i) {
        Logger("TextBox[%d][score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]",
               i,
               textBoxes[i].score,
               textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[0].y,
               textBoxes[i].boxPoint[1].x, textBoxes[i].boxPoint[1].y,
               textBoxes[i].boxPoint[2].x, textBoxes[i].boxPoint[2].y,
               textBoxes[i].boxPoint[3].x, textBoxes[i].boxPoint[3].y);
    }

    std::sort(textBoxes.begin(), textBoxes.end(), textBoxCompare);

    Logger("---------- step: drawTextBoxes ----------");
    drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

    //---------- getPartImages ----------
    std::vector<cv::Mat> partImages = getPartImages(src, textBoxes);

    Logger("---------- step: angleNet getAngles ----------");
    std::vector<Angle> angles;
    angles = angleNet.getAngles(partImages, doAngle, mostAngle);

    //Log Angles
    for (int i = 0; i < angles.size(); ++i) {
        Logger("angle[%d][index(%d), score(%f), time(%fms)]", i, angles[i].index, angles[i].score, angles[i].time);
    }

    //Rotate partImgs
    for (int i = 0; i < partImages.size(); ++i) {
        if (angles[i].index == 1) {
            partImages.at(i) = matRotateClockWise180(partImages[i]);
        }
    }

    Logger("---------- step: crnnNet getTextLine ----------");
    std::vector<TextLine> textLines = crnnNet.getTextLines(partImages);
    //Log TextLines
    for (int i = 0; i < textLines.size(); ++i) {
        Logger("textLine[%d](%s)", i, textLines[i].text.c_str());
        std::ostringstream txtScores;
        for (int s = 0; s < textLines[i].charScores.size(); ++s) {
            if (s == 0) {
                txtScores << textLines[i].charScores[s];
            } else {
                txtScores << " ," << textLines[i].charScores[s];
            }
        }
        Logger("textScores[%d]{%s}", i, std::string(txtScores.str()).c_str());
        Logger("crnnTime[%d](%fms)", i, textLines[i].time);
    }

    std::vector<TextBlock> textBlocks;
    for (int i = 0; i < textLines.size(); ++i) {
        std::vector<cv::Point> boxPoint = std::vector<cv::Point>(4);
        int padding = originRect.x;//padding conversion
        boxPoint[0] = cv::Point(textBoxes[i].boxPoint[0].x - padding, textBoxes[i].boxPoint[0].y - padding);
        boxPoint[1] = cv::Point(textBoxes[i].boxPoint[1].x - padding, textBoxes[i].boxPoint[1].y - padding);
        boxPoint[2] = cv::Point(textBoxes[i].boxPoint[2].x - padding, textBoxes[i].boxPoint[2].y - padding);
        boxPoint[3] = cv::Point(textBoxes[i].boxPoint[3].x - padding, textBoxes[i].boxPoint[3].y - padding);
        TextBlock textBlock{boxPoint, textBoxes[i].score, angles[i].index, angles[i].score,
                            angles[i].time, textLines[i].text, textLines[i].charScores, textLines[i].time,
                            angles[i].time + textLines[i].time};


        textBlock.charPoints = std::vector<cv::Point>();
        auto minXPoint = std::min_element(&(textBoxes[i].boxPoint[0]), &(textBoxes[i].boxPoint[3]), pointXCompare);
        auto maxXPoint = std::max_element(&(textBoxes[i].boxPoint[0]), &(textBoxes[i].boxPoint[3]), pointXCompare);
        auto minX = minXPoint->x;
        auto maxX = maxXPoint->x;

        TextLine curTextLine = textLines.at(i);
        auto wString = convertStringToWString(curTextLine.text);

        std::vector<int> colIndices = curTextLine.charColIndex;
        std::vector<int> columns = curTextLine.charColumNum;
        int totalColumns = curTextLine.colCount;
        float boxH = boxPoint[3].y - boxPoint[0].y;
        float longSideLen = std::sqrt(std::pow(boxPoint[1].x - boxPoint[0].x, 2) + std::pow(boxPoint[1].y - boxPoint[0].y, 2));
        float xLength = std::abs(maxXPoint->x - minXPoint->x);
        float yLength = std::abs(boxPoint[1].y - boxPoint[0].y);
        float angleCos = xLength / longSideLen;
        float angleSin = yLength / longSideLen;
        auto cellWidth = longSideLen / totalColumns;
        auto factor = 1.0F;
        if (textBoxes[i].boxPoint[1].y >= textBoxes[i].boxPoint[0].y) {
            factor = 1.0F;
        } else {
            factor = -1.0F;
        }
        for (auto k = 0; k < colIndices.size(); k++) {
            int curColIndex = colIndices.at(k);

            int columnStart = curColIndex - columns.at(k) / 2;
            int columnEnd = curColIndex + columns.at(k) / 2;

            auto x0 = textBoxes[i].boxPoint[0].x + columnStart * cellWidth * angleCos;
            auto y0 = textBoxes[i].boxPoint[0].y + factor * columnStart * cellWidth * angleSin;

            auto x1 = textBoxes[i].boxPoint[0].x + columnEnd * cellWidth * angleCos;
            auto y1 = textBoxes[i].boxPoint[0].y + factor * columnEnd * cellWidth * angleSin;


            auto x2 = textBoxes[i].boxPoint[3].x + columnEnd * cellWidth * angleCos;
            auto y2 = textBoxes[i].boxPoint[3].y + factor * columnEnd * cellWidth * angleSin;

            auto x3  = textBoxes[i].boxPoint[3].x + columnStart * cellWidth * angleCos;
            auto y3 = textBoxes[i].boxPoint[3].y + factor * columnStart * cellWidth * angleSin;

            auto charBox = std::vector<cv::Point>(4);
            charBox[0] = cv::Point(x0, y0);
            charBox[1] = cv::Point(x1, y1);
            charBox[2] = cv::Point(x2, y2);
            charBox[3] = cv::Point(x3, y3);
            auto color = cv::Scalar(255, 0, 0);// R(255) G(0) B(0)

//            drawTextBoxBlue(textBoxPaddingImg, charBox, 1);

            float xPos = textBoxes[i].boxPoint[0].x + (curColIndex * longSideLen / totalColumns) * angleCos;
            float yPos = (textBoxes[i].boxPoint[0].y + boxH / 2) + factor * (curColIndex * longSideLen / totalColumns) * angleSin;
            cv::drawMarker(textBoxPaddingImg,  cv::Point(xPos, yPos), color, cv::MarkerTypes::MARKER_CROSS, 8, 1);

            for (auto m = 0; m < 4; m++) {
                textBlock.charPoints.emplace_back(cv::Point(charBox.at(m).x - padding, charBox.at(m).y - padding));
            }
        }
        auto minSideLen = 0.0f;
        textBlock.boxBoundingPoint = getMinBoxes(boxPoint);

        textBlocks.emplace_back(textBlock);
    }

    double endTime = getCurrentTime();
    double fullTime = endTime - startTime;
    Logger("=====End detect=====");
    Logger("FullDetectTime(%fms)", fullTime);

    //cropped to original size
    cv::Mat textBoxImg;
    if (originRect.x > 0 && originRect.y > 0) {
        textBoxPaddingImg(originRect).copyTo(textBoxImg);
    } else {
        textBoxImg = textBoxPaddingImg;
    }

    std::string strRes;
    for (int i = 0; i < textBlocks.size(); ++i) {
        strRes.append(textBlocks[i].text);
        strRes.append("\n");
    }

    return OcrResult{dbNetTime, textBlocks, textBoxImg, fullTime, strRes};
}
