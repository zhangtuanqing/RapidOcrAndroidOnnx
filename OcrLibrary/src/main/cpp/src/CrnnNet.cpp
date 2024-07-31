#include "CrnnNet.h"
#include "OcrUtils.h"
#include <numeric>

CrnnNet::CrnnNet() {}

CrnnNet::~CrnnNet() {
    delete session;
    inputNamesPtr.clear();
    outputNamesPtr.clear();
}

void CrnnNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    //===session options===
    // Sets the number of threads used to parallelize the execution within nodes
    // A value of 0 means ORT will pick a default
    sessionOptions.SetIntraOpNumThreads(numThread);
    //set OMP_NUM_THREADS=16

    // Sets the number of threads used to parallelize the execution of the graph (across nodes)
    // If sequential execution is enabled this value is ignored
    // A value of 0 means ORT will pick a default
    sessionOptions.SetInterOpNumThreads(numThread);

    // Sets graph optimization level
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

char *readKeysFromAssets(AAssetManager *mgr, const std::string &keysName) {
    //LOGI("readKeysFromAssets start...");
    if (mgr == NULL) {
        LOGE(" %s", "AAssetManager==NULL");
        return NULL;
    }
    char *buffer;
    /*获取文件名并打开*/
    AAsset *asset = AAssetManager_open(mgr, keysName.c_str(), AASSET_MODE_UNKNOWN);
    if (asset == NULL) {
        LOGE(" %s", "asset==NULL");
        return NULL;
    }
    /*获取文件大小*/
    off_t bufferSize = AAsset_getLength(asset);
    //LOGI("file size : %d", bufferSize);
    buffer = (char *) malloc(bufferSize + 1);
    buffer[bufferSize] = 0;
    int numBytesRead = AAsset_read(asset, buffer, bufferSize);
    //LOGI("readKeysFromAssets: %d", numBytesRead);
    /*关闭文件*/
    AAsset_close(asset);
    //LOGI("readKeysFromAssets exit...");
    return buffer;
}

void CrnnNet::initModel(AAssetManager *mgr, const std::string &name, const std::string &keysName) {
    int dbModelDataLength = 0;
    void *dbModelData = getModelDataFromAssets(mgr, name.c_str(), dbModelDataLength);
    session = new Ort::Session(ortEnv, dbModelData, dbModelDataLength,
                               sessionOptions);
    free(dbModelData);
    inputNamesPtr = getInputNames(session);
    outputNamesPtr = getOutputNames(session);

    //load keys
    char *buffer = readKeysFromAssets(mgr, keysName);
    if (buffer != NULL) {
        std::istringstream inStr(buffer);
        std::string line;
        while (getline(inStr, line)) {
            keys.emplace_back(line);
        }
        free(buffer);
    } else {
        LOGE(" txt file not found");
        return;
    }
    keys.insert(keys.begin(),
                "#"); // blank char for ctc
    keys.emplace_back(" ");
    LOGI("keys size(%d)", keys.size());
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

TextLine CrnnNet::scoreToTextLine(const std::vector<float> &outputData, int h, int w) {
    auto keySize = keys.size();
    auto dataSize = outputData.size();
    std::string strRes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> columns;
    int lastIndex = 0;
    int maxIndex;
    float maxValue;

    auto emptyCount = 0;
    auto lastEmptyCount = 0;
    auto curCharNum = 0;
    for (int i = 0; i < h; i++) {
        int start = i * w;
        int stop = (i + 1) * w;
        if (stop > dataSize - 1) {
            stop = (i + 1) * w - 1;
        }
        maxIndex = int(argmax(&outputData[start], &outputData[stop]));
        maxValue = float(*std::max_element(&outputData[start], &outputData[stop]));

        if (maxIndex == 0) {
            emptyCount++;
        }

        if (maxIndex > 0 && maxIndex == lastIndex) {
            emptyCount++;
        }
        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            if (scores.size() <= 0) {
                int count = 1 + emptyCount;
                columns.emplace_back(count);
            } else {
                int count = columns.at(curCharNum - 1) + emptyCount / 2;
                columns.at(curCharNum - 1) = count;
                count = emptyCount / 2 + 1;
                columns.emplace_back(count);
            }
            scores.emplace_back(maxValue);
            indices.emplace_back(i + 1);
            strRes.append(keys[maxIndex]);
            curCharNum++;
            emptyCount = 0;
        }
        if (i == h - 1 && columns.size() >= 1) {
            columns.at(columns.size() - 1) += emptyCount;
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores, indices, columns, h};
}

TextLine CrnnNet::getTextLine(cv::Mat &src) {
    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);

    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(dstWidth, dstHeight));

    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues,
                                                                  normValues);

    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(),
                                                             inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());
    std::vector<const char *> inputNames = {inputNamesPtr.data()->get()};
    std::vector<const char *> outputNames = {outputNamesPtr.data()->get()};
    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                     inputNames.size(), outputNames.data(), outputNames.size());

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                          std::multiplies<int64_t>());

    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    std::vector<float> outputData(floatArray, floatArray + outputCount);
    return scoreToTextLine(outputData, outputShape[1], outputShape[2]);
}

std::vector<TextLine> CrnnNet::getTextLines(std::vector<cv::Mat> &partImg) {
    int size = partImg.size();
    std::vector<TextLine> textLines(size);
    for (int i = 0; i < size; ++i) {
        //getTextLine
        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        double endCrnnTime = getCurrentTime();
        textLine.time = endCrnnTime - startCrnnTime;
        textLines[i] = textLine;
    }
    return textLines;
}