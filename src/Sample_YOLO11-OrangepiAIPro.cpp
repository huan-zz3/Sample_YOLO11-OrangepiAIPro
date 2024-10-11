#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include "label.h"

using namespace std;
using namespace cv;
typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct BoundBox {
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex; // 类别索引
    size_t index; // 框索引
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

class SampleYOLOV8 {
    public:
    SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput>& inferOutputs);
    Result GetResult(std::vector<InferenceOutput>& inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleYOLOV8();
    private:
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
};

SampleYOLOV8::SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight) :
                           modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight)
{
}

SampleYOLOV8::~SampleYOLOV8()
{
    ReleaseResource();
}

Result SampleYOLOV8::InitResource()
{
    // init acl resource
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8::ProcessInput(string testImgPath)
{
    // read image from file
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }

    // copy image from host to dvpp
    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // image decoded from JPEG format to YUV
    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // zoom image to modelWidth_ * modelHeight_
    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8::Inference(std::vector<InferenceOutput>& inferOutputs)
{
    // create input data set of model
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

// 函数 SampleYOLOV8::GetResult 的定义
Result SampleYOLOV8::GetResult(std::vector<InferenceOutput>& inferOutputs,
                               string imagePath, size_t imageIndex, bool release)
{
    const uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    // 置信度阈值
    const float confidenceThreshold = 0.25;

    // 类别数量
    const size_t classNum = 80;

    // 每个框包含 (x, y, 宽, 高) 的偏移量
    const size_t offset = 4;

    // 总数量 = 类别数量 + (x, y, 宽, 高)
    const size_t totalNumber = classNum + offset;

    // 模型输出的总框数
    const size_t modelOutputBoxNum = 8400;

    // 起始索引对应 (x, y, 宽, 高)，之后的对象置信度
    const size_t startIndex = 4;

    // 从文件读取源图像
    const cv::Mat srcImage = cv::imread(imagePath);
    const int srcWidth = srcImage.cols;
    const int srcHeight = srcImage.rows;


    vector <BoundBox> boxes;// 存放所有框
    const size_t yIndex = 1;
    const size_t widthIndex = 2;
    const size_t heightIndex = 3;
    // const size_t classConfidenceIndex = 4;


    ACLLITE_LOG_INFO("根据置信度阈值过滤框...");
    // 根据置信度阈值过滤框
    for(size_t i = 0; i < modelOutputBoxNum; ++i)
    {
        float maxValue = 0;
        size_t maxIndex = 0;
        size_t _index = 0;
        for(size_t j = i + modelOutputBoxNum * 4; j < modelOutputBoxNum * totalNumber; j += modelOutputBoxNum)
        {
            float value = classBuff[j];
            if (value > maxValue)
            {
                maxIndex = _index;
                maxValue = value;
            }
            _index++;
        }
        // ACLLITE_LOG_INFO("i:%u, [x]:%f, [y]:%f, [w]:%f, [h]:%f, [maxIndex]:%u, [maxValue]:%f ...",
        //         i, classBuff[i], classBuff[i +modelOutputBoxNum], classBuff[i + modelOutputBoxNum *2], classBuff[i + modelOutputBoxNum *3], maxIndex, maxValue);

        if(maxValue >= confidenceThreshold)
        {
            BoundBox box;
            box.width = classBuff[i + modelOutputBoxNum * 2];
            box.height = classBuff[i + modelOutputBoxNum * 3];
            box.x = classBuff[i];
            box.y = classBuff[i + modelOutputBoxNum];
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;
            if(maxIndex < classNum)
            {
                boxes.push_back(box);
                // ACLLITE_LOG_INFO("i:%u, [x]:%f, [y]:%f, [w]:%f, [h]:%f, [classIndex]:%u, [score]:%f ...",
                // i, classBuff[i], classBuff[i +modelOutputBoxNum], classBuff[i + modelOutputBoxNum *2], classBuff[i + modelOutputBoxNum *3], maxIndex, maxValue);
            }
        }
    }


    ACLLITE_LOG_INFO("根据非极大值抑制(NMS)过滤框...");
    // 根据非极大值抑制(NMS)过滤框
    // 排序：使用 std::sort 对 boxes 向量中的 BoundBox 对象按置信度降序排序。
    // 初始化：创建两个 BoundBox 变量 boxMax 和 boxCompare 来存储当前最高置信度框和其他框的信息。
    // 循环处理：
    //      从 boxes 中取出第一个元素（即当前置信度最高的框）添加到 result 中。
    //      遍历剩余的框，计算每个框与 boxMax 的IoU。
    //      如果IoU超过阈值，则从 boxes 中移除该框，因为它是冗余的。
    //      如果IoU低于阈值，则继续检查下一个框。
    // 移除已处理框：在每次迭代结束时，移除已经处理过的最高置信度框。
    // 坐标平移：通过将框的坐标平移 maxLength * classIndex 来避免不同类别的框之间的误判，这在实际应用中可能不是必需的，除非有特殊需求。
    vector <BoundBox> result;
    result.clear();// 清空结果
    const float NMSThreshold = 0.45;// 阈值
    const int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;// 宽和高中较大的值
    std::sort(boxes.begin(), boxes.end(), sortScore);// 按置信度排序
    BoundBox boxMax;
    BoundBox boxCompare;

    while (boxes.size() != 0) { // boxes 列表不为空
        size_t index = 1;// 索引
        result.push_back(boxes[0]);// 将当前置信度最高的框添加到结果中
        while (boxes.size() > index) {
            boxMax.score = boxes[0].score;
            boxMax.classIndex = boxes[0].classIndex;
            boxMax.index = boxes[0].index;

            // 将点平移 maxLength * boxes[0].classIndex，避免不同类别框相撞
            boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
            boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
            boxMax.width = boxes[0].width;
            boxMax.height = boxes[0].height;

            boxCompare.score = boxes[index].score;
            boxCompare.classIndex = boxes[index].classIndex;
            boxCompare.index = boxes[index].index;

            // 将点平移 maxLength * boxes[0].classIndex，避免不同类别框相撞
            boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
            boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
            boxCompare.width = boxes[index].width;
            boxCompare.height = boxes[index].height;

            // 计算两个框的交集区域
            const float xLeft = max(boxMax.x, boxCompare.x);
            const float yTop = max(boxMax.y, boxCompare.y);
            const float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            const float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);

            // 计算交集面积
            const float width = max(0.0f, xRight - xLeft);
            const float hight = max(0.0f, yBottom - yTop);
            const float area = width * hight;

            // 计算交集面积与两个框面积之和（并集）的比值
            const float iou =  area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

            // 根据NMS阈值过滤框
            if (iou > NMSThreshold) {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }

    ACLLITE_LOG_INFO("设置 OpenCV 绘制标签参数...");
    // 设置 OpenCV 绘制标签参数
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255);
    const vector <cv::Scalar> colors{
        cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255),
        cv::Scalar(50, 205, 50), cv::Scalar(139, 85, 26)};
    const float xRate = static_cast<float>(srcWidth) / static_cast<float>(modelWidth_);
    const float yRate = static_cast<float>(srcHeight) / static_cast<float>(modelHeight_);

    ACLLITE_LOG_INFO("结果大小: %zu", result.size());
    // 使用 OpenCV 在图像上绘制框和标签
    int half = 2;
    for (size_t i = 0; i < result.size(); ++i) {
        ACLLITE_LOG_INFO("处理第 %zu 个框", i);
        cv::Point leftUpPoint, rightBottomPoint; // 左上角和右下角坐标
        leftUpPoint.x = (result[i].x - result[i].width / half) * xRate; // 左上角x坐标
        leftUpPoint.y = (result[i].y - result[i].height / half) * yRate; // 左上角y坐标
        rightBottomPoint.x = (result[i].x + result[i].width / half) * xRate; // 右下角x坐标
        rightBottomPoint.y = (result[i].y + result[i].height / half) * yRate; // 右下角y坐标
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid); // 绘制框
        string className = label[result[i].classIndex];// 获取类别名称
        string markString = to_string(result[i].score) + ":" + className;// 构造置信度+类名
        cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset), // 绘制标签
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }
    string savePath = "out_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(savePath, srcImage);
    if (release){
        free(classBuff);
        classBuff = nullptr;
    }
    return SUCCESS;
}


void SampleYOLOV8::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main()
{
    // const char* modelPath = "../model/yolov8n_2.om";
    const char* modelPath = "../model/yolo11n.om";
    const string imagePath = "../data";
    const int32_t modelWidth = 640;
    const int32_t modelHeight = 640;

    // all images in dir
    DIR *dir = opendir(imagePath.c_str());
    if (dir == nullptr)
    {
        ACLLITE_LOG_ERROR("file folder does no exist, please create folder %s", imagePath.c_str());
        return FAILED;
    }
    vector<string> allPath;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0
        || strcmp(entry->d_name, ".keep") == 0)
        {
            continue;
        }else{
            string name = entry->d_name;
            string imgDir = imagePath +"/"+ name;
            allPath.push_back(imgDir);
        }
    }
    closedir(dir);

    if (allPath.size() == 0){
        ACLLITE_LOG_ERROR("the directory is empty, please download image to %s", imagePath.c_str());
        return FAILED;
    }

    // inference
    string fileName;
    bool release = false;
    SampleYOLOV8 sampleYOLO(modelPath, modelWidth, modelHeight);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    for (size_t i = 0; i < allPath.size(); i++)
    {
        if (allPath.size() == i){
            release = true;
        }
        std::vector<InferenceOutput> inferOutputs;
        fileName = allPath.at(i).c_str();



        ret = sampleYOLO.ProcessInput(fileName);
        if (ret == FAILED) {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            return FAILED;
        }
        ACLLITE_LOG_INFO("ProcessInput OK");

        ret = sampleYOLO.Inference(inferOutputs);
        if (ret == FAILED) {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            return FAILED;
        }
        ACLLITE_LOG_INFO("Inference OK");

        ret = sampleYOLO.GetResult(inferOutputs, fileName, i, release);
        if (ret == FAILED) {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            return FAILED;
        }
        ACLLITE_LOG_INFO("GetResult OK");
    }
    return SUCCESS;
}