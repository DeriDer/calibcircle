#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
using namespace cv;

Mat resize_image(Mat& img)
{
    Mat res;
    if (img.cols < 1200)
        return img;
    // 计算缩小比例
    double scale = 0.5; // 缩小到原来的一半

    // 计算新的宽度和高度
    int newWidth = static_cast<int>(img.cols * scale);
    int newHeight = static_cast<int>(img.rows * scale);

    // 调整图像大小
    cv::Mat resizedImage;
    cv::resize(img, res, cv::Size(newWidth, newHeight));
    return res;
}

int main(int argc, char** argv) {
    // 读取图像
    Mat img = imread(argv[1], -1);

    if (img.empty() || argc != 2) {
        // 图像读取失败
        std::cout << "无法加载图像文件" << std::endl;
        return -1;
    }
    Mat image = resize_image(img);
    // 显示图像
    imshow("Image", image);

    // 等待键盘输入
    waitKey(0);
    std::vector<Point2f> pointBuf;
    std::vector<cv::Point2f> p;
    cv::Size s;
    s.height = 6;
    s.width = 9;
    SimpleBlobDetector::Params params;
    params.thresholdStep = 10;    //二值化的阈值步长，即公式1的t  
    params.minThreshold = 50;   //二值化的起始阈值，即公式1的T1  
    params.maxThreshold = 255;    //二值化的终止阈值，即公式1的T2  
    //重复的最小次数，只有属于灰度图像斑点的那些二值图像斑点数量大于该值时，该灰度图像斑点才被认为是特征点  
    params.minRepeatability = 2;     
    //最小的斑点距离，不同二值图像的斑点间距离小于该值时，被认为是同一个位置的斑点，否则是不同位置上的斑点  
    params.minDistBetweenBlobs = 10;  
  
    params.filterByColor = true;    //斑点颜色的限制变量  
    params.blobColor = 0;    //表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点  
  
    params.filterByArea = true;    //斑点面积的限制变量  
    params.minArea = 25;    //斑点的最小面积  
    params.maxArea = 5000;    //斑点的最大面积  
  
    params.filterByCircularity = false;    //斑点圆度的限制变量，默认是不限制  
    params.minCircularity = 0.8f;    //斑点的最小圆度  
    //斑点的最大圆度，所能表示的float类型的最大值  
    params.maxCircularity = std::numeric_limits<float>::max();  
  
    params.filterByInertia = true;    //斑点惯性率的限制变量  
    //minInertiaRatio = 0.6;  
    params.minInertiaRatio = 0.1f;    //斑点的最小惯性率  
    params.maxInertiaRatio = std::numeric_limits<float>::max();    //斑点的最大惯性率  
  
    params.filterByConvexity = true;    //斑点凸度的限制变量  
    //minConvexity = 0.8;  
    params.minConvexity = 0.95f;    //斑点的最小凸度  
    params.maxConvexity = std::numeric_limits<float>::max();    //斑点的最大凸度  

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    bool found1 = findCirclesGrid(image, s, p, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING, detector);

    //bool found = findCirclesGrid( image, Size(11,4), pointBuf, CALIB_CB_ASYMMETRIC_GRID, detector);
    std::cout << found1 << std::endl;
    std::cout << p << std::endl;
    drawChessboardCorners( image, s, Mat(p), found1 );
    imshow("Image", image);
    imwrite("output.jpg",image);
    waitKey(0);
    return 0;
}
