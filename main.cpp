/*
这份代码是找到了openvino官方写的推理代码内容https://github.com/openvino-book/yolov8_openvino_cpp.git
基础的模型的.xml和.param是未经优化的，后续会再做优化.


文件当中可以修改的参数有以下几个:
1. score_threshold 置信度阈值，程序只会挑选置信度大于score_threshold的预测框，score_threshold越大，则能够计算的预测框就越少，
    剩下的预测框置信度就越高!
2. nms_threshold nms阈值，程序会将不同边界框当中重合度较高的预测框删除，调高nms_threshold会让更多重合度较高的预测框被保留，
    调低nms_threshold，会让重合度较高的预测框被丢弃。
*/

#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file

std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
                                   cv::Scalar(255, 100, 50) , cv::Scalar(50, 100, 255) , cv::Scalar(255, 50, 100) };
const std::vector<std::string> class_names = {"weed" };

using namespace cv;
using namespace dnn;

// Keep the ratio before resize
Mat letterbox(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

int main(int argc, char* argv[])
{
    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    auto compiled_model = core.compile_model("/home/chengzi/Desktop/nanodet/demo_openvino/openvino/model/epoch300.xml", "CPU");

    // -------- Step 3. Create an Inference Request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // -------- Step 4.Read a picture file and do the preprocess --------
    Mat img = cv::imread("/home/chengzi/Desktop/nanodet/demo_ncnn/ncnn/resource/image/bonirob_2016-05-23-10-42-16_1_frame3.png");
    // Preprocess the image
    Mat letterbox_img = letterbox(img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "The shape of output tensor:" << output_shape << std::endl;
    int rows = output_shape[2];        //8400
    int dimensions = output_shape[1];  //84: box[cx, cy, w, h]+80 classes scores
    std::cout<<"rows is "<<rows<<" cols  is "<<dimensions << std::endl;
    // -------- Step 8. Postprocess the result --------
    float* data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,84]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;

    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 5);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold) {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);

            boxes.push_back(Rect(left, top, width, height));
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

    // -------- Visualize the detection results -----------
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        int class_id = class_ids[index];
        rectangle(img, boxes[index], colors[class_id % 6], 2, 8);
        std::cout<< "prediction box["<< i << "]'s box is "<< boxes[index].x<< ", "<< boxes[index].y<<", "<< boxes[index].width<<", "<< boxes[index].height<< std::endl;
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height+5);
        cv::rectangle(img, textBox, colors[class_id % 6], FILLED);
        putText(img, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
    }

    namedWindow("YOLOv8 OpenVINO Inference C++ Demo", WINDOW_AUTOSIZE);
    imshow("YOLOv8 OpenVINO Inference C++ Demo", img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}