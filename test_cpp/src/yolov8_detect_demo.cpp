#include "chrono"
#include "vector"
#include "iostream"
#include "rknn_api.h"
#include "opencv2/opencv.hpp"

// 模型文件名
static const char *MODEL_FILE = "./yolov8n.rknn";
// 模型输入宽
static int MODEL_WIDTH;
// 模型输入高
static int MODEL_HEIGHT;

static int class_num = 80;
static float conf = 0.4;
static float nms = 0.5;


struct FileData{
	size_t size;
	unsigned char *data;
};

struct box_info
{
	int x1, y1, x2, y2;
	float score;
	int lable_id;
};


// 读取模型文件数据
static void load_file(const char *filename, FileData &content)
{
	FILE *fp = fopen(filename, "rb");
	if (fp == nullptr) {
		printf("fopen %s fail!\n", filename);
		return;
	}
	fseek(fp, 0, SEEK_END);
	int model_len = ftell(fp);
	content.data = (unsigned char*)malloc(model_len);
	fseek(fp, 0, SEEK_SET);
	if (model_len != fread(content.data, 1, model_len, fp)) {
		printf("fread %s fail!\n", filename);
		free(content.data);
		return;
	}
	content.size = model_len;
	if (fp) {
		fclose(fp);
	}
}

// 预处理
static void preprocessing(cv::Mat origin, cv::Mat &output, float &radio_x, float &radio_y) {
	// 计算形变比例， input.w / model.w input.h / model.h
	radio_x = origin.cols* 1.0f / MODEL_WIDTH;
	radio_y = origin.rows* 1.0f / MODEL_HEIGHT;
	
	cv::resize(origin, output, cv::Size(MODEL_WIDTH, MODEL_HEIGHT), radio_x, radio_y);
	cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
	
	return ;
}

// 检测后处理
static std::vector<box_info> decode_infer(float *cls_output, float *boxes_output) {
	std::vector<int> strides = {8, 16, 32};

	std::vector<box_info> result;
	int skew = 0;
	int skew_last = 0;
	for (auto stride: strides) {
		int dh = MODEL_HEIGHT / stride;
		int dw = MODEL_WIDTH / stride;
		skew += dh * dw;
		for(int i=skew_last; i< skew; i ++) {
			for (int c=0; c < class_num; c++) {
				float c_score = cls_output[i * class_num + c];
				if (c_score >= conf) {
					box_info box;
					box.lable_id = c;
					box.score = c_score;
					box.x1 = boxes_output[i*4] * stride;
					box.y1 = boxes_output[i*4 + 1] * stride;
					box.x2 = boxes_output[i*4 + 2] * stride;
					box.y2 = boxes_output[i*4 + 3] * stride;
					result.push_back(box);
				}
			}
		}
		skew_last += dh * dw;
	}

	return result;
}


inline bool cmp(box_info a, box_info b) {
	if (a.score > b.score)
		return true;
	return false;
}

static void NMS(std::vector<box_info> &input_boxes) {
    std::sort(input_boxes.begin(), input_boxes.end(), cmp);
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

int main(int argc, char const *argv[])
{
	rknn_context ctx;
	int model_size[2];   // width height
	rknn_input_output_num io_num;
	rknn_tensor_attr *output_attrs;
	rknn_output *outputs_tensor;

    int ret;
    FileData content;
    load_file(MODEL_FILE, content);
    ret = rknn_init(&(ctx), (unsigned char*)content.data, content.size, 0, NULL);
    rknn_core_mask mode = RKNN_NPU_CORE_AUTO;
    mode = RKNN_NPU_CORE_AUTO;
    ret = rknn_set_core_mask(ctx, mode);
    if(ret != RKNN_SUCC)
    {
        printf("[ W Inference create error: rknn_init ]");
    }

	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC)
	{
		printf("[ W Inference create error: rknn_query ]");
	}

	rknn_tensor_attr input_attrs;
	input_attrs.index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs, sizeof(rknn_tensor_attr));

	// 打印输入shape
    std::cout << "[ W Model Shape: " << input_attrs.dims[3] << " " << input_attrs.dims[1] << " " <<input_attrs.dims[2] << " ]" << std::endl;
	MODEL_WIDTH = input_attrs.dims[1];
	MODEL_HEIGHT = input_attrs.dims[2];

	// 空间预分配
	output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr)*(io_num.n_output));
	memset(output_attrs, 0, sizeof(rknn_tensor_attr)*(io_num.n_output));
	outputs_tensor = (rknn_output*)malloc(sizeof(rknn_output)*(io_num.n_output));
	for (int i = 0; i < io_num.n_output; i++)
	{
		output_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));

		// 手动指定输出空间
		(outputs_tensor)[i].want_float = 1;
		(outputs_tensor)[i].is_prealloc = 1;
		(outputs_tensor)[i].buf = malloc(sizeof(float) * output_attrs[i].size);
		if (ret != RKNN_SUCC)
		{
			printf("[ W Inference create error: rknn_query ]");
		}
		// 打印输出shape
			std::cout << "Model OutShape " << i << ": ";
			for(size_t index=0; index < output_attrs[i].n_dims; index ++)
			{
				std::cout << output_attrs[i].dims[index] << " ";
			}
			std::cout << std::endl;
	}

	// 读图
	cv::Mat img = cv::imread("./zidane.jpg");
	// 预处理 (为了快速实现,直接resize，不做填充保持比例)
	cv::Mat input_image;
	float radio_x, radio_y;
	preprocessing(img, input_image, radio_x, radio_y);

	ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs, sizeof(rknn_tensor_attr));
	rknn_input inputs[io_num.n_input];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = MODEL_HEIGHT * MODEL_WIDTH * 3;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].buf = input_image.data;
	
	ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
	if (ret != RKNN_SUCC)
	{
		printf("[ W Inference forward error: rknn_inputs_set ]");
	}

    // while (1) 
    {
		auto start = std::chrono::high_resolution_clock::now();
		ret = rknn_run(ctx, nullptr);
		if (ret != RKNN_SUCC)
		{
			printf("[ W Inference forward error: rknn_run ]");
		}
		ret = rknn_outputs_get(ctx, io_num.n_output, outputs_tensor, NULL);
		if (ret != RKNN_SUCC)
		{
			printf("[ W Inference forward error: rknn_outputs_get ]");
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "forwart time: " << duration.count() * 1.0 /1000 << " ms" << std::endl;
	}

	float *cls_output = (float *) outputs_tensor[1].buf;
	float *boxes_output = (float *) outputs_tensor[0].buf;

	std::vector<box_info> result = decode_infer(cls_output, boxes_output);
	NMS(result);
	
	// 复原到原图上
	for(auto &ob: result) {
		ob.x1 *= radio_x; ob.x2 *= radio_x;
		ob.y1 *= radio_y; ob.y2 *= radio_y;
	}

	std::cout << "result size: " << result.size() << std::endl;
	for(auto ob: result) {
		std::cout << "lable id: " << ob.lable_id << " score: " << ob.score  <<  std::endl;
		std::cout << "wywh : " << ob.x1 << " " << ob.y1 << " " << ob.x2 << " " << ob.y2 << std::endl;
		std::cout << "========================================== " << std::endl;
		cv::rectangle(img, cv::Point(ob.x1, ob.y1), cv::Point(ob.x2, ob.y2), cv::Scalar(0, 0, 255), 3);
	}
	cv::imwrite("res.jpg", img);
    return 0;
}
