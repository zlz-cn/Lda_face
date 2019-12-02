#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

int g_howManyPhotoForTraining = 5;
//每个人取出8张作为训练
int g_photoNumberOfOnePerson = 10;
//ORL数据库每个人10张图像
using namespace cv;
using namespace std;

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// 创建和返回一个归一化后的图像矩阵:
	Mat dst;
	switch (src.channels()) {
	case1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
//使用CSV文件去读图像和标签，主要使用stringstream和getline方法
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main() {
	string output_folder;
	output_folder = string("data/output_folder");
	string fn_csv = string("data/at.txt");
	vector<Mat> allImages, train_images, test_images;
	vector<int> allLabels, train_labels, test_labels;
	try {
		read_csv(fn_csv, allImages, allLabels);
	}
	catch (cv::Exception & e) {
		cerr << "Error opening file " << fn_csv << ". Reason: " << e.msg << endl;
		exit(1);
	}
	if (allImages.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	int photoNumber = allImages.size();
	for (int i = 0; i < photoNumber; i++)
	{
		if ((i % g_photoNumberOfOnePerson) < g_howManyPhotoForTraining)
		{
			train_images.push_back(allImages[i]);
			train_labels.push_back(allLabels[i]);
		}
		else
		{
			test_images.push_back(allImages[i]);
			test_labels.push_back(allLabels[i]);
		}
	}
	Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
	model->train(train_images, train_labels);
	int iCorrectPrediction = 0;
	int predictedLabel;
	int testPhotoNumber = test_images.size();
	for (int i = 0; i < testPhotoNumber; i++)
	{
		predictedLabel = model->predict(test_images[i]);

		if (predictedLabel == test_labels[i])
			iCorrectPrediction++;
		string result_message = format("预测 Number = %d / 实际 Number = %d.", predictedLabel, test_labels[i]);
		cout << result_message << endl;
	}
	
	cout << "accuracy = " << float(iCorrectPrediction) / testPhotoNumber << endl;

	return 0;
}
