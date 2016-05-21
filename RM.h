#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <omp.h>
#include <vector>

namespace RM
{
	extern cv::VideoCapture vedio_stream;

	void img_init(cv::Mat& src);

	void get_binary(cv::Mat& src,cv::Mat& dst);

	void diff_proc(const std::vector<cv::Mat>& srcs,cv::Mat& dst,int thre);

	void morphology_filter(cv::Mat& img,cv::Mat& dst,int morph_operator,int morph_elem,int morph_size);

	void get_targets(const cv::Mat& src,const cv::Mat& src_b,std::vector<cv::RotatedRect>& res);

	int find_contours(const cv::Mat& binary_img,std::vector<std::vector<cv::Point> >& contour_container);

	bool is_siglight(const cv::Mat& src,const cv::RotatedRect& rect);

	bool is_in(const cv::Mat& src,const cv::Point& pos);
	bool is_in(const cv::Mat& src,int i,int j);

	void draw_boxes(cv::Mat& img,std::vector<cv::RotatedRect>& boxes);
}