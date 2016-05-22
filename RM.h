#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp> 
#include <opencv2/nonfree/nonfree.hpp>




#include <omp.h>
#include <vector>
#include <queue>

#define RM_ILLUMI_THRE 200
#define RM_DIFF_THRE 30
#define RM_DIFF_LIM 10

namespace RM
{
	extern cv::VideoCapture vedio_stream;

	extern std::vector<cv::Point2f> resrvd_ft;
	extern std::vector<cv::Point2f> cur_ft;

	extern cv::Mat last_frame;//Gray image.
	extern cv::Mat cur_frame;

	
	

	void img_init(cv::Mat& src);

	void get_binary(cv::Mat& src,cv::Mat& dst);

	void diff_proc(const std::vector<cv::Mat>& srcs,cv::Mat& dst,int thre);

	void morphology_filter(cv::Mat& img,cv::Mat& dst,int morph_operator,int morph_elem,int morph_size);

	void get_targets(const cv::Mat& src,const cv::Mat& src_b,std::vector<cv::RotatedRect>& res);

	int find_contours(const cv::Mat& binary_img,std::vector<std::vector<cv::Point> >& contour_container);

	void merge_boxes(const std::vector<cv::RotatedRect>& boxes,std::vector<cv::RotatedRect>& res);
	void merge_boxes(const cv::RotatedRect& a,const cv::RotatedRect& b,cv::RotatedRect& dst);

	bool is_siglamp(const cv::Mat& src,const cv::RotatedRect& rect,int illumi_thre,int diff_lim);

	bool is_in(const cv::Mat& src,const cv::Point& pos);
	bool is_in(const cv::Mat& src,int i,int j);

	void draw_boxes(cv::Mat& img,std::vector<cv::RotatedRect>& boxes);

	void detect_siglamp(cv::Mat& src,std::vector<cv::RotatedRect>& tars);
	void extract_ft(const cv::RotatedRect& sample);

	void match_ft(std::vector<cv::RotatedRect>& tars);
}