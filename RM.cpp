#include "RM.h"


namespace RM
{
	cv::VideoCapture vedio_stream;

	void img_init(cv::Mat& src)
	{
		cv::GaussianBlur(src,src,cv::Size(3,3),0,0);
	}

	void get_binary(cv::Mat& src,cv::Mat& dst)
	{
		std::vector<cv::Mat> channels;
		cv::split(src,channels);
	


		dst=cv::Mat(src.rows,src.cols,CV_8UC1);

		diff_proc(channels,dst,30);
		//morphology_filter(dst,dst,0,2,3);
	}

	void morphology_filter(cv::Mat& img,cv::Mat& dst,int morph_operator,int morph_elem,int morph_size)
	{
  
		int operation = morph_operator + 2;

		cv::Mat element = cv::getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

		/// 运行指定形态学操作
		morphologyEx( img, dst, operation, element );

		element.release();
  
	}

	void diff_proc(const std::vector<cv::Mat>& channels,cv::Mat& dst,int thre)
	{

		omp_set_num_threads(8);
#pragma opm parallel for

		for(int i=0;i<channels[0].rows;i++)
			for(int j=0;j<channels[0].cols;j++)
			{
				if(channels[2].at<unsigned char>(i,j)-channels[0].at<unsigned char>(i,j)>thre &&
					channels[2].at<unsigned char>(i,j)-channels[1].at<unsigned char>(i,j)>thre)
					dst.at<unsigned char>(i,j)=255;
				else
					dst.at<unsigned char>(i,j)=0;

			}
	}

	void get_targets(const cv::Mat& src,const cv::Mat& src_b,std::vector<cv::RotatedRect>& res)
	{
		std::vector<std::vector<cv::Point> > cons;
		find_contours(src_b,cons);
		
		omp_set_num_threads(8);
#pragma opm parallel for
		for(int i=0;i<cons.size();i++)
		{
			if(cons[i].size()<=10)
				continue;
			cv::RotatedRect rect=cv::fitEllipse(cons[i]);
			if(is_siglight(src,rect))
				res.push_back(rect);
		}
	
	}

	int find_contours(const cv::Mat& binary_img,std::vector<std::vector<cv::Point> >& contour_container)
	{
		std::vector<cv::Vec4i> hierarchy;
		findContours(binary_img,contour_container,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,cv::Point(0, 0));
		std::vector<std::vector<cv::Point> >::iterator it;
		return contour_container.size();
	}

	bool is_siglight(const cv::Mat& src,const cv::RotatedRect& rect)
	{
		for(int dx=-2;dx<=2;dx++)
			for(int dy=-2;dy<=2;dy++)
			{
				int x=rect.center.x+dx;
				int y=rect.center.y+dy;
				if(is_in(src,cv::Point(x,y)))
				{
					cv::Vec3b val=src.at<cv::Vec3b>(cv::Point(x,y));
					if(val[0]<200||val[1]<200||val[2]<200)
						return false;
				}
			}
		return true;
	}

	bool is_in(const cv::Mat& src,const cv::Point& pos)
	{
		int i=pos.y;
		int j=pos.x;
		return i>0 && i<src.rows && j>0 && j>0 && j<src.cols;
	}
	bool is_in(const cv::Mat& src,int i,int j)
	{
		return i>0 && i<src.rows && j>0 && j>0 && j<src.cols;
	}

	void draw_boxes(cv::Mat& img,std::vector<cv::RotatedRect>& boxes)
	{
		for(int i=0;i<boxes.size();i++)
			cv::rectangle(img,boxes[i].boundingRect(),cv::Scalar(0,0,255),2,8,0);
			
		
	}
}