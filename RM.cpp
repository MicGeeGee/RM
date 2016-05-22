#include "RM.h"


namespace RM
{
	cv::VideoCapture vedio_stream;

	std::vector<cv::KeyPoint> resrvd_kp;
	cv::Mat resrvd_dpt;
	std::vector<cv::KeyPoint> cur_kp;
	cv::Mat cur_dpt;

	cv::Mat last_frame;//Gray image.
	cv::Mat cur_frame;

	cv::SurfFeatureDetector surf_dtr=cv::SurfFeatureDetector(2500);


	
	void img_init(cv::Mat& src)
	{
		//cv::GaussianBlur(src,src,cv::Size(3,3),0,0);
		//morphology_filter(src,src,0,2,3);
	}

	void get_binary(cv::Mat& src,cv::Mat& dst)
	{
		std::vector<cv::Mat> channels;
		cv::split(src,channels);
		dst=cv::Mat(src.rows,src.cols,CV_8UC1);

		diff_proc(channels,dst,RM_DIFF_THRE);
		morphology_filter(dst,dst,2,2,2);
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
		unsigned char* p0=channels[0].data;
		unsigned char* p1=channels[1].data;
		unsigned char* p2=channels[2].data;

		unsigned char* p=dst.data;
		
		int step=channels[0].cols;



		omp_set_num_threads(8);
#pragma opm parallel for
		for(int i=0;i<channels[0].rows;i++)
			for(int j=0;j<channels[0].cols;j++)
			{
				if(p2[i*step+j]-p0[i*step+j]>thre &&
					p2[i*step+j]-p1[i*step+j]>thre)
					p[i*step+j]=255;
				else
					p[i*step+j]=0;
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
			if(is_siglamp(src,rect,RM_ILLUMI_THRE,RM_DIFF_LIM))
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


	void merge_boxes(const std::vector<cv::RotatedRect>& boxes,std::vector<cv::RotatedRect>& res)
	{
		if(boxes.size()<2)
			return;

		std::queue<cv::RotatedRect> box_que;
		for(int i=0;i<boxes.size();i++)
			box_que.push(boxes[i]);
		while(box_que.size()>=2)
		{
			cv::RotatedRect t_box;
			cv::RotatedRect box1=box_que.front();
			box_que.pop();
			cv::RotatedRect box2=box_que.front();
			box_que.pop();
			merge_boxes(box1,box2,t_box);
			box_que.push(t_box);
		}
		res.push_back(box_que.front());

	}
	void merge_boxes(const cv::RotatedRect& a,const cv::RotatedRect& b,cv::RotatedRect& dst)
	{
		float d_ang;
		d_ang=abs(a.angle-b.angle);
		while(d_ang>180)
			d_ang-=180;
		dst.center.x=(a.center.x+b.center.x)/2;
		dst.center.y=(a.center.y+b.center.y)/2;
		dst.angle=(a.angle+b.angle)/2;

		if(180-d_ang<10)
			dst.angle+=90;
		int nl=(a.size.height+b.size.height)/2;
		int nw=sqrt((a.center.x-b.center.x)*(a.center.x-b.center.x)+
			(a.center.y-b.center.y)*(a.center.y-b.center.y));
		if(nl<nw)
		{
			dst.size.height=nl;
			dst.size.width=nw;
		}
		else
		{
			dst.size.height=nw;
			dst.size.width=nl;
		}
	}


	bool is_siglamp(const cv::Mat& src,const cv::RotatedRect& rect,int illumi_thre,int diff_lim)
	{
		for(int dx=-2;dx<=2;dx++)
			for(int dy=-2;dy<=2;dy++)
			{
				int x=rect.center.x+dx;
				int y=rect.center.y+dy;
				if(is_in(src,cv::Point(x,y)))
				{
					cv::Vec3b val=src.at<cv::Vec3b>(cv::Point(x,y));

					if(val[0]<illumi_thre||val[1]<illumi_thre||val[2]<illumi_thre)
						return false;
					if(abs(val[0]-val[1])>diff_lim||abs(val[0]-val[2])>diff_lim||abs(val[2]-val[1])>diff_lim)
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
		{
			cv::ellipse(img,boxes[i],cv::Scalar(0,0,255),2);
			//cv::rectangle(img,boxes[i].boundingRect(),cv::Scalar(0,0,255),2,8,0);
			cv::circle(img,boxes[i].center,3,cv::Scalar(0,0,255),2,8,0);
		}
		
	}

	void detect_siglamp(cv::Mat& src,std::vector<cv::RotatedRect>& tars)
	{
		cv::Mat b_img;
		std::vector<cv::RotatedRect> boxes;

		img_init(src);
		RM::get_binary(src,b_img);
		
	//	cv::imshow("b_frame",b_img);
	//	cv::waitKey(1);

		RM::get_targets(src,b_img,boxes);
		RM::merge_boxes(boxes,tars);
	}

	void extract_ft(const cv::RotatedRect& sample)
	{
		cv::Mat roi;
		
		cv::Rect b_rect=sample.boundingRect();
		if(b_rect.x<0)
			b_rect.x=0;
		if(b_rect.y<0)
			b_rect.y=0;

		roi=cv::Mat(cur_frame,b_rect);
		
		cv::imshow("roi",roi);
		cv::waitKey(1);
	


		surf_dtr.detect(roi,resrvd_kp);
		for(int i=0;i<resrvd_kp.size()/2;i++)
			resrvd_kp.pop_back();
		for(int i=0;i<resrvd_kp.size()/2;i++)
			resrvd_kp.pop_back();
		for(int i=0;i<resrvd_kp.size()/2;i++)
			resrvd_kp.pop_back();

		surf_dtr.compute(roi,resrvd_kp,resrvd_dpt);

	/*
		//Position correction.
		for(int i=0;i<resrvd_ft.size();i++)
		{	
			resrvd_ft[i].x+=sample.boundingRect().x;
			resrvd_ft[i].y+=sample.boundingRect().y;
		}
*/	
	}

	void match_ft(std::vector<cv::RotatedRect>& tars)
	{
		surf_dtr.detect(cur_frame,cur_kp);
		
		for(int i=0;i<cur_kp.size()/2;i++)
			cur_kp.pop_back();
		for(int i=0;i<cur_kp.size()/2;i++)
			cur_kp.pop_back();
		for(int i=0;i<cur_kp.size()/2;i++)
			cur_kp.pop_back();

		surf_dtr.compute(cur_frame,cur_kp,cur_dpt);

		

		cv::FlannBasedMatcher matcher;
		std::vector<cv::DMatch> matches;
		
		matcher.match(cur_dpt,resrvd_dpt,matches);

		double max_dist=0;
		double min_dist=100;
		for(int i=0;i<matches.size();i++)
		{
			double dist=matches[i].distance;
			if(dist<min_dist)
				min_dist=dist;
			if(dist>max_dist)
				max_dist=dist;
		}

		std::vector<cv::DMatch> op_matches;
		std::vector<cv::Point2f> op_tars;

		for(int i=0;i<matches.size();i++)
		{
			if(matches[i].distance<3*min_dist)
			{
				op_matches.push_back(matches[i]);
				op_tars.push_back(cur_kp[matches[i].queryIdx].pt);
			}
		}
		


		cv::RotatedRect res;
		
		if(op_tars.size()>=5)
			res=cv::fitEllipse(op_tars);
		else
			while(op_tars.size()<5)
			{
				cv::Point2f ps[4];
				ps[0].x=op_tars[0].x+2;
				ps[0].y=op_tars[0].x+2;
				ps[1].x=op_tars[0].x+2;
				ps[1].y=op_tars[0].x-2;
				ps[2].x=op_tars[0].x-2;
				ps[2].y=op_tars[0].x-2;
				ps[3].x=op_tars[0].x-2;
				ps[3].y=op_tars[0].x+2;
				for(int i=0;i<4;i++)
					op_tars.push_back(ps[i]);
				res=cv::fitEllipse(op_tars);
			}

		tars.push_back(res);
		op_tars.clear();
	}
}