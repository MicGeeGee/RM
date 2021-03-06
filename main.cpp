#include "RM.h"
#include <iostream>



int main()
{
	RM::vedio_stream.open("RedCar.mp4");
	cv::Mat r_frame;
	std::vector<cv::RotatedRect> lamp_targets;
	std::vector<cv::RotatedRect> ft_targets;


	while(true)
	{
		RM::vedio_stream>>r_frame;
		cvtColor(r_frame,RM::cur_frame,CV_BGR2GRAY);

		if(r_frame.empty())
		{
			std::cout<<"The end."<<std::endl;
			break;
		}
		
		RM::detect_siglamp(r_frame,lamp_targets);
		
		if(lamp_targets.size()>0)
		{
			RM::extract_ft(lamp_targets[0]);
			RM::draw_boxes(r_frame,lamp_targets);
		}
		else if(RM::resrvd_kp.size()>0)
		{
			RM::match_ft(ft_targets);
			RM::draw_boxes(r_frame,ft_targets);
		}
		
		cv::imshow("RM",r_frame);
		cv::waitKey(1);
		lamp_targets.clear();
		ft_targets.clear();

		RM::last_frame=RM::cur_frame;
	}



	
	return 0;
}