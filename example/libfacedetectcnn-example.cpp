/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2019, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

int my_detect_by_video(unsigned char * result_buffer, const String& filename)
{
	
	VideoCapture cap;
	if(strlen(filename.c_str()) < 2)
		cap.open(0);
	else
		cap.open(filename); //打开视频，等价于   VideoCapture cap("E://2.avi");
	if (!cap.isOpened())
	{
		fprintf(stderr, "Can not load the video file %s.\n", filename);
		return -1;
	}

	double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);  //帧宽度
	double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //帧高度
	double frameRate = cap.get(CV_CAP_PROP_FPS);  //帧率 x frames/s
	double totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT); //总帧数
	cout << "视频宽度=" << width << endl;
	cout << "视频高度=" << height << endl;
	cout << "视频总帧数=" << totalFrames << endl;
	cout << "帧率=" << frameRate << endl;
	Mat frame;
	int * pResults = NULL;
	while (1)
	{
		cap >> frame;//等价于cap.read(frame);
		if (frame.empty())
		{
			fprintf(stderr, "frame.empty\n");
			break;
		}

		pResults = facedetect_cnn(result_buffer, (unsigned char*)(frame.ptr(0)), frame.cols, frame.rows, (int)frame.step);

		printf("%d faces detected.\n", (pResults ? *pResults : 0));
		Mat result_cnn = frame.clone();
		//print the detection results
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int confidence = p[4];
			int angle = p[5];

			printf("face_rect=[%d, %d, %d, %d], confidence=%d, angle=%d\n", x, y, w, h, confidence, angle);
			rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
		}
		imshow("result_cnn", result_cnn);
		if(waitKey(20) > 0)
			break;
	}
	return 0;
}
int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("Usage: %s <image_file_name>\n", argv[0]);
        return -1;
    }
	//lxl add for recog video
	if (strstr(argv[1], "demo") != NULL)
	{
		//pBuffer is used in the detection functions.
		//If you call functions in multiple threads, please create one buffer for each thread!
		unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
		if (!pBuffer)
		{
			fprintf(stderr, "Can not alloc buffer.\n");
			return -1;
		}
		cout << "for video test !!!" << endl;
		my_detect_by_video(pBuffer,argv[2]);
		//release the buffer
		free(pBuffer);
		cout << "for video test end!!!" << endl;
		return 0;
	}
	//load an image and convert it to gray (single-channel)
	Mat image = imread(argv[1]); 
	if(image.empty())
	{
		fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
		return -1;
	}

	int * pResults = NULL; 
    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }
#if defined(_OPENMP)
	//lxl add omp
	cout << "my pc max_threads:" << omp_get_max_threads() << endl;
#endif

	///////////////////////////////////////////
	// CNN face detection 
	// Best detection rate
	//////////////////////////////////////////
	//!!! The input image must be a RGB one (three-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);

    printf("%d faces detected.\n", (pResults ? *pResults : 0));
	Mat result_cnn = image.clone();
	//print the detection results
	for(int i = 0; i < (pResults ? *pResults : 0); i++)
	{
        short * p = ((short*)(pResults+1))+142*i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int confidence = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], confidence=%d, angle=%d\n", x, y, w, h, confidence, angle);
		rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
	}
	imshow("result_cnn", result_cnn);

	waitKey();

    //release the buffer
    free(pBuffer);

	return 0;
}