#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main()
{
	Mat matLeftImage;
	Mat matRightImage;
	Mat matGrayLImage;
	Mat matGrayRImage;

	matLeftImage = imread("object.jpg", IMREAD_COLOR);
	matRightImage = imread("scene.jpg", IMREAD_COLOR);

	if (matLeftImage.empty() || matRightImage.empty())
	{
		std::cout << "image load fail" << std::endl;
		return -1;
	}

	// Gray 이미지 변환 -> 특징점을 빠르게 찾기위함
	cvtColor(matLeftImage, matGrayLImage, CV_RGB2GRAY);
	cvtColor(matRightImage, matGrayRImage, CV_RGB2GRAY);

	if (!matGrayLImage.data || !matGrayRImage.data)

	{
		std::cout << "Gray fail" << std::endl;
		return -1;
	}

	//SIFT Detector를 이용해 키포인트 찾기
	double nMinHessian = 400.;  //thresold
	Ptr<SiftFeatureDetector> Detector = SIFT::create(nMinHessian);
	vector< KeyPoint > vtKeypointsObject, vtKeypointScene;

	// SIFT를 이용해 추출된 특징값을 KeyPoint에 저장
	Detector->detect(matGrayLImage, vtKeypointsObject);
	Detector->detect(matGrayRImage, vtKeypointScene);

	//Descriptors 점에대한 정보 수집
	Ptr<SiftDescriptorExtractor>Extractor = SIFT::create();
	Mat matDescriptorsObject, matDescriptorsScene;

	Extractor->compute(matGrayLImage, vtKeypointsObject, matDescriptorsObject);
	Extractor->compute(matGrayRImage, vtKeypointScene, matDescriptorsScene);

	//Decriptor를 이용해 FLANN을 매칭한다 (all 매칭)
	FlannBasedMatcher Matcher;
	vector<DMatch> matches;

	Matcher.match(matDescriptorsObject, matDescriptorsScene, matches);

	Mat matGoodMatches1;

	drawMatches(matLeftImage, vtKeypointsObject, matRightImage, vtKeypointScene,
		matches, matGoodMatches1,Scalar::all(-1), Scalar(-1), vector<char>(),
		DrawMatchesFlags::DEFAULT);
	imshow("all-matches", matGoodMatches1);

	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;

	// 두 개의 keypoint 사이에서 min-max를 계산
	for (int i = 0; i < matDescriptorsObject.rows; i++)
	{
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;
	}

	printf("-- Max iddst : %f \n", dMaxDist);
	printf("-- Min iddst : %f \n", dMinDist);

	// good maches를 만드는 과정
	vector<DMatch> good_matches;

	for (int i = 0; i < matDescriptorsObject.rows; i++)
	{
		if (matches[i].distance < 5 * dMinDist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	vector<Point2f> obj;
	vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//good matches의 키포인트를 얻는다
		obj.push_back(vtKeypointsObject[good_matches[i].queryIdx].pt);
		scene.push_back(vtKeypointScene[good_matches[i].trainIdx].pt);
	}

	vector<Point2f> obj_corners(4);

	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(matLeftImage.cols, 0);
	obj_corners[2] = Point(matLeftImage.cols, matLeftImage.rows);
	obj_corners[3] = Point(0, matLeftImage.rows);

	vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);

	perspectiveTransform(obj_corners, scene_corners, H);

	Mat matGoodMatcges;

	drawMatches(matLeftImage, vtKeypointsObject, matRightImage, vtKeypointScene, good_matches, matGoodMatcges, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);

	line(matGoodMatcges,

		scene_corners[0] + Point2f((float)matLeftImage.cols, 0), scene_corners[1] + Point2f((float)matLeftImage.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);

	line(matGoodMatcges,

		scene_corners[1] + Point2f((float)matLeftImage.cols, 0), scene_corners[2] + Point2f((float)matLeftImage.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);

	line(matGoodMatcges,

		scene_corners[2] + Point2f((float)matLeftImage.cols, 0), scene_corners[3] + Point2f((float)matLeftImage.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);

	line(matGoodMatcges,

		scene_corners[3] + Point2f((float)matLeftImage.cols, 0), scene_corners[0] + Point2f((float)matLeftImage.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);

	imshow("good-matches", matGoodMatcges);
	waitKey(0);
	imwrite("sift_goodmatch.jpg", matGoodMatcges);
	return 0;
}

