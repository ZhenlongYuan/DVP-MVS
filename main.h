#ifndef _MAIN_H_
#define _MAIN_H_

// Includes Opencv
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
// Includes STD libs
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cstdarg>
#include <random>
#include <unordered_map>
// Includes Boost filesystem
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>


// Define some const var
#define MAX_IMAGES 32
#define NEIGHBOUR_NUM 12
#define NUM_IMAGES 4
#define RANSAC_NUMBER 46857
#define EDGE_NEIGH_NUM 8
#define LAB_BOUNDARY_NUM 8
#define SURF_COEFF_NUM 10

#define DIJK 20

#define MAX_SEARCH_RADIUS 4096
#define DEBUG_POINT_X 753
#define DEBUG_POINT_Y 259
//#define DEBUG_COST_LINE
//#define DEBUG_NEIGHBOUR
// #define DEBUG_COMPLEX

using namespace boost::filesystem;

struct Camera {
	float K[9];
	float R[9];
	float t[3];
	float c[3];
	int height;
	int width;
	float depth_min;
	float depth_max;
};

struct PointList {
	float3 coord;
	float3 color;
};

enum RunState {
	FIRST_INIT,
	REFINE_INIT,
	REFINE_ITER,
};

enum PixelState {
	WEAK,
	STRONG,
	UNKNOWN
};

struct PatchMatchParams {
	int max_iterations = 3;
	int num_images = 5;
	float sigma_spatial = 5.0f;
	float sigma_color = 3.0f;
	int top_k = 4;
	float depth_min = 0.0f;
	float depth_max = 1.0f;
	bool geom_consistency = false;
	int strong_radius = 5;
	int strong_increment = 2;
	int weak_radius = 5;
	int weak_increment = 5;
	bool use_APD = true;
	//=====================================================
	bool use_edge = true;		// ǿ���صĸ���Ұ��չ��������
	bool use_limit = true;  	// ���������ǹ��ɵı�Ե����
	bool use_label = true;		// �����صĸ���Ұ��չ��������
	bool use_detail = false; 	// �ϲ���ʱ�Ա�Ե�Ŷ��ָ�ϸ��
	bool use_radius = true;		// ����Ӧ����ƥ�䴰�ڴ�С
	//=====================================================
	int weak_peak_radius = 2;
	int rotate_time = 4;
	float ransac_threshold = 0.005;
	float geom_factor = 0.2f;
	RunState state;
};

struct Problem {
	int index;
	int ref_image_id;
	std::vector<int> src_image_ids;
	path dense_folder;
	path result_folder;
	int scale_size = 1;
	PatchMatchParams params;
	bool show_medium_result = true;
	int iteration;
};

struct Triangle {
	cv::Point pt1, pt2, pt3;
	float rate1, rate2, rate3;
	Triangle(const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1), pt2(_pt2), pt3(_pt3) {}
};

#endif // !_MAIN_H_
