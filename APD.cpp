#include "APD.h"

#define DEBUG_SURF

#define _JACOBI_ROTATE(a, i, j, k, l) \
	g = (a)[j][i]; \
	h = (a)[l][k]; \
	(a)[j][i] = g - s * (h + g * tau); \
	(a)[l][k] = h + s *(g - h * tau);

struct ThreePoint {
	double x;
	double y;
	double z;
};

float getRandomFloat(float min, float max) {
	std::random_device rd;  // 用于生成种子
	std::mt19937 gen(rd()); // Mersenne Twister 随机数生成器
	std::uniform_real_distribution<> dis(min, max);
	return dis(gen);
}

// Function to compute the area of a triangle
double triangleArea(ThreePoint A, ThreePoint B, ThreePoint C) {
	return 0.5 * std::abs(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
}

// Function to calculate the third dimension Z of a point (X, Y) inside the triangle
double calculateZ(ThreePoint A, ThreePoint B, ThreePoint C, double X, double Y) {
	// Create point (X, Y, 0)
	ThreePoint P = { X, Y, 0 };

	// Calculate areas
	double areaABC = triangleArea(A, B, C);
	double areaPBC = triangleArea(P, B, C);
	double areaPCA = triangleArea(P, C, A);
	double areaPAB = triangleArea(P, A, B);

	// Compute the barycentric coordinates
	double u = areaPBC / areaABC;
	double v = areaPCA / areaABC;
	double w = areaPAB / areaABC;

	// Calculate the Z coordinate using barycentric coordinates
	double Z = u * A.z + v * B.z + w * C.z;

	return Z;
}

std::vector<Triangle> DelaunayTriangulation(int cols, int rows, const cv::Rect boundRC, std::vector<float2> xy_temps, std::vector<float> rates)
{
	std::vector<Triangle> results;
	std::vector<cv::Vec6f> temp_results;
	cv::Subdiv2D subdiv2d(boundRC);
	for (const auto xy_temp : xy_temps) {
		if (xy_temp.x >= 0 && xy_temp.x < cols && xy_temp.y >= 0 && xy_temp.y < rows)
			subdiv2d.insert(cv::Point2f(xy_temp.x, xy_temp.y));
	}
	subdiv2d.getTriangleList(temp_results);

	for (const auto temp_vec : temp_results) {
		cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
		cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
		cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
		Triangle temp = Triangle(pt1, pt2, pt3);

		for (int i = 0; i < xy_temps.size(); i++) {
			if (temp_vec[0] == xy_temps[i].x && temp_vec[1] == xy_temps[i].y)
				temp.rate1 = rates[i];
			if (temp_vec[2] == xy_temps[i].x && temp_vec[3] == xy_temps[i].y)
				temp.rate2 = rates[i];
			if (temp_vec[4] == xy_temps[i].x && temp_vec[5] == xy_temps[i].y)
				temp.rate3 = rates[i];
		}

		results.push_back(temp);
	}
	return results;
}

void Get3DPoint_YZL(const Camera camera, const int2 p, const float depth, float* X)
{
	X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
	X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
	X[2] = depth;
}

void Get3DPoint_YZL(const Camera camera, const short2 p, const float depth, float* X)
{
	X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
	X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
	X[2] = depth;
}

float4 GetViewDirection_YZL(const Camera camera, const int2 p, const float depth)
{
	float X[3];
	Get3DPoint_YZL(camera, p, depth, X);
	float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

	float4 view_direction;
	view_direction.x = X[0] / norm;
	view_direction.y = X[1] / norm;
	view_direction.z = X[2] / norm;
	view_direction.w = 0;
	return view_direction;
}

float ComputeDepthfromPlaneHypothesis_YZL(const Camera camera, const float4 plane_hypothesis, const int2 p)
{
	return -plane_hypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * plane_hypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y + camera.K[0] * plane_hypothesis.z);
}

struct Hypot {
	float4 hypothesis; // Assume normal is a 3D vector
	float grayscale;      // Grayscale value
};

cv::Mat Roberts(const cv::Mat& src_image) {
	cv::Mat dst_image = src_image.clone();
	for (int i = 0; i < dst_image.rows; i++) {
		for (int j = 0; j < dst_image.cols; j++) {
			int t1 = 0, t2 = 0;
			if (i > 0 && i < dst_image.rows - 1 && j > 0 && j < dst_image.cols - 1) {
				t1 = (src_image.at<uchar>(i, j) - src_image.at<uchar>(i + 1, j + 1));
				t2 = (src_image.at<uchar>(i + 1, j) - src_image.at<uchar>(i, j + 1));
			}
			else {
				t1 = t2 = 50;
			}
			dst_image.at<uchar>(i, j) = (uchar)sqrt(t1 * t1 + t2 * t2);
		}
	}
	return dst_image;
}

void Label_Seek(int label_1, int label_2, std::vector<std::vector<int>>& connection) {
	bool find_1 = false;
	bool find_2 = false;
	int ind_1 = -1;
	int ind_2 = -1;
	for (int y = 0; y < connection.size(); y++) {
		for (int x = 0; x < connection[y].size(); x++) {
			if (find_1 && find_2)
				break;
			if (label_1 == connection[y][x]) {
				find_1 = true;
				ind_1 = y;
			}
			if (label_2 == connection[y][x]) {
				find_2 = true;
				ind_2 = y;
			}
		}
		if (find_1 && find_2)
			break;
	}
	if (!find_1 && !find_2) { //两个都没找到
		std::vector<int> New;
		New.push_back(label_1);
		New.push_back(label_2);
		connection.push_back(New);
	}
	else if (!find_1 && find_2) { //找到2没找到1，将1带入2
		connection[ind_2].push_back(label_1); //2.pushback[1]
	}
	else if (find_1 && !find_2) {
		connection[ind_1].push_back(label_2);
	}
	else if (find_1 && find_2) { //两个都找到，说明有vector互相融入
		if (ind_1 != ind_2) {
			for (int y = 0; y < connection[ind_2].size(); y++) { //将2带入1
				bool repet = false;
				for (int x = 0; x < connection[ind_1].size(); x++) {
					if (connection[ind_1][x] == connection[ind_2][y]) {
						repet = true;
						break;
					}
				}
				if (!repet) { //如果没有重复
					connection[ind_1].push_back(connection[ind_2][y]);
				}
			}
			connection.erase(connection.begin() + ind_2); //去除2
		}
		else { //ind_1 == ind_2 说明不用管已经规劝了
		}
	}
}

void Label_Update(cv::Mat& label_mask, std::vector<int>& label_cnt) {
	//label更新
	std::vector<std::vector<int>> connection;
	for (int i = 0; i < label_mask.rows - 1; ++i) {
		for (int j = 0; j < label_mask.cols - 1; ++j) {
			int center = label_mask.at<int>(i, j);
			int right = label_mask.at<int>(i, j + 1);
			int down = label_mask.at<int>(i + 1, j);
			if (center != 0 && right != 0 && center != right) {
				Label_Seek(center, right, connection);
			}
			if (center != 0 && down != 0 && center != down) {
				Label_Seek(center, down, connection);
			}
		}
	}

	//构建一个找寻表
	std::vector<int> label_cnt_map;
	for (int i = 0; i < label_cnt.size(); ++i) {
		label_cnt_map.push_back(i);
		label_cnt[i] = 0;
	}
	for (int y = 0; y < connection.size(); y++) {
		for (int x = 0; x < connection[y].size(); x++) {
			label_cnt_map[connection[y][x]] = connection[y][0];
		}
	}

	for (int i = 0; i < label_mask.rows; ++i) {
		for (int j = 0; j < label_mask.cols; ++j) {
			int label = label_mask.at<int>(i, j);
			label_mask.at<int>(i, j) = label_cnt_map[label]; //重新组织

			int real_label = label_mask.at<int>(i, j);
			label_cnt[real_label]++;
		}
	}
}

// 求连通区域
void Connect(const cv::Mat& dst_image, cv::Mat& label_mask, std::vector<int>& label_cnt) {
	std::vector<std::vector<int>> left_neigh(dst_image.rows);
	std::vector<std::vector<int>> up_neigh(dst_image.rows);

	for (int y = 0; y < dst_image.rows; y++) {
		left_neigh[y].resize(dst_image.cols);
		up_neigh[y].resize(dst_image.cols);
		for (int x = 0; x < dst_image.cols; x++) {
			// 左连通
			if (x == 0) {
				left_neigh[y][x] = 0;
			}
			else {
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y, x - 1) == 0) {
					left_neigh[y][x] = 1;
				}
				else {
					left_neigh[y][x] = 0;
				}
			}
			// 上连通
			if (y == 0) {
				up_neigh[y][x] = 0;
			}
			else {
				if (dst_image.at<uchar>(y, x) == 0 && dst_image.at<uchar>(y - 1, x) == 0) {
					up_neigh[y][x] = 1;
				}
				else {
					up_neigh[y][x] = 0;
				}
			}
		}
	}

	// 维护一个并查集
	int cnt = 1;
	std::vector<int> connection;
	connection.push_back(0);
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			if (dst_image.at<uchar>(y, x) == 255) {
				label_mask.at<int>(y, x) = 0;
			}
			else {
				bool left = false, up = false;
				if (left_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					left = true;
				}
				if (up_neigh[y][x] == 1) {
					label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					up = true;
				}
				if (left == false && up == false) {
					label_mask.at<int>(y, x) = cnt;
					connection.push_back(cnt);
					cnt++;
				}
				else if (left == true && up == true) {
					int left_label = label_mask.at<int>(y, x - 1);
					int up_label = label_mask.at<int>(y - 1, x);
					if (left_label > up_label) {
						connection[left_label] = up_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y - 1, x);
					}
					else if (left_label < up_label) {
						connection[up_label] = left_label;
						label_mask.at<int>(y, x) = label_mask.at<int>(y, x - 1);
					}
				}
			}
		}
	}

	for (size_t i = 1; i < connection.size(); i++) {
		int cur_label = connection[i];
		int pre_label = connection[cur_label];
		while (pre_label != cur_label) {
			cur_label = pre_label;
			pre_label = connection[pre_label];
		}
		connection[i] = cur_label;
	}

	int label_num = 1;
	std::vector<int> mapping;
	mapping.push_back(0);
	for (size_t i = 1; i < connection.size(); i++) {
		mapping.push_back(0);
		if (connection[i] == (int)i) {
			mapping[i] = label_num;
			label_num++;  //标签总数
		}
	}

	// 重编号
	for (size_t i = 1; i < connection.size(); i++) {
		connection[i] = mapping[connection[i]];
	}

	for (int i = 0; i < label_num; i++) {
		label_cnt.push_back(0);
	}

	// 连通区域计数
	for (int y = 0; y < dst_image.rows; y++) {
		for (int x = 0; x < dst_image.cols; x++) {
			int label = label_mask.at<int>(y, x);
			label_mask.at<int>(y, x) = connection[label];
			label_cnt[connection[label]]++;
		}
	}
}

cv::Mat EdgeSegment(const int scale, const cv::Mat& src_image, int mode, bool use_canny) {
	/*
		mode - Edge:0; Label:1; Segmentation Image:2
	*/
	const int robthr = 4;
	const int weak_tex_num = (int)(1.0 * src_image.rows * src_image.cols / (1024 << scale << scale));

	cv::Mat src_down;
	cv::resize(src_image, src_down, cv::Size(src_image.cols / 2, src_image.rows / 2), 0, 0, cv::INTER_LINEAR);

	cv::Mat dst_image;
	if (!use_canny) {
		cv::resize(src_down, src_down, cv::Size(src_down.cols / 2, src_down.rows / 2), 0, 0, cv::INTER_LINEAR);

		const int houthr = (int)MIN(src_down.cols, src_down.rows) / 30.0;
		const int min_line_length = (int)MIN(src_down.cols, src_down.rows) / 30.0;
		const int max_line_gap = (int)MIN(src_down.cols, src_down.rows) / 30.0;

		dst_image = Roberts(src_down);
		cv::threshold(dst_image, dst_image, robthr, 255, cv::THRESH_BINARY);

		cv::Mat lab_mask0(dst_image.rows, dst_image.cols, CV_32S);
		std::vector<int> label_cnt0;
		Connect(dst_image, lab_mask0, label_cnt0);

		Label_Update(lab_mask0, label_cnt0);

		for (size_t k = 1; k < label_cnt0.size(); k++) {
			if (label_cnt0[k] < weak_tex_num) continue;
			int weak_index = k;
			cv::Mat img_weak(dst_image.rows, dst_image.cols, CV_8UC1, cv::Scalar(0)); // 直接创建单通道二值图像

			for (int y = 0; y < img_weak.rows; y++) {
				for (int x = 0; x < img_weak.cols; x++) {
					int label = lab_mask0.at<int>(y, x);
					if (label == weak_index) continue;

					bool border = false;
					if (x > 0 && lab_mask0.at<int>(y, x - 1) == weak_index) border = true;
					if (x < img_weak.cols - 1 && lab_mask0.at<int>(y, x + 1) == weak_index) border = true;
					if (y > 0 && lab_mask0.at<int>(y - 1, x) == weak_index) border = true;
					if (y < img_weak.rows - 1 && lab_mask0.at<int>(y + 1, x) == weak_index) border = true;

					if (border)
						img_weak.at<uchar>(y, x) = 255; // 设置边界像素为白色
				}
			}

			std::vector<cv::Vec4i> lines;
			cv::HoughLinesP(img_weak, lines, 1, CV_PI / 180, houthr, min_line_length, max_line_gap); // 直接使用二值图像
			for (size_t i = 0; i < lines.size(); i++) {
				cv::line(dst_image, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0), 1); // 修改为绘制蓝色线条
			}
		}
	}
	else {
		// 求像素中值
		int rows = src_image.rows;
		int cols = src_image.cols;
		int median_val = -1;
		float histogram[256] = { 0 };
		//先计算图像的直方图
		for (int i = 0; i < rows; ++i)
		{
			///获取i行首像素的指针
			const uchar* p = src_image.ptr<uchar>(i);
			///遍历i行像素
			for (int j = 0; j < cols; ++j) {
				histogram[int(*p++)]++;
			}
		}
		int HalfNum = rows * cols / 2;
		int tempSum = 0;
		for (int i = 0; i < 255; i++) {
			tempSum = tempSum + histogram[i];
			if (tempSum > HalfNum) {
				median_val = i;
				break;
			}
		}

		const float sigma = 0.67;
		int threshold1 = (1 - sigma) * median_val;
		int threshold2 = median_val;

		cv::Canny(src_image, dst_image, threshold1, threshold2, 3, true);
	}

	if (mode == 0) {
		cv::resize(dst_image, dst_image, cv::Size(src_image.cols, src_image.rows), 0, 0, cv::INTER_LINEAR);
	}
	else {
		const float factor = 1.0f / (float)(1 << scale);
		const int new_cols = std::round(src_image.cols * factor);
		const int new_rows = std::round(src_image.rows * factor);
		cv::resize(dst_image, dst_image, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
	}

	cv::threshold(dst_image, dst_image, robthr, 255, cv::THRESH_BINARY);

	cv::Mat label_mask(dst_image.rows, dst_image.cols, CV_32S);
	std::vector<int> label_cnt;
	std::vector<int> weakLabel;

	for (int y = 0; y < dst_image.rows; y++) {
		if (dst_image.data[y * dst_image.cols + 1] == 0)
			dst_image.data[y * dst_image.cols] = 0;
		if (dst_image.data[y * dst_image.cols + dst_image.cols - 2] == 0)
			dst_image.data[y * dst_image.cols + dst_image.cols - 1] = 0;
	}
	for (int x = 0; x < dst_image.cols; x++) {
		if (dst_image.data[1 * dst_image.cols + x] == 0)
			dst_image.data[0 * dst_image.cols + x] = 0;
		if (dst_image.data[(dst_image.rows - 2) * dst_image.cols + x] == 0)
			dst_image.data[(dst_image.rows - 1) * dst_image.cols + x] = 0;
	}

	if (mode == 0) {
		return dst_image;
	}

	Connect(dst_image, label_mask, label_cnt);

	Label_Update(label_mask, label_cnt);

	int label_num = label_cnt.size();
	std::vector<cv::Vec3b> colors(label_num);
	colors[0] = cv::Vec3b(0, 0, 0);
	int label_cnt_max = 0;
	for (int i = 1; i < label_num; i++) {
		label_cnt_max = MAX(label_cnt_max, label_cnt[i]);
		colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}

	cv::Mat img_connect(dst_image.rows, dst_image.cols, CV_8UC3);
	for (int y = 0; y < img_connect.rows; y++) {
		for (int x = 0; x < img_connect.cols; x++) {
			img_connect.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			int label = label_mask.at<int>(y, x);
			img_connect.at<cv::Vec3b>(y, x) = colors[label];
			if (label_cnt[label] <= weak_tex_num && label != 0) {
				label_mask.at<int>(y, x) = -1;
			}
		}
	}

	if (mode == 1) {
		return label_mask;
	}

	return img_connect;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
	float3 pointX;
	float3 tmpX;
	// Reprojection
	pointX.x = depth * (x - camera.K[2]) / camera.K[0];
	pointX.y = depth * (y - camera.K[5]) / camera.K[4];
	pointX.z = depth;

	// Rotation
	tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
	tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
	tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

	// Transformation
	float3 C;
	C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
	C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
	C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
	pointX.x = tmpX.x + C.x;
	pointX.y = tmpX.y + C.y;
	pointX.z = tmpX.z + C.z;

	return pointX;
}

float3 Get3DPoint(const Camera camera, int x, int y, const float depth)
{
	float3 X;
	X.x = depth * (x - camera.K[2]) / camera.K[0];
	X.y = depth * (y - camera.K[5]) / camera.K[4];
	X.z = depth;
	return X;
}

void ProjectCamera(const float3 PointX, const Camera camera, float2& point, float& depth)
{
	float3 tmp;
	tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
	tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
	tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

	depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
	point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
	point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

bool ReadBinMat(const path& mat_path, cv::Mat& mat)
{
	ifstream in(mat_path, std::ios_base::binary);
	if (in.bad()) {
		std::cerr << "Error opening file: " << mat_path << std::endl;
		return false;
	}

	int version, rows, cols, type;
	in.read((char*)(&version), sizeof(int));
	in.read((char*)(&rows), sizeof(int));
	in.read((char*)(&cols), sizeof(int));
	in.read((char*)(&type), sizeof(int));

	if (version != 1) {
		in.close();
		std::cerr << "Version error: " << mat_path << std::endl;
		return false;
	}

	mat = cv::Mat(rows, cols, type);
	in.read((char*)mat.data, sizeof(char) * mat.step * mat.rows);
	in.close();
	return true;

}

int writeDepthDmb(const path& mat_path, const cv::Mat_<float> depth)
{
	FILE* outimage;
	outimage = fopen(mat_path.string().c_str(), "wb");
	if (!outimage) {
		std::cout << "Error opening file " << mat_path << std::endl;
	}

	int32_t type = 1;
	int32_t h = depth.rows;
	int32_t w = depth.cols;
	int32_t nb = 1;

	fwrite(&type, sizeof(int32_t), 1, outimage);
	fwrite(&h, sizeof(int32_t), 1, outimage);
	fwrite(&w, sizeof(int32_t), 1, outimage);
	fwrite(&nb, sizeof(int32_t), 1, outimage);

	float* data = (float*)depth.data;

	int32_t datasize = w * h * nb;
	fwrite(data, sizeof(float), datasize, outimage);

	fclose(outimage);
	return 0;
}


int writeNormalDmb(const path& mat_path, const cv::Mat_<cv::Vec3f> normal)
{
	FILE* outimage;
	outimage = fopen(mat_path.string().c_str(), "wb");
	if (!outimage) {
		std::cout << "Error opening file " << mat_path << std::endl;
	}

	int32_t type = 1;
	int32_t h = normal.rows;
	int32_t w = normal.cols;
	int32_t nb = 3;

	fwrite(&type, sizeof(int32_t), 1, outimage);
	fwrite(&h, sizeof(int32_t), 1, outimage);
	fwrite(&w, sizeof(int32_t), 1, outimage);
	fwrite(&nb, sizeof(int32_t), 1, outimage);

	float* data = (float*)normal.data;

	int32_t datasize = w * h * nb;
	fwrite(data, sizeof(float), datasize, outimage);

	fclose(outimage);
	return 0;
}

bool WriteBinMat(const path& mat_path, const cv::Mat& mat) {

	ofstream out(mat_path, std::ios_base::binary);
	if (out.bad()) {
		std::cout << "Error opening file: " << mat_path << std::endl;
		return false;
	}
	int version = 1;
	int rows = mat.rows;
	int cols = mat.cols;
	int type = mat.type();

	out.write((char*)&version, sizeof(int));
	out.write((char*)&rows, sizeof(int));
	out.write((char*)&cols, sizeof(int));
	out.write((char*)&type, sizeof(int));
	out.write((char*)mat.data, sizeof(char) * mat.step * mat.rows);
	out.close();
	return true;
}

bool ReadCamera(const path& cam_path, Camera& cam)
{
	ifstream in(cam_path);
	if (in.bad()) {
		return false;
	}

	std::string line;
	in >> line;

	for (int i = 0; i < 3; ++i) {
		in >> cam.R[3 * i + 0] >> cam.R[3 * i + 1] >> cam.R[3 * i + 2] >> cam.t[i];
	}

	float tmp[4];
	in >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
	in >> line;

	for (int i = 0; i < 3; ++i) {
		in >> cam.K[3 * i + 0] >> cam.K[3 * i + 1] >> cam.K[3 * i + 2];
	}
	// compute camera center in world coord
	const auto& R = cam.R;
	const auto& t = cam.t;
	for (int j = 0; j < 3; ++j) {
		cam.c[j] = -float(double(R[0 + j]) * double(t[0]) + double(R[3 + j]) * double(t[1]) + double(R[6 + j]) * double(t[2]));
	}
	// ====================================================================
	// TAT & ETH version read
	float depth_num;
	float interval;
	in >> cam.depth_min >> interval >> depth_num >> cam.depth_max;
	// ====================================================================
	////DTU version read
	//float depth_num = 192;
	//float interval;
	//in >> cam.depth_min >> interval;
	//cam.depth_max = interval * depth_num + cam.depth_min;
	////====================================================================
	in.close();
	return true;;
}

bool ShowDepthMap(const path& depth_path, const cv::Mat& depth, float depth_min, float depth_max)
{
	const float deltaDepth = depth_max - depth_min;
	// save image
	cv::Mat result_img(depth.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < depth.cols; i++) {
		for (int j = 0; j < depth.rows; j++) {
			if (depth.at<float>(j, i) < depth_min || depth.at<float>(j, i) > depth_max || isnan(depth.at<float>(j, i))) {
				continue;
			}
			float pixel_val = (depth_max - depth.at<float>(j, i)) / deltaDepth;
			if (pixel_val > 1) {
				pixel_val = 1;
			}
			if (pixel_val < 0) {
				pixel_val = 0;
			}
			pixel_val = pixel_val * 255;
			if (pixel_val > 255) {
				pixel_val = 255;
			}
			else if (pixel_val < 0) {
				pixel_val = 0;
			}
			auto& pixel = result_img.at<cv::Vec3b>(j, i);
			if (pixel_val <= 51)
			{
				pixel[0] = 255;
				pixel[1] = pixel_val * 5;
				pixel[2] = 0;
			}
			else if (pixel_val <= 102)
			{
				pixel_val -= 51;
				pixel[0] = 255 - pixel_val * 5;
				pixel[1] = 255;
				pixel[2] = 0;
			}
			else if (pixel_val <= 153)
			{
				pixel_val -= 102;
				pixel[0] = 0;
				pixel[1] = 255;
				pixel[2] = pixel_val * 5;
			}
			else if (pixel_val <= 204)
			{
				pixel_val -= 153;
				pixel[0] = 0;
				pixel[1] = 255 - static_cast<unsigned char>(pixel_val * 128.0 / 51 + 0.5);
				pixel[2] = 255;
			}
			else if (pixel_val <= 255)
			{
				pixel_val -= 204;
				pixel[0] = 0;
				pixel[1] = 127 - static_cast<unsigned char>(pixel_val * 127.0 / 51 + 0.5);
				pixel[2] = 255;
			}

		}
	}
	cv::imwrite(depth_path.string(), result_img);
	return true;
}

bool ShowNormalMap(const path& normal_path, const cv::Mat& normal)
{
	if (normal.empty()) {
		return false;
	}
	cv::Mat normalized_normal = normal.clone();
	for (int i = 0; i < normalized_normal.rows; i++) {
		for (int j = 0; j < normalized_normal.cols; j++) {
			cv::Vec3f normal_val = normalized_normal.at<cv::Vec3f>(i, j);
			float norm = sqrt(pow(normal_val[0], 2) + pow(normal_val[1], 2) + pow(normal_val[2], 2));
			if (norm == 0) {
				normalized_normal.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
			}
			else {
				normalized_normal.at<cv::Vec3f>(i, j) = normal_val / norm;
			}
		}
	}

	cv::Mat img(normalized_normal.size(), CV_8UC3, cv::Scalar(0.f, 0.f, 0.f));
	normalized_normal.convertTo(img, img.type(), 255.f / 2.f, 255.f / 2.f);
	cv::imwrite(normal_path.string(), img);
	return true;
}

bool ShowWeakImage(const path& weak_path, const cv::Mat& weak) {
	// show image
	if (weak.empty()) {
		return false;
	}
	const int width = weak.cols;
	const int height = weak.rows;
	cv::Mat weak_info_image(height, width, CV_8UC3);
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			switch (weak.at<uchar>(r, c))
			{
			case WEAK:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
				break;
			case STRONG:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 255, 0);
				break;
			case UNKNOWN:
				weak_info_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255);
				break;
			}
		}
	}
	// save
	cv::imwrite(weak_path.string(), weak_info_image);
	return true;
}

bool ShowEdgeImage(const path& edge_path, const cv::Mat& edge) {
	if (edge.empty()) {
		return false;
	}
	// save new edge image
	int height = edge.rows;
	int width = edge.cols;
	cv::Mat edge_image(height, width, CV_8UC3);
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			switch (edge.at<uchar>(r, c))
			{
			case 0:
				edge_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				break;
			case 255:
				edge_image.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
				break;
			default:
				edge_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255);
				break;
			}
		}
	}
	cv::imwrite(edge_path.string(), edge_image);
	return true;
}

bool ExportPointCloud(const path& point_cloud_path, std::vector<PointList>& pointcloud)
{
	ofstream out(point_cloud_path, std::ios::binary);
	if (out.bad()) {
		return false;
	}

	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "element vertex " << int(pointcloud.size()) << "\n";
	out << "property float x\n";
	out << "property float y\n";
	out << "property float z\n";
	out << "property uchar diffuse_blue\n";
	out << "property uchar diffuse_green\n";
	out << "property uchar diffuse_red\n";
	out << "end_header\n";

	for (size_t idx = 0; idx < pointcloud.size(); idx++)
	{
		float px = pointcloud[idx].coord.x;
		float py = pointcloud[idx].coord.y;
		float pz = pointcloud[idx].coord.z;


		cv::Vec3b pixel;
		pixel[0] = static_cast<uchar>(pointcloud[idx].color.x);
		pixel[1] = static_cast<uchar>(pointcloud[idx].color.y);
		pixel[2] = static_cast<uchar>(pointcloud[idx].color.z);

		out.write((char*)&px, sizeof(float));
		out.write((char*)&py, sizeof(float));
		out.write((char*)&pz, sizeof(float));

		out.write((char*)&pixel[0], sizeof(uchar));
		out.write((char*)&pixel[1], sizeof(uchar));
		out.write((char*)&pixel[2], sizeof(uchar));
	}
	out.close();
	return true;
}

void StringAppendV(std::string* dst, const char* format, va_list ap) {
	// First try with a small fixed size buffer.
	static const int kFixedBufferSize = 1024;
	char fixed_buffer[kFixedBufferSize];

	// It is possible for methods that use a va_list to invalidate
	// the data in it upon use.  The fix is to make a copy
	// of the structure before using it and use that copy instead.
	va_list backup_ap;
	va_copy(backup_ap, ap);
	int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
	va_end(backup_ap);

	if (result < kFixedBufferSize) {
		if (result >= 0) {
			// Normal case - everything fits.
			dst->append(fixed_buffer, result);
			return;
		}

#ifdef _MSC_VER
		// Error or MSVC running out of space.  MSVC 8.0 and higher
		// can be asked about space needed with the special idiom below:
		va_copy(backup_ap, ap);
		result = vsnprintf(nullptr, 0, format, backup_ap);
		va_end(backup_ap);
#endif

		if (result < 0) {
			// Just an error.
			return;
		}
	}

	// Increase the buffer size to the size requested by vsnprintf,
	// plus one for the closing \0.
	const int variable_buffer_size = result + 1;
	std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

	// Restore the va_list before we use it again.
	va_copy(backup_ap, ap);
	result =
		vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
	va_end(backup_ap);

	if (result >= 0 && result < variable_buffer_size) {
		dst->append(variable_buffer.get(), result);
	}
}

std::string StringPrintf(const char* format, ...) {
	va_list ap;
	va_start(ap, format);
	std::string result;
	StringAppendV(&result, format, ap);
	va_end(ap);
	return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
	const int line) {
	if (error != cudaSuccess) {
		std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
			file.c_str(), line)
			<< std::endl;
		exit(EXIT_FAILURE);
	}
}

void CudaCheckError(const char* file, const int line) {
	cudaError error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
			line, cudaGetErrorString(error))
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	error = cudaDeviceSynchronize();
	if (cudaSuccess != error) {
		std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
			file, line, cudaGetErrorString(error))
			<< std::endl;
		std::cerr
			<< "This error is likely caused by the graphics card timeout "
			"detection mechanism of your operating system. Please refer to "
			"the FAQ in the documentation on how to solve this problem."
			<< std::endl;
		exit(EXIT_FAILURE);
	}
}

std::string ToFormatIndex(int index) {
	std::stringstream ss;
	ss << std::setw(8) << std::setfill('0') << index;
	return ss.str();
}

APD::APD(const Problem& problem) {
	params_host = problem.params;
	this->problem = problem;
}

APD::~APD() {
	delete[] plane_hypotheses_host;

	if (problem.params.use_edge || problem.params.use_limit) {
		cudaFree(edge_cuda);
	}
	if (problem.params.use_edge) {
		cudaFree(edge_neigh_cuda);
		cudaFree(complex_cuda);
	}
	if (problem.params.use_label) {
		cudaFree(label_cuda);
		cudaFree(label_boundary_cuda);
	}

	if (problem.params.use_radius) {
		cudaFree(radius_cuda);
	}
	// free images
	{
		for (int i = 0; i < num_images; ++i) {
			cudaDestroyTextureObject(texture_objects_host.images[i]);
			cudaFreeArray(cuArray[i]);
		}
		cudaFree(texture_objects_cuda);
	}
	// may free depths
	if (params_host.geom_consistency) {
		for (int i = 0; i < num_images; ++i) {
			cudaDestroyTextureObject(texture_depths_host.images[i]);
			cudaFreeArray(cuDepthArray[i]);
		}
		cudaFree(texture_depths_cuda);
	}
	// may free supports
	cudaFree(candidate_cuda);
	cudaFree(cameras_cuda);
	cudaFree(plane_hypotheses_cuda);
	cudaFree(fit_plane_hypotheses_cuda);
	cudaFree(costs_cuda);
	cudaFree(rand_states_cuda);
	cudaFree(selected_views_cuda);
	cudaFree(params_cuda);
	cudaFree(helper_cuda);
	cudaFree(neighbours_cuda);
	cudaFree(neigbours_map_cuda);
	cudaFree(weak_info_cuda);
	cudaFree(weak_reliable_cuda);
	cudaFree(view_weight_cuda);
	cudaFree(weak_nearest_strong);
#ifdef DEBUG_COST_LINE
	cudaFree(weak_ncc_cost_cuda);
#endif // DEBUG_COST_LINE

}

void APD::InuputInitialization() {
	images.clear();
	cameras.clear();
	// get folder
	path image_folder = problem.dense_folder / path("images");
	path cam_folder = problem.dense_folder / path("cams");
	//path weak_folder = problem.dense_folder / path("weaks");
	// =================================================
	// read ref image and src images
	// ref
	{
		path ref_image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		cv::Mat_<uint8_t> image_uint = cv::imread(ref_image_path.string(), cv::IMREAD_GRAYSCALE);
		cv::Mat image_float;
		image_uint.convertTo(image_float, CV_32FC1);
		images.push_back(image_float);
		width = image_float.cols;
		height = image_float.rows;
	}
	// src
	for (const auto& src_idx : problem.src_image_ids) {
		path src_image_path = image_folder / path(ToFormatIndex(src_idx) + ".jpg");
		cv::Mat_<uint8_t> image_uint = cv::imread(src_image_path.string(), cv::IMREAD_GRAYSCALE);
		cv::Mat image_float;
		image_uint.convertTo(image_float, CV_32FC1);

		cv::Mat Resize_Image = cv::Mat::zeros(height, width, image_float.type());
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (i < image_float.rows && j < image_float.cols) {
					Resize_Image.at<float>(i, j) = image_float.at<float>(i, j);
				}
			}
		}
		images.push_back(Resize_Image);
		// assert: images_float.cols == width;
		// assert: images_float.rows == height;
	}
	if (images.size() > MAX_IMAGES) {
		std::cerr << "Can't process so much images: " << images.size() << std::endl;
		exit(EXIT_FAILURE);
	}
	// =================================================
	// read ref camera and src camera
	// ref
	{
		path ref_cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera cam;
		ReadCamera(ref_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// src
	for (const auto& src_idx : problem.src_image_ids) {
		path src_cam_path = cam_folder / path(ToFormatIndex(src_idx) + "_cam.txt");
		Camera cam;
		ReadCamera(src_cam_path, cam);
		cam.width = width;
		cam.height = height;
		cameras.push_back(cam);
	}
	// =================================================
	// set some params
	params_host.depth_min = cameras[0].depth_min * 0.6f;
	params_host.depth_max = cameras[0].depth_max * 1.2f;
	params_host.num_images = (int)images.size();
	num_images = (int)images.size();
	// =================================================
	std::cout << "Read images and camera done\n";
	std::cout << "Depth range: " << params_host.depth_min << " " << params_host.depth_max << std::endl;
	std::cout << "Num images: " << params_host.num_images << std::endl;
	// =================================================
	// scale images
	if (problem.scale_size != 1) {
		for (int i = 0; i < num_images; ++i) {
			const float factor = 1.0f / (float)(problem.scale_size);
			const int new_cols = std::round(images[i].cols * factor);
			const int new_rows = std::round(images[i].rows * factor);

			const float scale_x = new_cols / static_cast<float>(images[i].cols);
			const float scale_y = new_rows / static_cast<float>(images[i].rows);

			cv::Mat_<float> scaled_image_float;
			cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
			images[i] = scaled_image_float.clone();

			width = scaled_image_float.cols;
			height = scaled_image_float.rows;

			cameras[i].K[0] *= scale_x;
			cameras[i].K[2] *= scale_x;
			cameras[i].K[4] *= scale_y;
			cameras[i].K[5] *= scale_y;
			cameras[i].width = width;
			cameras[i].height = height;
		}
		std::cout << "Scale images and cameras done\n";
	}
	std::cout << "Image size: " << width << " * " << height << std::endl;
	// =================================================
	// read depth form geom consistency
	if (params_host.geom_consistency) { //YZL 后面除了第一轮，都需要用到深度来计算可见性
	//if (true) {
		depths.clear();
		path ref_depth_path = problem.result_folder / path("depths.dmb");
		cv::Mat ref_depth;
		ReadBinMat(ref_depth_path, ref_depth);
		depths.push_back(ref_depth);
		for (const auto& src_idx : problem.src_image_ids) {
			path src_depth_path = problem.dense_folder / path("APD") / path(ToFormatIndex(src_idx)) / path("depths.dmb");
			cv::Mat src_depth;
			ReadBinMat(src_depth_path, src_depth);
			depths.push_back(src_depth);
		}
		for (auto& depth : depths) {
			if (depth.cols != width || depth.rows != height) {
				RescaleMatToTargetSize<float>(depth, depth, cv::Size(width, height));
			}
		}

	}
	// =================================================
	// read weak info
	if (params_host.use_APD) {
		path weak_info_path = problem.result_folder / path("weak.bin");
		if (!exists(weak_info_path)) {
			std::cerr << "Can't find weak info file: " << weak_info_path.string() << std::endl;
			exit(EXIT_FAILURE);
		}
		ReadBinMat(weak_info_path, weak_info_host);
		if (weak_info_host.cols != width || weak_info_host.rows != height) {
			std::cerr << "Weak info doesn't match the images' size!\n";
			RescaleMatToTargetSize<uchar>(weak_info_host, weak_info_host, cv::Size(width, height));
			std::cout << "Scale done\n";
		}

		neighbours_map_host = cv::Mat::zeros(weak_info_host.size(), CV_32SC1);
		weak_count = 0;
		for (int r = 0; r < weak_info_host.rows; ++r) {
			for (int c = 0; c < weak_info_host.cols; ++c) {
				int val = weak_info_host.at<uchar>(r, c);
				// point is not strong
				if (val == WEAK) {
					neighbours_map_host.at<int>(r, c) = weak_count;
					weak_count++;
				}
			}
		}
		std::cout << "Weak count: " << weak_count << " / " << weak_info_host.cols * weak_info_host.rows << " = " << (float)weak_count / (float)(weak_info_host.cols * weak_info_host.rows) * 100 << "%" << std::endl;
	}
	else {
		weak_info_host = cv::Mat::zeros(height, width, CV_8UC1);
		weak_count = 0;
		for (int r = 0; r < weak_info_host.rows; ++r) {
			for (int c = 0; c < weak_info_host.cols; ++c) {
				weak_info_host.at<uchar>(r, c) = STRONG;
			}
		}
	}
	// ==================================================================================================
	// 这里想直接进行初始化

	plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];

	if (params_host.state == FIRST_INIT) { //原图大小,这里只需要对自身重新生成，然后根据生成结果去用于自身初始化即可
	//if (false){
		path dep_folder = problem.dense_folder / path("dep");
		path sfm_folder = problem.dense_folder / path("sfm");

		path ref_dep_folder = dep_folder / path(ToFormatIndex(problem.ref_image_id) + ".dmb");
		cv::Mat dep;
		ReadBinMat(ref_dep_folder, dep);

		for (int y = 0; y < dep.rows; y++) {
			for (int x = 0; x < dep.cols; x++) {
				dep.at<float>(y, x) = 255 - (dep.at<float>(y, x));
			}
		}

		path ref_sfm_folder = sfm_folder / path(ToFormatIndex(problem.ref_image_id) + ".txt");

		ifstream file(ref_sfm_folder);
		std::stringstream iss;
		std::string line;

		// 定义临时存储当前文件坐标和颜色的vector
		std::vector<float2> xy_temp;
		std::vector<float3> xyz_temp;
		std::vector<float2> xy_temps;
		std::vector<float> rates;
		cv::Mat all_rate_map(dep.rows, dep.cols, CV_32FC1);

		iss.clear();
		while (std::getline(file, line)) {
			//iss.str(line);
			std::istringstream iss(line);

			float x_2d, y_2d, x_3d, y_3d, z_3d;
			int r, g, b;
			iss >> x_2d >> y_2d >> x_3d >> y_3d >> z_3d >> r >> g >> b;
			xy_temp.push_back(make_float2(x_2d, y_2d));
			xyz_temp.push_back(make_float3(x_3d, y_3d, z_3d));
		}

		//std::cout << "xyz_temp: " << xyz_temp[0].x << xyz_temp[0].y << xyz_temp[0].z << std::endl;
		//std::cout << "xyz_temp.size(): " << xyz_temp.size() << std::endl;

		for (int i = 0; i < xy_temp.size(); i++) {
			float3 PointX = xyz_temp[i];
			float2 point;
			float proj_depth;

			path ref_cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
			Camera cam;
			ReadCamera(ref_cam_path, cam);
			ProjectCamera(PointX, cam, point, proj_depth);

			if (int(point.x + 0.5f) > 0 && int(point.x + 0.5f) < dep.cols && int(point.y + 0.5f) > 0 && int(point.y + 0.5f) < dep.rows) {
				float rate = dep.at<float>(int(point.y + 0.5f), int(point.x + 0.5f)) / proj_depth;
				rates.push_back(rate);
				xy_temps.push_back(xy_temp[i]);
			}
		}

		float rate_max = 0;
		for (int i = 0; i < rates.size(); i++) {
			rate_max = MAX(rate_max, rates[i]);
			//std::cout << rates[i] << " ";
		}

		float middle_rate = rates[rates.size() / 2];
		for (int y = 0; y < dep.rows; y++) {
			for (int x = 0; x < dep.cols; x++) {
				all_rate_map.at<float>(y, x) = middle_rate;
			}
		}

		//std::cout << "middle_rate: " << middle_rate << std::endl;
		path ref_image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		cv::Mat image_A = cv::imread(ref_image_path.string(), cv::IMREAD_COLOR);
		cv::Mat srcImage = image_A.clone();

		for (int i = 0; i < rates.size(); i++) {
			int red = rates[i] / rate_max * 255;
			if (xy_temps[i].x >= 0 && xy_temps[i].x < width && xy_temps[i].y >= 0 && xy_temps[i].y < height) {
				float real = dep.at<float>(xy_temps[i].y, xy_temps[i].x) / middle_rate;
				cv::circle(image_A, cv::Point(xy_temps[i].x, xy_temps[i].y), 5, cv::Scalar(0, 0, red), -1);
				// Convert red value to string
				std::stringstream ss;
				ss << real;
				// Draw text next to the circle
				int near;
				if (xy_temps[i].x < image_A.cols - 20)
					near = 10;
				else
					near = -10;
				cv::Point textOrg(xy_temps[i].x + near, xy_temps[i].y); // Adjust text position as needed
				cv::putText(image_A, ss.str(), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
			}
		}

		path colmap_img_path = problem.result_folder / path("COLMAP_" + std::to_string(problem.iteration) + ".jpg");
		cv::imwrite(colmap_img_path.string(), image_A);
		const cv::Rect imageRC(0, 0, dep.cols, dep.rows);
		const auto triangles = DelaunayTriangulation(dep.cols, dep.rows, imageRC, xy_temps, rates);

		for (const auto triangle : triangles) {
			if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
				cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255), 4);
				cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255), 4);
				cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255), 4);
			}
		}

		path triangulation_path = problem.result_folder / path("Tri_" + std::to_string(problem.iteration) + ".jpg");
		cv::imwrite(triangulation_path.string(), srcImage);
		cv::Mat_<float> mask_tri = cv::Mat::zeros(dep.rows, dep.cols, CV_32FC1);
		uint32_t idx = 0;

		for (const auto triangle : triangles) {
			if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
				float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
				float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
				float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));
				float max_edge_length = std::max(L01, std::max(L02, L12));
				float step = 1.0 / max_edge_length;

				for (float p = 0; p < 1.0; p += step) {
					for (float q = 0; q < 1.0 - p; q += step) {
						int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
						int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;

						mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas	

						ThreePoint A = { triangle.pt1.x, triangle.pt1.y, triangle.rate1 };
						ThreePoint B = { triangle.pt2.x, triangle.pt2.y, triangle.rate2 };
						ThreePoint C = { triangle.pt3.x, triangle.pt3.y, triangle.rate3 };

						double rateZ = calculateZ(A, B, C, x, y);
						all_rate_map.at<float>(y, x) = rateZ;
					}
				}
				idx++;
			}
		}

		for (int y = 0; y < dep.rows; y++) {
			for (int x = 0; x < dep.cols; x++) {
				dep.at<float>(y, x) /= all_rate_map.at<float>(y, x);
			}
		}

		path depth_img_path = problem.result_folder / path("depth_anything_" + std::to_string(problem.iteration) + ".jpg");
		ShowDepthMap(depth_img_path, dep, cameras[0].depth_min * 0.6f, cameras[0].depth_max * 1.2f);

		if (dep.cols != width || dep.rows != height) {
			RescaleMatToTargetSize<float>(dep, dep, cv::Size2i(width, height));
		}

		cv::Mat normalMap(dep.rows, dep.cols, CV_32FC3);

		for (int y = 1; y < height - 1; ++y) {
			for (int x = 1; x < width - 1; ++x) {
				float3 X = Get3DPoint(cameras[0], x, y, dep.at<float>(y, x));
				float3 X_dx = Get3DPoint(cameras[0], x + 1, y, dep.at<float>(y, x + 1));
				float3 X_dy = Get3DPoint(cameras[0], x, y + 1, dep.at<float>(y + 1, x));

				// 计算梯度向量
				cv::Vec3f dPdx(X_dx.x - X.x, X_dx.y - X.y, X_dx.z - X.z);
				cv::Vec3f dPdy(X_dy.x - X.x, X_dy.y - X.y, X_dy.z - X.z);

				// 计算法线
				cv::Vec3f normal = dPdx.cross(dPdy);
				normal = cv::normalize(normal);

				//看方向
				float norm = sqrt(X.x * X.x + X.y * X.y + X.z * X.z);
				float4 view_direction;
				view_direction.x = X.x / norm;
				view_direction.y = X.y / norm;
				view_direction.z = X.z / norm;
				view_direction.w = 0;

				float dot_product = normal[0] * view_direction.x + normal[1] * view_direction.y + normal[2] * view_direction.z;
				if (dot_product > 0.0f) {
					normal[0] = -normal[0];
					normal[1] = -normal[1];
					normal[2] = -normal[2];
				}

				cv::Mat R = (cv::Mat_<float>(3, 3) <<
					cameras[0].R[0], cameras[0].R[1], cameras[0].R[2],
					cameras[0].R[3], cameras[0].R[4], cameras[0].R[5],
					cameras[0].R[6], cameras[0].R[7], cameras[0].R[8]);
				cv::Mat Rt = R.t();
				cv::Vec3f transformed_normal;
				transformed_normal[0] = Rt.at<float>(0, 0) * normal[0] + Rt.at<float>(0, 1) * normal[1] + Rt.at<float>(0, 2) * normal[2];
				transformed_normal[1] = Rt.at<float>(1, 0) * normal[0] + Rt.at<float>(1, 1) * normal[1] + Rt.at<float>(1, 2) * normal[2];
				transformed_normal[2] = Rt.at<float>(2, 0) * normal[0] + Rt.at<float>(2, 1) * normal[1] + Rt.at<float>(2, 2) * normal[2];
				normal = transformed_normal;

				normalMap.at<cv::Vec3f>(y, x) = normal;
			}
		}

		path normal_img_path = problem.result_folder / path("normal_COLMAP.jpg");
		ShowNormalMap(normal_img_path, normalMap);

		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				plane_hypotheses_host[center].x = normalMap.at<cv::Vec3f>(row, col)[0];
				plane_hypotheses_host[center].y = normalMap.at<cv::Vec3f>(row, col)[1];
				plane_hypotheses_host[center].z = normalMap.at<cv::Vec3f>(row, col)[2];
				plane_hypotheses_host[center].w = dep.at<float>(row, col);
			}
		}

	}

	// ==================================================================================================
	selected_views_host = cv::Mat::zeros(height, width, CV_32SC1);
	if (params_host.state != FIRST_INIT) {
		// input plane hypotheses from existed result
		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("APD_normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);
		if (depth.cols != width || depth.rows != height || normal.cols != width || normal.rows != height) {
			std::cerr << "Depth and Normal doesn't match the images' size!\n";
			RescaleMatToTargetSize<float>(depth, depth, cv::Size2i(width, height));
			RescaleMatToTargetSize<cv::Vec3f>(normal, normal, cv::Size2i(width, height));
		}
		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				int center = row * width + col;
				plane_hypotheses_host[center].w = depth.at<float>(row, col);
				plane_hypotheses_host[center].x = normal.at<cv::Vec3f>(row, col)[0];
				plane_hypotheses_host[center].y = normal.at<cv::Vec3f>(row, col)[1];
				plane_hypotheses_host[center].z = normal.at<cv::Vec3f>(row, col)[2];
			}
		}
		{
			path selected_view_path = problem.result_folder / path("selected_views.bin");
			ReadBinMat(selected_view_path, selected_views_host);
			if (selected_views_host.cols != width || selected_views_host.rows != height) {
				std::cerr << "Select view doesn't match the images' size!\n";
				RescaleMatToTargetSize<unsigned int>(selected_views_host, selected_views_host, cv::Size2i(width, height));
			}
		}



		// TANKS AND TEMPLES

		// path mvs_folder = problem.dense_folder / path("MVS3");

		// path ref_dep_folder = mvs_folder / path(ToFormatIndex(problem.ref_image_id) + ".png");
		// cv::Mat mvs_image = cv::imread(ref_dep_folder.string(), cv::IMREAD_COLOR);

		// cv::resize(mvs_image, mvs_image, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

		// //双视图
		// for (int n = 0; n <= problem.src_image_ids.size(); n++) { //-1是为了看全局的
		// 	cv::Mat img_connect_copy = mvs_image.clone();

		// 	if (n < problem.src_image_ids.size()) {
		// 		//加入可见性
		// 		for (int i = 0; i < height; ++i) {
		// 			for (int j = 0; j < width; ++j) {
		// 				int center = i * width + j;
		// 				unsigned int views = selected_views_host.at<int>(i, j);
		// 				if ((views >> n) & 1) { //可见
		// 				}
		// 				else { //不可见
		// 					img_connect_copy.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
		// 				}
		// 			}
		// 		}
		// 	}

		// 	std::string view_name = "View_con_" + std::to_string(n) + ".png";
		// 	path view_path = problem.result_folder / path(view_name);
		// 	cv::imwrite(view_path.string(), img_connect_copy);
		// }

	}
	// =================================================
}

void APD::CudaSpaceInitialization() {
	// =================================================
	// move images to gpu

	for (int i = 0; i < num_images; ++i) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaMallocArray(&cuArray[i], &channelDesc, width, height);
		cudaMemcpy2DToArray(cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray[i];
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;
		cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
	}
	cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
	cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
	// may move depths to gpu

	if (params_host.geom_consistency) {
		//if (true) {
		for (int i = 0; i < num_images; ++i) {
			int height = depths[i].rows;
			int width = depths[i].cols;
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			cudaMallocArray(&cuDepthArray[i], &channelDesc, width, height);
			cudaMemcpy2DToArray(cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(cudaResourceDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cuDepthArray[i];
			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(cudaTextureDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 0;
			cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
		}
		cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
		cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
	}
	// =================================================
	// move camera to gpu

	cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
	cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);
	// malloc memory for important data structure
	const int length = width * height;

	// define cost
	cudaMalloc((void**)&costs_cuda, sizeof(float) * length);
	// malloc memory for rand states
	cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * length);
	// malloc for selected_views

	cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * length);
	cudaMemcpy(selected_views_cuda, selected_views_host.ptr<unsigned int>(0), sizeof(unsigned int) * length, cudaMemcpyHostToDevice);
	// view weight

	cudaMalloc((void**)&view_weight_cuda, sizeof(uchar) * length * MAX_IMAGES);
	// move plane hypotheses to gpu
	cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * length);
	cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * length, cudaMemcpyHostToDevice);
	// malloc memory for fit plane 

	cudaMalloc((void**)&fit_plane_hypotheses_cuda, sizeof(float4) * length);
	cudaMemset(fit_plane_hypotheses_cuda, 0, sizeof(float4) * length);

	cudaMalloc((void**)(&candidate_cuda), length * LAB_BOUNDARY_NUM * NUM_IMAGES * sizeof(short2));

	// malloc edge array
	if (problem.params.use_edge || problem.params.use_limit) {
		cudaMalloc((void**)&edge_cuda, sizeof(uint8_t) * length);
		cudaMemcpy(edge_cuda, edge_host.ptr<uchar>(0), sizeof(uchar) * length, cudaMemcpyHostToDevice);
	}
	if (problem.params.use_edge) {
		cudaMalloc((void**)(&edge_neigh_cuda), length * EDGE_NEIGH_NUM * sizeof(short2));
		cudaMalloc((void**)&complex_cuda, sizeof(float) * weak_count);
	}
	if (problem.params.use_label) {
		cudaMalloc((void**)(&label_cuda), length * sizeof(int));
		cudaMemcpy(label_cuda, label_host.ptr<int>(0), sizeof(int) * (height * width), cudaMemcpyHostToDevice);
		cudaMalloc((void**)(&label_boundary_cuda), weak_count * LAB_BOUNDARY_NUM * sizeof(short2));
	}
	if (problem.params.use_radius) {
		cudaMalloc((void**)(&radius_cuda), length * sizeof(int));
		cudaMemcpy(radius_cuda, radius_host.ptr<int>(0), sizeof(int) * (height * width), cudaMemcpyHostToDevice);
	}

	// malloc memory for weak info
	cudaMalloc((void**)(&weak_info_cuda), length * sizeof(uchar));
	cudaMemcpy(weak_info_cuda, weak_info_host.ptr<uchar>(0), length * sizeof(uchar), cudaMemcpyHostToDevice);
	// malloc memory for weak reliable info
	cudaMalloc((void**)(&weak_reliable_cuda), length * sizeof(uchar));
	// malloc memory for nearest strong points
	cudaMalloc((void**)(&weak_nearest_strong), length * sizeof(short2));
	// move neighbour map to gpu
	cudaMalloc((void**)(&neigbours_map_cuda), length * sizeof(int));
	cudaMemcpy(neigbours_map_cuda, neighbours_map_host.ptr<int>(0), length * sizeof(int), cudaMemcpyHostToDevice);
	// malloc memory for deformable ncc
	cudaMalloc((void**)(&neighbours_cuda), weak_count * NEIGHBOUR_NUM * sizeof(short2));
	// move param to gpu
	cudaMalloc((void**)(&params_cuda), sizeof(PatchMatchParams));
	cudaMemcpy(params_cuda, &params_host, sizeof(PatchMatchParams), cudaMemcpyHostToDevice);
	// =================================================
#ifdef DEBUG_COST_LINE
	cudaMalloc((void**)(&weak_ncc_cost_cuda), sizeof(float) * width * height * 61);
#endif // DEBUG_COST_LINE
}

void APD::SupportInitialization() {
	int scale = 0;
	while ((1 << scale) < problem.scale_size) scale++;

	if (problem.params.use_edge || problem.params.use_limit) {
		int scale = 0;
		while ((1 << scale) < problem.scale_size) scale++;
		path edge_path = problem.result_folder / path("edges_" + std::to_string(scale) + ".dmb");
		// read image edge info
		ReadBinMat(edge_path, edge_host);
	}

	// !!!!!!!!!重要!!!!!!!!!
	// 两种版本 Different
	// 这里是DVP-MVS的结果，但是需要部署DEPANY然后预处理数据集，所以较为麻烦，可先用简易版本替换计算
	// if (problem.params.use_label) {
	// 	path label_path = problem.result_folder / path("labels_" + std::to_string(scale) + ".dmb");
	// 	ReadBinMat(label_path, label_host);
	// }

	// 这里是简易版本的TSAR-MVS的结果
	if (problem.params.use_label) {
		path mvs_folder = problem.dense_folder / path("MVS4");
		path ref_dep_folder = mvs_folder / path(ToFormatIndex(problem.ref_image_id) + ".dmb");
		cv::Mat ref_dep;
		ReadBinMat(ref_dep_folder, ref_dep);
		if (ref_dep.cols != width || ref_dep.rows != height) {
			std::cerr << "Depth and Normal doesn't match the images' size!\n";
			RescaleMatToTargetSize<float>(ref_dep, label_host, cv::Size2i(width, height));
		}
	}

	if (problem.params.use_radius) {
		int strong_radius = problem.params.strong_radius;
		if (problem.params.state == FIRST_INIT) {
			radius_host = cv::Mat::zeros(height, width, CV_32S);
			for (int r = 0; r < height; r++)
				for (int c = 0; c < width; c++)
					radius_host.at<int>(r, c) = strong_radius;
		}
		else {
			path radius_path = problem.result_folder / path("radius.bin");
			ReadBinMat(radius_path, radius_host);
		}
		if (radius_host.cols != width || radius_host.rows != height) {
			std::cerr << "Radius map doesn't match the images' size!\n";
			RescaleMatToTargetSize<int>(radius_host, radius_host, cv::Size2i(width, height));
		}
		for (int r = 0; r < height; r++)
			for (int c = 0; c < width; c++)
				if (weak_info_host.at<uchar>(r, c) == UNKNOWN)
					radius_host.at<int>(r, c) = strong_radius;
	}
}

void APD::SetDataPassHelperInCuda() {
	helper_host.width = this->width;
	helper_host.height = this->height;
	helper_host.ref_index = this->problem.ref_image_id;
	helper_host.texture_depths_cuda = this->texture_depths_cuda;
	helper_host.texture_objects_cuda = this->texture_objects_cuda;
	helper_host.cameras_cuda = this->cameras_cuda;
	helper_host.costs_cuda = this->costs_cuda;
	helper_host.neighbours_cuda = this->neighbours_cuda;
	helper_host.neighbours_map_cuda = this->neigbours_map_cuda;
	helper_host.plane_hypotheses_cuda = this->plane_hypotheses_cuda;
	helper_host.rand_states_cuda = this->rand_states_cuda;
	helper_host.selected_views_cuda = this->selected_views_cuda;
	helper_host.weak_info_cuda = this->weak_info_cuda;
	helper_host.params = params_cuda;
	helper_host.debug_point = make_int2(DEBUG_POINT_X, DEBUG_POINT_Y);
	helper_host.show_ncc_info = false;
	helper_host.fit_plane_hypotheses_cuda = fit_plane_hypotheses_cuda;
	helper_host.weak_reliable_cuda = weak_reliable_cuda;
	helper_host.view_weight_cuda = view_weight_cuda;
	helper_host.weak_nearest_strong = weak_nearest_strong;
	helper_host.edge_cuda = edge_cuda;
	helper_host.edge_neigh_cuda = edge_neigh_cuda;
	helper_host.label_cuda = label_cuda;
	helper_host.label_boundary_cuda = label_boundary_cuda;
	helper_host.candidate_cuda = candidate_cuda;
	helper_host.complex_cuda = complex_cuda;
	helper_host.radius_cuda = radius_cuda;
	// todo 显存优化，仅弱像素使用的包括：label_boundary_cuda, complex_cuda
#ifdef DEBUG_COST_LINE
	helper_host.weak_ncc_cost_cuda = weak_ncc_cost_cuda;
#endif // DEBUG_COST_LINE
	cudaMalloc((void**)(&helper_cuda), sizeof(DataPassHelper));
	cudaMemcpy(helper_cuda, &helper_host, sizeof(DataPassHelper), cudaMemcpyHostToDevice);
}

float4 APD::GetPlaneHypothesis(int r, int c) {
	return plane_hypotheses_host[c + r * width];
}

int APD::GetPixelSelectedViews(int r, int c) {
	return selected_views_host.at<int>(r, c);
}

void APD::SetPixelSelectedViews(int r, int c, int temp_selected_views) {
	selected_views_host.at<int>(r, c) = temp_selected_views;
}

cv::Mat APD::GetEdge() {
	return edge_host;
}

cv::Mat APD::GetPixelStates() {
	return weak_info_host;
}

cv::Mat APD::GetSelectedViews() {
	return selected_views_host;
}

cv::Mat APD::GetRadiusMap() {
	return radius_host;
}

int APD::GetWidth() {
	return width;
}

int APD::GetHeight() {
	return height;
}

float APD::GetDepthMin() {
	return params_host.depth_min;
}

float APD::GetDepthMax() {
	return params_host.depth_max;
}

void RescaleImageAndCamera(cv::Mat& src, cv::Mat& dst, cv::Mat& depth, Camera& camera)
{
	const int cols = depth.cols;
	const int rows = depth.rows;

	if (cols == src.cols && rows == src.rows) {
		dst = src.clone();
		return;
	}

	const float scale_x = cols / static_cast<float>(src.cols);
	const float scale_y = rows / static_cast<float>(src.rows);

	cv::resize(src, dst, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);

	camera.K[0] *= scale_x;
	camera.K[2] *= scale_x;
	camera.K[4] *= scale_y;
	camera.K[5] *= scale_y;
	camera.width = cols;
	camera.height = rows;
}

template <typename TYPE>
void RescaleMatToTargetSize(const cv::Mat& src, cv::Mat& dst, const cv::Size2i& target_size) {
	if (src.cols == target_size.width && src.rows == target_size.height) {
		return;
	}
	const float scale_x = target_size.width / static_cast<float>(src.cols);
	const float scale_y = target_size.height / static_cast<float>(src.rows);

	int type = src.type();
	cv::Mat src_clone = src.clone();
	dst = cv::Mat(target_size.height, target_size.width, type);

	for (int r = 0; r < target_size.height; ++r) {
		for (int c = 0; c < target_size.width; ++c) {
			int o_r = static_cast<int>(r / scale_x);
			int o_c = static_cast<int>(c / scale_y);
			if (o_r < 0 || o_c < 0 || o_r >= src_clone.rows || o_c >= src_clone.cols) {
				continue;
			}
			dst.at<TYPE>(r, c) = src_clone.at<TYPE>(o_r, o_c);
		}
	}
}

float GetAngle(const cv::Vec3f& v1, const cv::Vec3f& v2)
{
	float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	float angle = acosf(dot_product);
	//if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
	if (angle != angle)
		return 0.0f;

	return angle;
}

// ETH version
void RunFusion(const path& dense_folder, const std::vector<Problem>& problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	std::vector<cv::Mat> weaks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	weaks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto& problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);

		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("APD_normals.dmb");
		path weak_path = problem.result_folder / path("weak.bin");
		cv::Mat depth, normal, weak;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);
		ReadBinMat(weak_path, weak);

		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			blocks.emplace_back(block_jpg);
		}

		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
		RescaleMatToTargetSize<uchar>(weak, weak, cv::Size2i(depth.cols, depth.rows));
		weaks.emplace_back(weak);
	}
	std::vector<PointList> PointCloud;
	PointCloud.clear();

	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto& problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				if (masks[ref_index].at<uchar>(r, c) == 1) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				int num_consistent = 0;
				float dynamic_consistency = 0.0f;
				std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);

						if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
							used_list[j].x = src_c;
							used_list[j].y = src_r;
							float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
							dynamic_consistency += exp(-tmp_index);
							num_consistent++;
						}
					}
				}
				float factor = (weaks[ref_index].at<uchar>(r, c) == WEAK ? 0.45f : 0.3f);
				if (num_consistent >= 1 && (dynamic_consistency > factor * num_consistent)) {
					PointList point3D;
					point3D.coord = consistent_Point;
					float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };
					for (int j = 0; j < num_ngb; ++j) {
						if (used_list[j].x == -1)
							continue;
						int src_index = imageIdToindexMap[problem.src_image_ids[j]];
						masks[src_index].at<uchar>(used_list[j].y, used_list[j].x) = 1;
						const auto& color = images[src_index].at<cv::Vec3b>(used_list[j].y, used_list[j].x);
						consistent_Color[0] += color[0];
						consistent_Color[1] += color[1];
						consistent_Color[2] += color[2];
					}
					consistent_Color[0] /= (num_consistent + 1);
					consistent_Color[1] /= (num_consistent + 1);
					consistent_Color[2] /= (num_consistent + 1);
					point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
					PointCloud.emplace_back(point3D);
				}

			}
		}
	}
	path ply_path = dense_folder / path("APD") / path("APD.ply");
	ExportPointCloud(ply_path, PointCloud);
}

void RunFusion_TAT_Intermediate(const path& dense_folder, const std::vector<Problem>& problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");
	const float dist_base = 0.25f;
	const float depth_base = 1.0f / 3500.0f;

	const float angle_base = 0.06981317007977318f; // 4 degree
	const float angle_grad = 0.05235987755982988f; // 3 degree

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto& problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);

		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("APD_normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);

		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			blocks.emplace_back(block_jpg);
		}

		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	struct CostData
	{
		float dist;
		float depth;
		float angle;
		int src_r;
		int src_c;
		bool use;

		CostData() {
			dist = FLT_MAX;
			depth = FLT_MAX;
			angle = FLT_MAX;
		}
	};


	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto& problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		std::vector<CostData> diff(num_ngb, CostData());
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);
						diff[j].dist = reproj_error;
						diff[j].depth = relative_depth_diff;
						diff[j].angle = angle;
						diff[j].src_r = src_r;
						diff[j].src_c = src_c;
					}
				}
				for (int k = 2; k <= num_ngb; ++k) {
					int count = 0;
					for (int j = 0; j < num_ngb; ++j) {
						diff[j].use = false;
						if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base && diff[j].angle < (k * angle_grad + angle_base)) {
							count++;
							diff[j].use = true;
						}
					}
					if (count >= k) {
						PointList point3D;
						float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };
						for (int j = 0; j < num_ngb; ++j) {
							if (diff[j].use) {
								int src_index = imageIdToindexMap[problem.src_image_ids[j]];
								consistent_Color[0] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[0];
								consistent_Color[1] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[1];
								consistent_Color[2] += (float)images[src_index].at<cv::Vec3b>(diff[j].src_r, diff[j].src_c)[2];
							}
						}
						consistent_Color[0] /= (count + 1.0f);
						consistent_Color[1] /= (count + 1.0f);
						consistent_Color[2] /= (count + 1.0f);

						point3D.coord = consistent_Point;
						point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
						PointCloud.emplace_back(point3D);
						masks[ref_index].at<uchar>(r, c) = 1;
						break;
					}
				}
			}
		}
	}
	path ply_path = dense_folder / path("APD") / path("APD.ply");
	ExportPointCloud(ply_path, PointCloud);
}

void RunFusion_TAT_advanced(const path& dense_folder, const std::vector<Problem>& problems)
{
	int num_images = problems.size();
	path image_folder = dense_folder / path("images");
	path cam_folder = dense_folder / path("cams");
	const float dist_base = 0.25f;
	const float depth_base = 1.0f / 3000.0f;

	std::vector<cv::Mat> images;
	std::vector<Camera> cameras;
	std::vector<cv::Mat> depths;
	std::vector<cv::Mat> normals;
	std::vector<cv::Mat> masks;
	std::vector<cv::Mat> blocks;
	images.clear();
	cameras.clear();
	depths.clear();
	normals.clear();
	masks.clear();
	blocks.clear();
	std::unordered_map<int, int> imageIdToindexMap;

	path block_folder = dense_folder / path("blocks");
	bool use_block = false;
	if (exists(block_folder)) {
		use_block = true;
	}

	for (int i = 0; i < num_images; ++i) {
		const auto& problem = problems[i];
		std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		path image_path = image_folder / path(ToFormatIndex(problem.ref_image_id) + ".jpg");
		imageIdToindexMap.emplace(problem.ref_image_id, i);
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
		path cam_path = cam_folder / path(ToFormatIndex(problem.ref_image_id) + "_cam.txt");
		Camera camera;
		ReadCamera(cam_path, camera);

		path depth_path = problem.result_folder / path("depths.dmb");
		path normal_path = problem.result_folder / path("APD_normals.dmb");
		cv::Mat depth, normal;
		ReadBinMat(depth_path, depth);
		ReadBinMat(normal_path, normal);

		if (use_block) {
			path block_path = block_folder / path("mask_" + std::to_string(problem.ref_image_id) + ".jpg");
			cv::Mat block_jpg = cv::imread(block_path.string(), cv::IMREAD_GRAYSCALE);
			blocks.emplace_back(block_jpg);
		}

		cv::Mat scaled_image;
		RescaleImageAndCamera(image, scaled_image, depth, camera);
		images.emplace_back(scaled_image);
		cameras.emplace_back(camera);
		depths.emplace_back(depth);
		normals.emplace_back(normal);
		cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		masks.emplace_back(mask);
	}

	std::vector<PointList> PointCloud;
	PointCloud.clear();

	struct CostData
	{
		float dist;
		float depth;
		float angle;

		CostData() {
			dist = FLT_MAX;
			depth = FLT_MAX;
			angle = FLT_MAX;
		}
	};


	for (int i = 0; i < num_images; ++i) {
		std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
		const auto& problem = problems[i];
		int ref_index = imageIdToindexMap[problem.ref_image_id];
		const int cols = depths[ref_index].cols;
		const int rows = depths[ref_index].rows;
		int num_ngb = problem.src_image_ids.size();
		std::vector<CostData> diff(num_ngb, CostData());
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (use_block && blocks[ref_index].at<uchar>(r, c) < 128) {
					continue;
				}

				float ref_depth = depths[ref_index].at<float>(r, c);
				if (ref_depth <= 0.0)
					continue;
				const cv::Vec3f ref_normal = normals[ref_index].at<cv::Vec3f>(r, c);
				float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[ref_index]);
				float3 consistent_Point = PointX;
				float consistent_Color[3] = { (float)images[ref_index].at<cv::Vec3b>(r, c)[0], (float)images[ref_index].at<cv::Vec3b>(r, c)[1], (float)images[ref_index].at<cv::Vec3b>(r, c)[2] };

				for (int j = 0; j < num_ngb; ++j) {
					int src_index = imageIdToindexMap[problem.src_image_ids[j]];
					const int src_cols = depths[src_index].cols;
					const int src_rows = depths[src_index].rows;
					float2 point;
					float proj_depth;
					ProjectCamera(PointX, cameras[src_index], point, proj_depth);
					int src_r = int(point.y + 0.5f);
					int src_c = int(point.x + 0.5f);
					if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
						if (masks[src_index].at<uchar>(src_r, src_c) == 1)
							continue;
						float src_depth = depths[src_index].at<float>(src_r, src_c);
						if (src_depth <= 0.0)
							continue;
						const cv::Vec3f src_normal = normals[src_index].at<cv::Vec3f>(src_r, src_c);
						float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_index]);
						float2 tmp_pt;
						ProjectCamera(tmp_X, cameras[ref_index], tmp_pt, proj_depth);
						float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
						float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
						float angle = GetAngle(ref_normal, src_normal);
						diff[j].dist = reproj_error;
						diff[j].depth = relative_depth_diff;
						diff[j].angle = angle;
					}
				}
				for (int k = 2; k <= num_ngb; ++k) {
					int count = 0;
					for (int j = 0; j < num_ngb; ++j) {
						if (diff[j].dist < k * dist_base && diff[j].depth < k * depth_base) {
							count++;
						}
					}
					if (count >= k) {
						PointList point3D;
						point3D.coord = consistent_Point;
						point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
						PointCloud.emplace_back(point3D);
						masks[ref_index].at<uchar>(r, c) = 1;
						break;
					}
				}
			}
		}
	}
	path ply_path = dense_folder / path("APD") / path("APD.ply");
	ExportPointCloud(ply_path, PointCloud);
}

void ExportDepthImagePointCloud(
	const path& point_cloud_path,
	const path& image_path,
	const path& cam_path,
	cv::Mat& depth,
	float depth_min,
	float depth_max
) {
	std::vector<PointList> PointCloud;
	PointCloud.clear();

	cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);

	Camera camera;
	ReadCamera(cam_path, camera);

	cv::Mat scaled_image;
	RescaleImageAndCamera(image, scaled_image, depth, camera);

	for (int i = 0; i < depth.cols; i++) {
		for (int j = 0; j < depth.rows; j++) {
			if (depth.at<float>(j, i) < depth_min || depth.at<float>(j, i) > depth_max || isnan(depth.at<float>(j, i))) {
				continue;
			}

			PointList point3D;
			point3D.coord = Get3DPointonWorld(i, j, depth.at<float>(j, i), camera);
			point3D.color = { (float)scaled_image.at<cv::Vec3b>(j, i)[0], (float)scaled_image.at<cv::Vec3b>(j, i)[1], (float)scaled_image.at<cv::Vec3b>(j, i)[2] };
			PointCloud.push_back(point3D);
		}
	}

	ExportPointCloud(point_cloud_path, PointCloud);
}