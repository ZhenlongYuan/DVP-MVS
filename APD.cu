#include "APD.h"

__device__ void matMul3x3(const float* A, const float* B, float* C) {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			C[i * 3 + j] = 0;
			for (int k = 0; k < 3; ++k) {
				C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
			}
		}
	}
}

__device__ void matMul3x1(const float* A, const float* B, float* C) {
	C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
	C[1] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
	C[2] = A[6] * B[0] + A[7] * B[1] + A[7] * B[2];
}

__device__ void matTranspose3x3(const float* A, float* At) {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			At[j * 3 + i] = A[i * 3 + j];
		}
	}
}

__device__ void matSub3x1(float* t, const float* B, const float* R_) {
	for (int i = 0; i < 3; ++i) {
		t[i] = B[i] - R_[i];
	}
}

__device__ void inv3x3(const float* A, float* Ainv) {
	float det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
	float invDet = 1.0f / det;

	Ainv[0] = (A[4] * A[8] - A[5] * A[7]) * invDet;
	Ainv[1] = (A[2] * A[7] - A[1] * A[8]) * invDet;
	Ainv[2] = (A[1] * A[5] - A[2] * A[4]) * invDet;
	Ainv[3] = (A[5] * A[6] - A[3] * A[8]) * invDet;
	Ainv[4] = (A[0] * A[8] - A[2] * A[6]) * invDet;
	Ainv[5] = (A[2] * A[3] - A[0] * A[5]) * invDet;
	Ainv[6] = (A[3] * A[7] - A[4] * A[6]) * invDet;
	Ainv[7] = (A[1] * A[6] - A[0] * A[7]) * invDet;
	Ainv[8] = (A[0] * A[4] - A[1] * A[3]) * invDet;
}

__device__ void crossProduct3x1(const float* a, const float* b, float* c) {
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ void computeEssentialMatrix(const float* R, const float* t, float* E) {
	float t_x[9] = { 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0 };
	matMul3x3(t_x, R, E);
}

__device__ void computeFundamentalMatrix(const float* K, const float* E, float* F) {
	float K_inv[9], Kt_inv[9], tmp[9];

	// ����K�������
	inv3x3(K, K_inv);

	// ����(K_inv).t()
	matTranspose3x3(K_inv, Kt_inv);

	// ���� Kt_inv * E * K_inv
	matMul3x3(Kt_inv, E, tmp);
	matMul3x3(tmp, K_inv, F);
}

__device__ void computeEpipolarLine(const float* F, const float* x, float* l) {
	for (int i = 0; i < 3; ++i) {
		l[i] = 0;
		for (int j = 0; j < 3; ++j) {
			l[i] += F[i * 3 + j] * x[j];
		}
	}
}

__device__  void sort_large_k(float* d, const int n, const int k)
{
	for (int i = 0; i < k; ++i) {
		int maxIdx = i;
		for (int j = i + 1; j < n; ++j) {
			if (d[j] > d[maxIdx]) {
				maxIdx = j;
			}
		}
		float temp = d[i];
		d[i] = d[maxIdx];
		d[maxIdx] = temp;
	}
}

__device__  void sort_small_k(float* d, const int n, const int k)
{
	for (int i = 0; i < k; ++i) {
		int maxIdx = i;
		for (int j = i + 1; j < n; ++j) {
			if (d[j] < d[maxIdx]) {
				maxIdx = j;
			}
		}
		float temp = d[i];
		d[i] = d[maxIdx];
		d[maxIdx] = temp;
	}
}


__device__  void sort_small(float* d, const int n)
{
	int j;
	for (int i = 1; i < n; i++) {
		float tmp = d[i];
		for (j = i; j >= 1 && tmp < d[j - 1]; j--)
			d[j] = d[j - 1];
		d[j] = tmp;
	}
}

__device__ void sort_small_weighted(short2* points, float* w, int n)
{
	int j;
	for (int i = 1; i < n; i++) {
		short2 tmp = points[i];
		float tmp_w = w[i];
		for (j = i; j >= 1 && tmp_w < w[j - 1]; j--) {
			points[j] = points[j - 1];
			w[j] = w[j - 1];
		}
		points[j] = tmp;
		w[j] = tmp_w;
	}
}

__device__ void sort_small_weighted(float3* points, float* w, int n)
{
	int j;
	for (int i = 1; i < n; i++) {
		float3 tmp = points[i];
		float tmp_w = w[i];
		for (j = i; j >= 1 && tmp_w < w[j - 1]; j--) {
			points[j] = points[j - 1];
			w[j] = w[j - 1];
		}
		points[j] = tmp;
		w[j] = tmp_w;
	}
}

__device__ int FindMinCostIndex(const float* costs, const int n)
{
	float min_cost = costs[0];
	int min_cost_idx = 0;
	for (int idx = 1; idx < n; ++idx) {
		if (costs[idx] <= min_cost) {
			min_cost = costs[idx];
			min_cost_idx = idx;
		}
	}
	return min_cost_idx;
}

__device__ int FindMaxCostIndex(const float* costs, const int n)
{
	float max_cost = costs[0];
	int max_cost_idx = 0;
	for (int idx = 1; idx < n; ++idx) {
		if (costs[idx] >= max_cost) {
			max_cost = costs[idx];
			max_cost_idx = idx;
		}
	}
	return max_cost_idx;
}

__device__  void setBit(unsigned int* input, const unsigned int n)
{
	(*input) |= (unsigned int)(1 << n);
}

__device__  void unSetBit(unsigned int* input, const unsigned int n)
{
	(*input) &= (unsigned int)(0xFFFFFFFE << n);
}

__device__  int isSet(unsigned int input, const unsigned int n)
{
	return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4* result)
{
	result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
	result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
	result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ float Vec3DotVec3(const float3 vec1, const float3 vec2)
{
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ float Vec3DotVec3(const float3 vec1, const float4 vec2)
{
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}


__device__ float3 Vec3CrossVec3(const float3 vec1, const float3 vec2)
{
	float3 cross_vec;
	cross_vec.x = vec1.y * vec2.z - vec2.y * vec1.z;
	cross_vec.y = -(vec1.x * vec2.z - vec2.x * vec1.z);
	cross_vec.z = vec1.x * vec2.y - vec2.x * vec1.y;
	return cross_vec;
}

__device__ float Vec2DotVec2(float2 a, float2 b) {
	return a.x * b.x + a.y * b.y;
}

__device__ float Vec2CrossVec2(float2 a, float2 b) {
	return a.x * b.y - a.y * b.x;
}

__device__ float ModuleVec3(float3 vec) {
	return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ float3 ScalarMultiVec3(float scalar, float3 vec) {
	return make_float3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
}

__device__ bool PointinTriangle(short2 A, short2 B, short2 C, int2 P)
{
	float2 AB = make_float2(B.x - A.x, B.y - A.y);
	float2 BC = make_float2(C.x - B.x, C.y - B.y);
	float2 CA = make_float2(A.x - C.x, A.y - C.y);
	float AB_ = sqrt(AB.x * AB.x + AB.y * AB.y);
	float BC_ = sqrt(BC.x * BC.x + BC.y * BC.y);
	float CA_ = sqrt(CA.x * CA.x + CA.y * CA.y);
	if (AB_ <= 2 || BC_ <= 2 || CA_ <= 2) {
		return false;
	}
	if (!(AB_ + BC_ > CA_ && BC_ + CA_ > AB_ && AB_ + CA_ > BC_)) {
		return false;
	}
	float2 PA = make_float2(A.x - P.x, A.y - P.y);
	float2 PB = make_float2(B.x - P.x, B.y - P.y);
	float2 PC = make_float2(C.x - P.x, C.y - P.y);
	float t1 = Vec2CrossVec2(PA, PB);
	float t2 = Vec2CrossVec2(PB, PC);
	float t3 = Vec2CrossVec2(PC, PA);
	return t1 * t2 >= 0 && t1 * t3 >= 0;
}

__device__ bool BresenhamLine(int2 A, int2 B, const DataPassHelper* helper) {
	uchar* edge = helper->edge_cuda;
	int height = helper->height;
	int width = helper->width;
	int max_step = (int)(MAX(height, width) / 30.0);
	int x0 = B.x, y0 = B.y, x1 = A.x, y1 = A.y;

	int ABx = A.x - B.x, ABy = A.y - B.y;
	if (ABx * ABx + ABy * ABy > 9 * max_step * max_step)
		return false;

	if (edge[x0 + y0 * width] || edge[x1 + y1 * width])
		return false;

	int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
	int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
	int erro = (dx > dy ? dx : dy) / 2;
	int step = 0;

	bool tagx = true, tagy = true;
	while (tagx || tagy) {
		if (x0 == x1) tagx = false;
		if (y0 == y1) tagy = false;
		int e2 = erro;
		if (e2 > -dx) {
			erro -= dy;
			x0 += sx;
		}
		if (e2 < dy) {
			erro += dx;
			y0 += sy;
		}

		// if line intersect with edge
		if (edge[x0 + y0 * width])
			return true;

		step += 1;

		if (step >= max_step)
			break;
	}

	return false;
}

__device__ bool BresenhamLine(short2 A, short2 B, const DataPassHelper* helper) {
	int2 iA = make_int2(A.x, A.y);
	int2 iB = make_int2(B.x, B.y);
	return BresenhamLine(iA, iB, helper);
}

__device__ float TriangleArea(float3 A, float3 B, float3 C)
{
	float3 AB = make_float3(B.x - A.x, B.y - A.y, B.z - A.z);
	float3 BC = make_float3(C.x - B.x, C.y - B.y, C.z - B.z);
	float3 CA = make_float3(A.x - C.x, A.y - C.y, A.z - C.z);
	float AB_ = sqrt(AB.x * AB.x + AB.y * AB.y + AB.z * AB.z);
	float BC_ = sqrt(BC.x * BC.x + BC.y * BC.y + BC.z * BC.z);
	float CA_ = sqrt(CA.x * CA.x + CA.y * CA.y + CA.z * CA.z);
	float P = (AB_ + BC_ + CA_) / 2.0f;
	return sqrt(P * (P - AB_) * (P - BC_) * (P - CA_));
}

__device__ void NormalizeVec3(float4* vec)
{
	const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
	const float inverse_sqrt = rsqrtf(normSquared);
	vec->x *= inverse_sqrt;
	vec->y *= inverse_sqrt;
	vec->z *= inverse_sqrt;
}

__device__ void NormalizeVec3(float3* vec) {
	const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
	const float inverse_sqrt = rsqrtf(normSquared);
	vec->x *= inverse_sqrt;
	vec->y *= inverse_sqrt;
	vec->z *= inverse_sqrt;
}

__device__ void NormalizeVec2(float2* vec)
{
	const float normSquared = vec->x * vec->x + vec->y * vec->y;
	const float inverse_sqrt = rsqrtf(normSquared);
	vec->x *= inverse_sqrt;
	vec->y *= inverse_sqrt;
}

__device__ void TransformPDFToCDF(float* probs, const int num_probs)
{
	float prob_sum = 0.0f;
	for (int i = 0; i < num_probs; ++i) {
		prob_sum += probs[i];
	}
	const float inv_prob_sum = 1.0f / prob_sum;

	float cum_prob = 0.0f;
	for (int i = 0; i < num_probs; ++i) {
		const float prob = probs[i] * inv_prob_sum;
		cum_prob += prob;
		probs[i] = cum_prob;
	}
}

__device__ void Get3DPoint(const Camera camera, const int2 p, const float depth, float* X)
{
	X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
	X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
	X[2] = depth;
}

__device__ void Get3DPoint(const Camera camera, const short2 p, const float depth, float* X)
{
	X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
	X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
	X[2] = depth;
}

__device__ float4 GetViewDirection(const Camera camera, const int2 p, const float depth)
{
	float X[3];
	Get3DPoint(camera, p, depth, X);
	float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

	float4 view_direction;
	view_direction.x = X[0] / norm;
	view_direction.y = X[1] / norm;
	view_direction.z = X[2] / norm;
	view_direction.w = 0;
	return view_direction;
}

__device__ float GetDistance2Origin(const Camera camera, const int2 p, const float depth, const float4 normal)
{
	float X[3];
	Get3DPoint(camera, p, depth, X);
	return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

__device__   float SpatialGauss(float x1, float y1, float x2, float y2, float sigma, float mu = 0.0)
{
	float dis = pow(x1 - x2, 2) + pow(y1 - y2, 2) - mu;
	return exp(-1.0 * dis / (2 * sigma * sigma));
}

__device__  float RangeGauss(float x, float sigma, float mu = 0.0)
{
	float x_p = x - mu;
	return exp(-1.0 * (x_p * x_p) / (2 * sigma * sigma));
}

__device__ float ComputeDepthfromPlaneHypothesis(const Camera camera, const float4 plane_hypothesis, const int2 p)
{
	return -plane_hypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * plane_hypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y + camera.K[0] * plane_hypothesis.z);
}

__device__ float determinant3x3(const float mat[9]) {
	return mat[0] * (mat[4] * mat[8] - mat[7] * mat[5]) -
		mat[1] * (mat[3] * mat[8] - mat[6] * mat[5]) +
		mat[2] * (mat[3] * mat[7] - mat[6] * mat[4]);
}

__device__ void inverse3x3(const float mat[9], float invMat[9]) {
	float det = determinant3x3(mat);
	assert(fabs(det) > 1e-5);  // ȷ������ʽ��Ϊ0

	invMat[0] = (mat[4] * mat[8] - mat[7] * mat[5]) / det;
	invMat[1] = -(mat[1] * mat[8] - mat[7] * mat[2]) / det;
	invMat[2] = (mat[1] * mat[5] - mat[4] * mat[2]) / det;
	invMat[3] = -(mat[3] * mat[8] - mat[6] * mat[5]) / det;
	invMat[4] = (mat[0] * mat[8] - mat[6] * mat[2]) / det;
	invMat[5] = -(mat[0] * mat[5] - mat[3] * mat[2]) / det;
	invMat[6] = (mat[3] * mat[7] - mat[6] * mat[4]) / det;
	invMat[7] = -(mat[0] * mat[7] - mat[6] * mat[1]) / det;
	invMat[8] = (mat[0] * mat[4] - mat[3] * mat[1]) / det;
}

__device__ float GetDepthFrom3DPoint(const float x, const float y, const float3 pointX, const Camera camera) {
	// ��ʱ����
	float3 tmpX;

	// ����ƽ�Ʊ任
	tmpX.x = pointX.x - camera.c[0];
	tmpX.y = pointX.y - camera.c[1];
	tmpX.z = pointX.z - camera.c[2];

	// ���� R �������
	float invR[9];
	inverse3x3(camera.R, invR);

	// ������ת�任
	float3 pointCam;
	pointCam.x = invR[0] * tmpX.x + invR[1] * tmpX.y + invR[2] * tmpX.z;
	pointCam.y = invR[3] * tmpX.x + invR[4] * tmpX.y + invR[5] * tmpX.z;
	pointCam.z = invR[6] * tmpX.x + invR[7] * tmpX.y + invR[8] * tmpX.z;

	return pointCam.z;
}

__device__ float3 Get3DPointonWorld_cu(const float x, const float y, const float depth, const Camera camera)
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
	pointX.x = tmpX.x + camera.c[0];
	pointX.y = tmpX.y + camera.c[1];
	pointX.z = tmpX.z + camera.c[2];

	return pointX;
}

__device__ void ProjectonCamera_cu(const float3 PointX, const Camera camera, float2& point, float& depth)
{
	float3 tmp;
	tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
	tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
	tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

	depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
	point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
	point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

__device__ float4 GenerateRandomNormal_YZL(DataPassHelper* helper, const Camera camera, const int2 p, curandState* rand_state, const float depth)
{
	//2. ���ټ������
	//3. ��
	int width = helper->width;
	int height = helper->height;
	Camera* cameras = helper->cameras_cuda;
	int center = p.y * width + p.x;
	unsigned int* selected_views = helper->selected_views_cuda;
	PatchMatchParams* params = helper->params;
	float4 view_direction[20] = { 0 };
	view_direction[0] = GetViewDirection(camera, p, depth);
	int index = 1;
	for (int src_idx = 1; src_idx < helper->params->num_images; ++src_idx) {
		const Camera ref_camera = helper->cameras_cuda[0];
		const Camera src_camera = helper->cameras_cuda[src_idx];
		const cudaTextureObject_t depth_image = helper->texture_depths_cuda[0].images[src_idx];
		if (isSet(selected_views[center], src_idx - 1) == 1) {
			float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);

			float2 src_pt;
			float src_d;
			ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);

			int2 src_pt_int = make_int2((int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);
			float src_depth;

			if (params->geom_consistency) {
				if (src_pt_int.x >= 0 && src_pt_int.x < width && src_pt_int.y >= 0 && src_pt_int.y < height)
					src_depth = tex2D<float>(depth_image, (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);
			}
			else {
				src_depth = 1.0f;
			}

			//view_direction[index++] = GetViewDirection(cameras[src_idx], src_pt_int, src_depth);

			//src����ϵת��Ϊref����ϵ
			float4 direction = GetViewDirection(cameras[src_idx], src_pt_int, src_depth);
			float R_t[9], R_c[9], R_f[3];
			matTranspose3x3(cameras[src_idx].R, R_t);
			matMul3x3(cameras[0].R, R_t, R_c);
			float dir[3] = { direction.x, direction.y, direction.x };
			matMul3x1(R_c, dir, R_f);
			float norm = sqrt(R_f[0] * R_f[0] + R_f[1] * R_f[1] + R_f[2] * R_f[2]);
			float4 view_direction_this;
			view_direction_this.x = R_f[0] / norm;
			view_direction_this.y = R_f[1] / norm;
			view_direction_this.z = R_f[2] / norm;
			view_direction_this.w = 0;
			view_direction[index++] = view_direction_this;
		}
	}

	int times = 200;
	float4 normal;
	while (times > 0) { //100
		float q1 = 1.0f;
		float q2 = 1.0f;
		float s = 2.0f;
		while (s >= 1.0f) {
			q1 = 2.0f * curand_uniform(rand_state) - 1.0f;
			q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
			s = q1 * q1 + q2 * q2;
		}
		const float sq = sqrt(1.0f - s);
		normal.x = 2.0f * q1 * sq;
		normal.y = 2.0f * q2 * sq;
		normal.z = 1.0f - 2.0f * s;
		normal.w = 0;

		bool satisfy = true;
		for (int i = 0; i < index; i++) {
			float dir = normal.x * view_direction[i].x + normal.y * view_direction[i].y + normal.z * view_direction[i].z;
			if (dir > 0.0f) {
				satisfy = false;
				break;
			}
		}
		if (satisfy)
			break;
		else
			times--;
	}
	NormalizeVec3(&normal);

	return normal;
}

__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, curandState* rand_state, const float depth)
{
	//float4 normal;
	//float q1 = 1.0f;
	//float q2 = 1.0f;
	//float s = 2.0f;
	//while (s >= 1.0f) {
	//	q1 = 2.0f * curand_uniform(rand_state) - 1.0f;
	//	q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
	//	s = q1 * q1 + q2 * q2;
	//}
	//const float sq = sqrt(1.0f - s);
	//normal.x = 2.0f * q1 * sq;
	//normal.y = 2.0f * q2 * sq;
	//normal.z = 1.0f - 2.0f * s;
	//normal.w = 0;
	//float4 view_direction = GetViewDirection(camera, p, depth);
	//float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
	//if (dot_product > 0.0f) {
	//	normal.x = -normal.x;
	//	normal.y = -normal.y;
	//	normal.z = -normal.z;
	//}
	//NormalizeVec3(&normal);
	//return normal;
}

__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, curandState* rand_state, const float perturbation, float depth)
{
	float4 view_direction = GetViewDirection(camera, p, 1.0f);
	float4 normal_perturbed;

	int times = 200;
	while (times > 0) {
		const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
		const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
		const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

		const float sin_a1 = sin(a1);
		const float sin_a2 = sin(a2);
		const float sin_a3 = sin(a3);
		const float cos_a1 = cos(a1);
		const float cos_a2 = cos(a2);
		const float cos_a3 = cos(a3);

		float R[9];
		R[0] = cos_a2 * cos_a3;
		R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
		R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
		R[3] = cos_a2 * sin_a3;
		R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
		R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
		R[6] = -sin_a2;
		R[7] = cos_a2 * sin_a1;
		R[8] = cos_a1 * cos_a2;


		Mat33DotVec3(R, normal, &normal_perturbed);

		if (Vec3DotVec3(normal_perturbed, view_direction) < 0.0f) {
			normal_perturbed = normal;
			break;
		}
		else {
			normal_perturbed = normal;
			times--;
		}
	}

	NormalizeVec3(&normal_perturbed);
	return normal_perturbed;
}

__device__ float4 GenerateRandomPlaneHypothesis_YZL(DataPassHelper* helper, const Camera camera, const int2 p, curandState* rand_state, const float depth_min, const float depth_max)
{
	float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
	float4 plane_hypothesis = GenerateRandomNormal_YZL(helper, camera, p, rand_state, depth);
	plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
	return plane_hypothesis;
}

__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState* rand_state, const float depth_min, const float depth_max)
{
	float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
	float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, depth);
	plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
	return plane_hypothesis;
}

__device__ void ComputeHomography(const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, float* H)
{
	float ref_C[3];
	float src_C[3];
	ref_C[0] = -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]);
	ref_C[1] = -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]);
	ref_C[2] = -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2]);
	src_C[0] = -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]);
	src_C[1] = -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]);
	src_C[2] = -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2]);

	float R_relative[9];
	float C_relative[3];
	float t_relative[3];
	R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] * ref_camera.R[2];
	R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] * ref_camera.R[5];
	R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] * ref_camera.R[8];
	R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] * ref_camera.R[2];
	R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] * ref_camera.R[5];
	R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] * ref_camera.R[8];
	R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] * ref_camera.R[2];
	R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] * ref_camera.R[5];
	R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] * ref_camera.R[8];
	C_relative[0] = (ref_C[0] - src_C[0]);
	C_relative[1] = (ref_C[1] - src_C[1]);
	C_relative[2] = (ref_C[2] - src_C[2]);
	t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
	t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
	t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

	H[0] = R_relative[0] - t_relative[0] * plane_hypothesis.x / plane_hypothesis.w;
	H[1] = R_relative[1] - t_relative[0] * plane_hypothesis.y / plane_hypothesis.w;
	H[2] = R_relative[2] - t_relative[0] * plane_hypothesis.z / plane_hypothesis.w;
	H[3] = R_relative[3] - t_relative[1] * plane_hypothesis.x / plane_hypothesis.w;
	H[4] = R_relative[4] - t_relative[1] * plane_hypothesis.y / plane_hypothesis.w;
	H[5] = R_relative[5] - t_relative[1] * plane_hypothesis.z / plane_hypothesis.w;
	H[6] = R_relative[6] - t_relative[2] * plane_hypothesis.x / plane_hypothesis.w;
	H[7] = R_relative[7] - t_relative[2] * plane_hypothesis.y / plane_hypothesis.w;
	H[8] = R_relative[8] - t_relative[2] * plane_hypothesis.z / plane_hypothesis.w;

	float tmp[9];
	tmp[0] = H[0] / ref_camera.K[0];
	tmp[1] = H[1] / ref_camera.K[4];
	tmp[2] = -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[5] / ref_camera.K[4] + H[2];
	tmp[3] = H[3] / ref_camera.K[0];
	tmp[4] = H[4] / ref_camera.K[4];
	tmp[5] = -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[5] / ref_camera.K[4] + H[5];
	tmp[6] = H[6] / ref_camera.K[0];
	tmp[7] = H[7] / ref_camera.K[4];
	tmp[8] = -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[5] / ref_camera.K[4] + H[8];

	H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
	H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
	H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
	H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
	H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
	H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];
	H[6] = src_camera.K[8] * tmp[6];
	H[7] = src_camera.K[8] * tmp[7];
	H[8] = src_camera.K[8] * tmp[8];
}

__device__ float2 ComputeCorrespondingPoint(const float* H, const int2 p)
{
	float3 pt;
	pt.x = H[0] * p.x + H[1] * p.y + H[2];
	pt.y = H[3] * p.x + H[4] * p.y + H[5];
	pt.z = H[6] * p.x + H[7] * p.y + H[8];
	return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis)
{
	float4 transformed_normal;
	transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
	transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
	transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
	transformed_normal.w = plane_hypothesis.w;
	return transformed_normal;
}

__device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis)
{
	float4 transformed_normal;
	transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
	transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
	transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
	transformed_normal.w = plane_hypothesis.w;
	return transformed_normal;
}

__device__ short2 GetNeighbourPoint(const int2 p, const int index, const DataPassHelper* helper) {
	const unsigned offset = helper->neighbours_map_cuda[p.x + p.y * helper->width] * NEIGHBOUR_NUM;
	short2 neighbour_pt = helper->neighbours_cuda[offset + index];
	return neighbour_pt;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
	const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
	const float color_dist = fabs(pix - center_pix);
	return exp(-spatial_dist / (2.0f * sigma_spatial * sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}

__device__ float ComputeBilateralWeight_YZL(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
	const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
	const float color_dist = fabs(pix - center_pix);
	return exp(-color_dist / (2.0f * sigma_color * sigma_color));
}

struct Point {
	int i, j;
	double angle;
	double weight;
};

// ����Ƕ�
__device__ double calculateAngle(int i, int j) {
	double angle = atan2((double)j, (double)i);
	double angleInDegrees = angle * (180.0 / M_PI);
	if (angleInDegrees < 0) {
		angleInDegrees += 360.0;
	}
	return angleInDegrees;
}

// ������ൽ����
__device__ int getRegion(double angle) {
	if (angle >= 0 && angle < 30) return 0;
	if (angle >= 30 && angle < 60) return 1;
	if (angle >= 60 && angle < 90) return 2;
	if (angle >= 90 && angle < 120) return 3;
	if (angle >= 120 && angle < 150) return 4;
	if (angle >= 150 && angle < 180) return 5;
	if (angle >= 180 && angle < 210) return 6;
	if (angle >= 210 && angle < 240) return 7;
	if (angle >= 240 && angle < 270) return 8;
	if (angle >= 270 && angle < 300) return 9;
	if (angle >= 300 && angle < 330) return 10;
	if (angle >= 330 && angle < 360) return 11;
	return -1; // �������
}

__device__ void bubbleSort(Point* points, int n) {
	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j < n - 1 - i; ++j) {
			if (points[j].weight < points[j + 1].weight) {
				Point temp = points[j];
				points[j] = points[j + 1];
				points[j + 1] = temp;
			}
		}
	}
}

__device__ float ComputeBilateralNCCNew(
	const int2 p,
	const int src_idx,
	const float4 plane_hypothesis,
	const DataPassHelper* helper
) {
	const cudaTextureObject_t ref_image = helper->texture_objects_cuda[0].images[0];
	const Camera ref_camera = helper->cameras_cuda[0];
	const cudaTextureObject_t src_image = helper->texture_objects_cuda[0].images[src_idx];
	const Camera src_camera = helper->cameras_cuda[src_idx];
	const PatchMatchParams* params = helper->params;
	const uchar* weak_info = helper->weak_info_cuda;
	const int width = helper->width;
	const int height = helper->height;
	const int center = p.x + p.y * width;
	unsigned int* selected_views = helper->selected_views_cuda;
	short2* candidate = helper->candidate_cuda;
	const float cost_max = 2.0f;

	float H[9];
	ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
	float2 pt = ComputeCorrespondingPoint(H, p);
	if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
		return cost_max;
	}

	float cost = 0.0f;
	if (weak_info[center] == WEAK) {
		// for weak texture area use deformable ncc
		const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
		// the strong points
		float center_cost = 0.0f;
		float strong_cost = 0.0f;
		int strong_count = 0;
		for (int k = 0; k < NEIGHBOUR_NUM; ++k) {
			const short2 neighbour_pt = GetNeighbourPoint(p, k, helper);
			if (neighbour_pt.x == -1 || neighbour_pt.y == -1) {
				continue;
			}
			float2 neighbour_src_pt = ComputeCorrespondingPoint(H, make_int2(neighbour_pt.x, neighbour_pt.y));
			if (neighbour_src_pt.x < 0 || neighbour_src_pt.y < 0 || neighbour_src_pt.x >= width || neighbour_src_pt.y >= height) {
				if (k != 0) {
					unsigned int view_info = helper->selected_views_cuda[neighbour_pt.x + neighbour_pt.y * width];
					if (isSet(view_info, src_idx - 1)) {
						strong_cost += cost_max;
						strong_count++;
					}
					continue;
				}
				else {
					return cost_max;
				}
			}
			// compute ncc for this point
			float sum_ref = 0.0f;
			float sum_ref_ref = 0.0f;
			float sum_src = 0.0f;
			float sum_src_src = 0.0f;
			float sum_ref_src = 0.0f;
			float bilateral_weight_sum = 0.0f;
			const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
			int radius = (k == 0 ? helper->params->strong_radius : params->weak_radius);
			int increment = (k == 0 ? params->strong_increment : params->weak_increment);

			if (helper->params->use_radius && k == 0) {
				radius = helper->radius_cuda[p.x + p.y * helper->width];
				increment = MAX(2, (int)(2.0 * radius / 5.0));
			}
			if (k == 0) {
				for (int i = -radius; i <= radius; i += increment) {
					float sum_ref_row = 0.0f;
					float sum_src_row = 0.0f;
					float sum_ref_ref_row = 0.0f;
					float sum_src_src_row = 0.0f;
					float sum_ref_src_row = 0.0f;
					float bilateral_weight_sum_row = 0.0f;
					for (int j = -radius; j <= radius; j += increment) {
						const int2 ref_pt = make_int2(neighbour_pt.x + i, neighbour_pt.y + j);
						const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
						float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
						const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
						float weight = ComputeBilateralWeight_YZL(i, j, ref_pix, ref_center_pix, helper->params->sigma_spatial, helper->params->sigma_color);
						sum_ref_row += weight * ref_pix;
						sum_ref_ref_row += weight * ref_pix * ref_pix;
						sum_src_row += weight * src_pix;
						sum_src_src_row += weight * src_pix * src_pix;
						sum_ref_src_row += weight * ref_pix * src_pix;
						bilateral_weight_sum_row += weight;
					}
					sum_ref += sum_ref_row;
					sum_ref_ref += sum_ref_ref_row;
					sum_src += sum_src_row;
					sum_src_src += sum_src_src_row;
					sum_ref_src += sum_ref_src_row;
					bilateral_weight_sum += bilateral_weight_sum_row;
				}
			}
			else {
				//if (true) {
				if (isSet(selected_views[neighbour_pt.x + neighbour_pt.y * width], src_idx - 1) == 1) { //�ɼ��ټ�
					int nei_center = neighbour_pt.x + neighbour_pt.y * width;
					for (int k = 0; k < 9; k++) { //ÿ���ھӶ��Ӹ��ھӵ����9��
						int i = 0;
						int j = 0;
						if (k != 8) {
							i = candidate[nei_center * LAB_BOUNDARY_NUM * NUM_IMAGES + (src_idx - 1) * LAB_BOUNDARY_NUM + k].x;
							j = candidate[nei_center * LAB_BOUNDARY_NUM * NUM_IMAGES + (src_idx - 1) * LAB_BOUNDARY_NUM + k].y;
						}
						if (i == 0 && j == 0) {
							if (k == 0) { i = -5; j = -5; }
							else if (k == 1) { i = -5; j = 0; }
							else if (k == 2) { i = -5; j = 5; }
							else if (k == 3) { i = 0; j = -5; }
							else if (k == 4) { i = 0; j = 5; }
							else if (k == 5) { i = 5; j = -5; }
							else if (k == 6) { i = 5; j = 0; }
							else if (k == 7) { i = 5; j = 5; }
						}
						float sum_ref_row = 0.0f;
						float sum_src_row = 0.0f;
						float sum_ref_ref_row = 0.0f;
						float sum_src_src_row = 0.0f;
						float sum_ref_src_row = 0.0f;
						float bilateral_weight_sum_row = 0.0f;

						const int2 ref_pt = make_int2(neighbour_pt.x + i, neighbour_pt.y + j);
						const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
						float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
						const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
						float weight = ComputeBilateralWeight_YZL(i, j, ref_pix, ref_center_pix, helper->params->sigma_spatial, helper->params->sigma_color);
						sum_ref_row += weight * ref_pix;
						sum_ref_ref_row += weight * ref_pix * ref_pix;
						sum_src_row += weight * src_pix;
						sum_src_src_row += weight * src_pix * src_pix;
						sum_ref_src_row += weight * ref_pix * src_pix;
						bilateral_weight_sum_row += weight;
						sum_ref += sum_ref_row;
						sum_ref_ref += sum_ref_ref_row;
						sum_src += sum_src_row;
						sum_src_src += sum_src_src_row;
						sum_ref_src += sum_ref_src_row;
						bilateral_weight_sum += bilateral_weight_sum_row;
					}
				}

			}
			const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
			sum_ref *= inv_bilateral_weight_sum;
			sum_ref_ref *= inv_bilateral_weight_sum;
			sum_src *= inv_bilateral_weight_sum;
			sum_src_src *= inv_bilateral_weight_sum;
			sum_ref_src *= inv_bilateral_weight_sum;
			const float var_ref = sum_ref_ref - sum_ref * sum_ref;
			const float var_src = sum_src_src - sum_src * sum_src;
			const float kMinVar = 1e-5f;
			float temp_cost = 0.0f;
			if (var_ref < kMinVar || var_src < kMinVar) {
				temp_cost = cost_max;
			}
			else {
				const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
				const float var_ref_src = sqrt(var_ref * var_src);
				temp_cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
			}
			if (k == 0) {
				center_cost = temp_cost;
			}
			else {
				strong_cost += temp_cost;
				strong_count++;
			}
		}
		if (strong_count == 0) {
			cost = center_cost;
		}
		else {
			strong_cost /= strong_count;
			strong_cost = MIN(strong_cost, cost_max);
			cost = 0.25 * center_cost + 0.75 * strong_cost;
		}
	}
	else {
		printf("error\n");
	}

	return cost;
}

__device__ float ComputeBilateralNCCOld(
	const int2 p,
	const int src_idx,
	const float4 plane_hypothesis,
	const DataPassHelper* helper
) {
	const cudaTextureObject_t ref_image = helper->texture_objects_cuda[0].images[0];
	const Camera ref_camera = helper->cameras_cuda[0];
	const cudaTextureObject_t src_image = helper->texture_objects_cuda[0].images[src_idx];
	const Camera src_camera = helper->cameras_cuda[src_idx];

	const float cost_max = 2.0f;

	float H[9];
	ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
	float2 pt = ComputeCorrespondingPoint(H, p);
	if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
		return cost_max;
	}
	int radius = helper->params->strong_radius;
	int increment = helper->params->strong_increment;
	if (helper->params->use_radius) {
		radius = helper->radius_cuda[p.x + p.y * helper->width];
		increment = MAX(2, (int)(2.0 * radius / 5.0));
	}

	float cost = 0.0f;
	{
		float sum_ref = 0.0f;
		float sum_ref_ref = 0.0f;
		float sum_src = 0.0f;
		float sum_src_src = 0.0f;
		float sum_ref_src = 0.0f;
		float bilateral_weight_sum = 0.0f;
		const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

		for (int i = -radius; i <= radius; i += increment) {
			float sum_ref_row = 0.0f;
			float sum_src_row = 0.0f;
			float sum_ref_ref_row = 0.0f;
			float sum_src_src_row = 0.0f;
			float sum_ref_src_row = 0.0f;
			float bilateral_weight_sum_row = 0.0f;

			for (int j = -radius; j <= radius; j += increment) {
				const int2 ref_pt = make_int2(p.x + i, p.y + j);
				const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
				float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
				const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

				float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, helper->params->sigma_spatial, helper->params->sigma_color);

				sum_ref_row += weight * ref_pix;
				sum_ref_ref_row += weight * ref_pix * ref_pix;
				sum_src_row += weight * src_pix;
				sum_src_src_row += weight * src_pix * src_pix;
				sum_ref_src_row += weight * ref_pix * src_pix;
				bilateral_weight_sum_row += weight;
			}

			sum_ref += sum_ref_row;
			sum_ref_ref += sum_ref_ref_row;
			sum_src += sum_src_row;
			sum_src_src += sum_src_src_row;
			sum_ref_src += sum_ref_src_row;
			bilateral_weight_sum += bilateral_weight_sum_row;
		}

		const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
		sum_ref *= inv_bilateral_weight_sum;
		sum_ref_ref *= inv_bilateral_weight_sum;
		sum_src *= inv_bilateral_weight_sum;
		sum_src_src *= inv_bilateral_weight_sum;
		sum_ref_src *= inv_bilateral_weight_sum;

		const float var_ref = sum_ref_ref - sum_ref * sum_ref;
		const float var_src = sum_src_src - sum_src * sum_src;

		const float kMinVar = 1e-5f;
		if (var_ref < kMinVar || var_src < kMinVar) {
			cost = cost_max;
		}
		else {
			const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
			const float var_ref_src = sqrt(var_ref * var_src);
			cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
		}
	}

	return cost;
}

__device__ float ComputeMultiViewInitialCostandSelectedViews(
	const int2 p,
	DataPassHelper* helper
) {
	PatchMatchParams* params = helper->params;
	unsigned int* selected_views = helper->selected_views_cuda;
	int center = p.x + p.y * helper->width;
	float4 plane_hypothesis = helper->plane_hypotheses_cuda[center];

	float cost_max = 2.0f;
	float cost_vector[32] = { 2.0f };
	float cost_vector_copy[32] = { 2.0f };
	int cost_count = 0;
	int num_valid_views = 0;

	for (int i = 1; i < params->num_images; ++i) {
		float c = 0.0f;
		c = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
		cost_vector[i - 1] = c;
		cost_vector_copy[i - 1] = c;
		cost_count++;
		if (c < cost_max) {
			num_valid_views++;
		}
	}

	sort_small(cost_vector, cost_count);
	selected_views[center] = 0;

	int top_k = min(num_valid_views, params->top_k);
	if (top_k > 0) {
		float cost = 0.0f;
		for (int i = 0; i < top_k; ++i) {
			cost += cost_vector[i];
		}
		float cost_threshold = cost_vector[top_k - 1];
		for (int i = 0; i < params->num_images - 1; ++i) {
			if (cost_vector_copy[i] <= cost_threshold) {
				setBit(&(selected_views[center]), i);
			}
		}
		return cost / top_k;
	}
	else {
		return cost_max;
	}
}

__device__ float ComputeMultiViewInitialCost(
	const int2 p,
	DataPassHelper* helper
) {
	PatchMatchParams* params = helper->params;
	unsigned int* selected_views = helper->selected_views_cuda;
	int center = p.x + p.y * helper->width;
	float4 plane_hypothesis = helper->plane_hypotheses_cuda[center];

	const float cost_max = 2.0f;
	int cost_count = 0;
	float cost = 0.0f;

	for (int i = 1; i < params->num_images; ++i) {
		if (isSet(selected_views[center], i - 1)) {
			float c = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
			if (c < cost_max) {
				cost_count++;
				cost += c;
			}
			else {
				unSetBit(&(selected_views[center]), i - 1);
			}
		}
	}
	if (cost_count == 0) {
		return cost_max;
	}
	else {
		return cost / cost_count;
	}
}

__device__ void ComputeMultiViewCostVectorNew(
	const int2 p,
	float4 plane_hypothesis,
	float* cost_vector,
	DataPassHelper* helper
) {
	for (int i = 1; i < helper->params->num_images; ++i) {
		cost_vector[i - 1] = ComputeBilateralNCCNew(p, i, plane_hypothesis, helper);
	}
}

__device__ void ComputeMultiViewCostVectorOld(
	const int2 p,
	float4 plane_hypothesis,
	float* cost_vector,
	DataPassHelper* helper
) {
	for (int i = 1; i < helper->params->num_images; ++i) {
		cost_vector[i - 1] = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
	}
}

__device__ float ComputeGeomConsistencyCost(
	const int2 p,
	const int src_idx,
	const float4 plane_hypothesis,
	DataPassHelper* helper
) {
	const Camera ref_camera = helper->cameras_cuda[0];
	const Camera src_camera = helper->cameras_cuda[src_idx];
	const cudaTextureObject_t depth_image = helper->texture_depths_cuda[0].images[src_idx];

	const float max_cost = 3.0f;

	float center_cost = 0.0f;
	{
		float depth = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
		float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);

		float2 src_pt;
		float src_d;
		ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);

		const float src_depth = tex2D<float>(depth_image, (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);

		if (src_depth == 0.0f) {
			return max_cost;
		}

		float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);

		float2 backward_point;
		float ref_d;
		ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);

		const float diff_col = p.x - backward_point.x;
		const float diff_row = p.y - backward_point.y;
		center_cost = sqrt(diff_col * diff_col + diff_row * diff_row);
	}
	return min(max_cost, center_cost);
}

__global__ void InitRandomStates(
	DataPassHelper* helper
) {
	const int width = helper->width;
	const int height = helper->height;
	curandState* rand_states = helper->rand_states_cuda;

	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	curand_init(clock64(), p.y, p.x, &rand_states[center]);
}

__global__ void RandomInitialization(
	DataPassHelper* helper
) {
	int width = helper->width;
	int height = helper->height;
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	Camera* cameras = helper->cameras_cuda;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	curandState* rand_states = helper->rand_states_cuda;
	PatchMatchParams* params = helper->params;

	if (params->state == FIRST_INIT) {
		if (plane_hypotheses[center].w > params->depth_max || plane_hypotheses[center].w < params->depth_min)
			plane_hypotheses[center] = GenerateRandomPlaneHypothesis_YZL(helper, cameras[0], p, &rand_states[center], params->depth_min, params->depth_max);
		else {
			/*plane_hypotheses_temp.w = plane_hypotheses[center].w;
			plane_hypotheses[center] = plane_hypotheses_temp;*/
		}
		//plane_hypotheses[center] = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], params->depth_min, params->depth_max);

		costs[center] = ComputeMultiViewInitialCostandSelectedViews(p, helper);
	}
	else {
		float4 plane_hypothesis;
		plane_hypothesis = plane_hypotheses[center];
		plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
		float depth = plane_hypothesis.w;
		plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
		plane_hypotheses[center] = plane_hypothesis;
		costs[center] = ComputeMultiViewInitialCost(p, helper);
	}
}

__device__ void PlaneHypothesisRefinementStrong(
	float left_min,
	float left_max,
	float right_min,
	float right_max,
	float4* plane_hypothesis,
	float* depth,
	float* cost,
	curandState* rand_state,
	const uchar* view_weights,
	const float weight_norm,
	const int2 p,
	DataPassHelper* helper

) {
	float depth_perturbation = 0.02f;
	float normal_perturbation = 0.02f;
	const Camera* cameras = helper->cameras_cuda;
	const PatchMatchParams* params = helper->params;
	float depth_min = params->depth_min;
	float depth_max = params->depth_max;

	float depth_rand = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
	float4 plane_hypothesis_rand = GenerateRandomNormal_YZL(helper, cameras[0], p, rand_state, *depth);
	float depth_perturbed = *depth;
	const float depth_min_perturbed = (1 - depth_perturbation) * depth_perturbed;
	const float depth_max_perturbed = (1 + depth_perturbation) * depth_perturbed;
	do {
		depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
	} while (depth_perturbed < depth_min && depth_perturbed > depth_max);

	//float depth_epi_neg = -1;
	//if (left_min != -1 && left_max != -1) {
	//	depth_epi_neg = curand_uniform(rand_state) * (left_max - left_min) + left_min;
	//}
	//float depth_epi_pos = -1;
	//if (right_min != -1 && right_max != -1) {
	//	depth_epi_pos = curand_uniform(rand_state) * (right_max - right_min) + right_min;
	//}
	//const int num_planes = 7;
	//float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, depth_perturbed, depth_epi_neg, depth_epi_pos};
	//float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis, *plane_hypothesis, *plane_hypothesis };

	float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, normal_perturbation * M_PI, *depth);

	float4 plane_hypothesis_perturbed_2 = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, normal_perturbation * M_PI, *depth);

	const int num_planes = 6;
	float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, *depth, depth_perturbed };
	float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, plane_hypothesis_perturbed_2 , *plane_hypothesis };

	for (int i = 0; i < num_planes; ++i) {
		float cost_vector[32] = { 2.0f };
		float4 temp_plane_hypothesis = normals[i];
		temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
		ComputeMultiViewCostVectorOld(p, temp_plane_hypothesis, cost_vector, helper);

		float temp_cost = 0.0f;
		for (int j = 0; j < params->num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				temp_cost += view_weights[j] * cost_vector[j];
			}
		}
		temp_cost /= weight_norm;

		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
		if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
			*depth = depth_before;
			*plane_hypothesis = temp_plane_hypothesis;
			*cost = temp_cost;
		}
	}
}

__device__ void ExtendedUpFarPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.x + p.y * width;
	int up_far = center - width * 5 - width * 2 * checkerboard_iter * 5;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.y > 4 + 2 * 5 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[up_far];
		costMinPoint = up_far;
#pragma omp parallel for
		for (int i = 1; i < 5; ++i) {
			if (p.y > 4 + 2 * i + 2 * 5 * checkerboard_iter) {
				int pointTemp = up_far - 2 * i * width;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else {
				break;
			}
		}
		if (costMin > costs[position[1]]) {
			costMin = costs[position[1]];
			costMinPoint = position[1];
			symbol = false;
		}
		up_far = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[up_far], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[1][i - 1];
			}
		}
	}
	else if (flag[1] == true) {
		up_far = position[1];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[1][i - 1];
		}
	}
	position[1] = up_far;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[1][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ void ExtendedDownFarPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	int down_far = center + width * 5 + width * 2 * checkerboard_iter * 5;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.y < height - 5 - 2 * 5 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[down_far];
		costMinPoint = down_far;
#pragma omp parallel for
		for (int i = 1; i < 5; ++i) {
			if (p.y < height - 5 - 2 * i - 2 * 5 * checkerboard_iter) {
				int pointTemp = down_far + 2 * i * width;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else {
				break;
			}
		}
		if (costMin > costs[position[3]]) {
			costMin = costs[position[3]];
			costMinPoint = position[3];
			symbol = false;
		}
		down_far = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[down_far], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[3][i - 1];
			}
		}
	}
	else if (flag[3] == true) {
		down_far = position[3];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[3][i - 1];
		}
	}
	position[3] = down_far;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[3][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ void ExtendedLeftFarPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.x + p.y * width;
	int left_far = center - 5 - 5 * 2 * checkerboard_iter;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.x > 4 + 2 * 5 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[left_far];
		costMinPoint = left_far;
#pragma omp parallel for
		for (int i = 1; i < 5; ++i) {
			if (p.x > 4 + 2 * i + 2 * 5 * checkerboard_iter) {
				int pointTemp = left_far - 2 * i;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else {
				break;
			}
		}
		if (costMin > costs[position[5]]) {
			costMin = costs[position[5]];
			costMinPoint = position[5];
			symbol = false;
		}
		left_far = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_far], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[5][i - 1];
			}
		}
	}
	else if (flag[5] == true) {
		left_far = position[5];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[5][i - 1];
		}
	}
	position[5] = left_far;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[5][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ void ExtendedRightFarPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	int right_far = center + 5 + 5 * 2 * checkerboard_iter;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.x < width - 5 - 2 * 5 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[right_far];
		costMinPoint = right_far;
#pragma omp parallel for
		for (int i = 1; i < 5; ++i) {
			if (p.x < width - 5 - 2 * i - 2 * 5 * checkerboard_iter) {
				int pointTemp = right_far + 2 * i;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else {
				break;
			}
		}
		if (costMin > costs[position[7]]) {
			costMin = costs[position[7]];
			costMinPoint = position[7];
			symbol = false;
		}
		right_far = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_far], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[7][i - 1];
			}
		}
	}
	else if (flag[7] == true) {
		right_far = position[7];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[7][i - 1];
		}
	}
	position[7] = right_far;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[7][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ void ExtendedLeftUpPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	int left_up = center - 5 - 6 * width - 4 * 2 * checkerboard_iter - 4 * 2 * width * checkerboard_iter;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.x > 4 + 8 * checkerboard_iter && p.y > 5 + 8 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[left_up];
		costMinPoint = left_up;
		int position_x = p.x - 5 - 8 * checkerboard_iter;
		int position_y = p.y - 6 - 8 * checkerboard_iter;
#pragma omp parallel for
		for (int i = 0; i < 7; i++) {
			if (i % 2 == 0) {
				position_x = position_x - 2;
			}
			else
			{
				position_y = position_y - 2;
			}
			if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
				int pointTemp = position_y * width + position_x;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else
			{
				break;
			}
		}
		if (costMin > costs[position[0]]) {
			costMin = costs[position[0]];
			costMinPoint = position[0];
			symbol = false;
		}
		left_up = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_up], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[0][i - 1];
			}
		}
	}
	else if (flag[0] == true) {
		left_up = position[0];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[0][i - 1];
		}
	}
	position[0] = left_up;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[0][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ void ExtendedRightUpPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	int right_up = center + 6 - 5 * width + 4 * 2 * checkerboard_iter - 4 * 2 * width * checkerboard_iter;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.x < width - 6 - 8 * checkerboard_iter && p.y > 4 + 8 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[right_up];
		costMinPoint = right_up;
		int position_x = p.x + 6 + 8 * checkerboard_iter;
		int position_y = p.y - 5 - 8 * checkerboard_iter;
#pragma omp parallel for
		for (int i = 0; i < 7; i++) {
			if (i % 2 == 0) {
				position_y = position_y - 2;
			}
			else
			{
				position_x = position_x + 2;
			}
			if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
				int pointTemp = position_y * width + position_x;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else
			{
				break;
			}
		}
		if (costMin > costs[position[2]]) {
			costMin = costs[position[2]];
			costMinPoint = position[2];
			symbol = false;
		}
		right_up = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_up], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[2][i - 1];
			}
		}
	}
	else if (flag[2] == true) {
		right_up = position[2];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[2][i - 1];
		}
	}
	position[2] = right_up;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[2][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ void ExtendedRightDownPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	int right_down = center + 5 + 6 * width + 4 * 2 * checkerboard_iter + 4 * 2 * width * checkerboard_iter;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.x < width - 5 - 8 * checkerboard_iter && p.y < height - 6 - 8 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[right_down];
		costMinPoint = right_down;
		int position_x = p.x + 5 + 8 * checkerboard_iter;
		int position_y = p.y + 6 + 8 * checkerboard_iter;
#pragma omp parallel for
		for (int i = 0; i < 7; i++) {
			if (i % 2 == 0) {
				position_x = position_x + 2;
			}
			else
			{
				position_y = position_y + 2;
			}
			if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
				int pointTemp = position_y * width + position_x;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else
			{
				break;
			}
		}
		if (costMin > costs[position[4]]) {
			costMin = costs[position[4]];
			costMinPoint = position[4];
			symbol = false;
		}
		right_down = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_down], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[4][i - 1];
			}
		}
	}
	else if (flag[4] == true) {
		right_down = position[4];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[4][i - 1];
		}
	}
	position[4] = right_down;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[4][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ void ExtendedLeftDownPropagation(DataPassHelper* helper, const Camera* cameras, float4* plane_hypotheses, float* costs, unsigned int* selected_views, const int2 p, const PatchMatchParams params, int checkerboard_iter, int* position, float cost_array[][32], bool* flag) {
	int width = cameras[0].width;
	int height = cameras[0].height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	int left_down = center - 6 + 5 * width - 4 * 2 * checkerboard_iter + 4 * 2 * width * checkerboard_iter;
	float cost_array_tmp[32];
	float costMin;
	int costMinPoint;
	if (p.x > 5 + 8 * checkerboard_iter && p.y < height - 5 - 8 * checkerboard_iter) {
		bool symbol = true;
		costMin = costs[left_down];
		costMinPoint = left_down;
		int position_x = p.x - 6 - 8 * checkerboard_iter;
		int position_y = p.y + 5 + 8 * checkerboard_iter;
#pragma omp parallel for
		for (int i = 0; i < 7; i++) {
			if (i % 2 == 0) {
				position_y = position_y + 2;
			}
			else
			{
				position_x = position_x - 2;
			}
			if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
				int pointTemp = position_y * width + position_x;
				float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
				if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
					costMin = costs[pointTemp];
					costMinPoint = pointTemp;
				}
			}
			else {
				break;
			}
		}
		if (costMin > costs[position[6]]) {
			costMin = costs[position[6]];
			costMinPoint = position[6];
			symbol = false;
		}
		left_down = costMinPoint;
		if (symbol) {
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_down], cost_array_tmp, helper);
		}
		else {
			for (int i = 1; i < params.num_images; ++i) {
				cost_array_tmp[i - 1] = cost_array[6][i - 1];
			}
		}
	}
	else if (flag[6] == true) {
		left_down = position[6];
		for (int i = 1; i < params.num_images; ++i) {
			cost_array_tmp[i - 1] = cost_array[6][i - 1];
		}
	}
	position[6] = left_down;
	for (int j = 1; j < params.num_images; j++) {
		cost_array[6][j - 1] = cost_array_tmp[j - 1];
	}
}

__device__ bool JudgeExtend(const int iter, int Extended_iter, int orientation, float(*cost_array)[32], const PatchMatchParams params, int flag) {
	if (!flag) {
		return false;
	}
	int iter_tmp = 3 - Extended_iter;
	float good_threshold = 0.8 * exp(-iter * iter * iter_tmp / 90.0);
	float bad_threshold = 1.2;
	int good_sum = 0;
	int bad_sum = 0;
	for (int i = 1; i < params.num_images; i++) {
		if (cost_array[orientation][i - 1] < good_threshold) {
			good_sum++;
		}
		if (cost_array[orientation][i - 1] > bad_threshold) {
			bad_sum++;
		}
	}
	if (good_sum >= 1 && bad_sum <= 2) {
		return false;
	}
	else {
		return true;
	}
}

__device__ void PlaneHypothesisRefinementWeak(
	float left_min,
	float left_max,
	float right_min,
	float right_max,
	float4* plane_hypothesis,
	float* depth,
	float* cost,
	curandState* rand_state,
	const uchar* view_weights,
	const float weight_norm,
	const int2 p,
	DataPassHelper* helper

) {
	float depth_perturbation = 0.02f;
	float normal_perturbation = 0.02f;
	const Camera* cameras = helper->cameras_cuda;
	const PatchMatchParams* params = helper->params;
	float depth_min = params->depth_min;
	float depth_max = params->depth_max;
	const int center = p.x + p.y * helper->width;

	if (helper->weak_info_cuda[center] == WEAK)
	{   // test the fit plane
		float4 fit_plane_hypothesis = helper->fit_plane_hypotheses_cuda[center];
		if (fit_plane_hypothesis.x == 0 && fit_plane_hypothesis.y == 0 && fit_plane_hypothesis.z == 0) {
			return;
		}

		float cost_vector[32] = { 2.0f };
		ComputeMultiViewCostVectorNew(p, fit_plane_hypothesis, cost_vector, helper);

		float temp_cost = 0.0f;
		for (int j = 0; j < params->num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				if (params->geom_consistency) {
					temp_cost += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, fit_plane_hypothesis, helper));
				}
				else {
					temp_cost += view_weights[j] * cost_vector[j];
				}
			}
		}
		temp_cost /= weight_norm;

		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], fit_plane_hypothesis, p);
		if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
			*depth = depth_before;
			*plane_hypothesis = fit_plane_hypothesis;
			*cost = temp_cost;
		}
	}
	// random refine
	{
		float depth_rand = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
		float4 plane_hypothesis_rand = GenerateRandomNormal_YZL(helper, cameras[0], p, rand_state, *depth);
		float depth_perturbed = *depth;
		const float depth_min_perturbed = (1 - depth_perturbation) * depth_perturbed;
		const float depth_max_perturbed = (1 + depth_perturbation) * depth_perturbed;
		do {
			depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
		} while (depth_perturbed < depth_min && depth_perturbed > depth_max);

		//float depth_epi_neg = -1;
		//if (left_min != -1 && left_max != -1) {
		//	depth_epi_neg = curand_uniform(rand_state) * (left_max - left_min) + left_min;
		//}
		//float depth_epi_pos = -1;
		//if (right_min != -1 && right_max != -1) {
		//	depth_epi_pos = curand_uniform(rand_state) * (right_max - right_min) + right_min;
		//}
		//const int num_planes = 7;
		//float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, depth_perturbed, depth_epi_neg, depth_epi_pos};
		//float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis, *plane_hypothesis, *plane_hypothesis };

		float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, normal_perturbation * M_PI, *depth);

		float4 plane_hypothesis_perturbed_2 = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, normal_perturbation * M_PI, *depth);

		const int num_planes = 6;
		float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, *depth, depth_perturbed };
		float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, plane_hypothesis_perturbed_2 , *plane_hypothesis };

		for (int i = 0; i < num_planes; ++i) {
			float cost_vector[32] = { 2.0f };
			float4 temp_plane_hypothesis = normals[i];
			temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
			ComputeMultiViewCostVectorNew(p, temp_plane_hypothesis, cost_vector, helper);

			float temp_cost = 0.0f;
			for (int j = 0; j < params->num_images - 1; ++j) {
				if (view_weights[j] > 0) {
					if (params->geom_consistency) {
						temp_cost += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, temp_plane_hypothesis, helper));
					}
					else {
						temp_cost += view_weights[j] * cost_vector[j];
					}
				}
			}
			temp_cost /= weight_norm;

			float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
			if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
				*depth = depth_before;
				*plane_hypothesis = temp_plane_hypothesis;
				*cost = temp_cost;
			}
		}
	}
}

__device__ void CheckerboardPropagationStrong(
	const int2 p,
	const int iter,
	DataPassHelper* helper
) {
	const int width = helper->width;
	const int height = helper->height;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	curandState* rand_states = helper->rand_states_cuda;
	unsigned int* selected_views = helper->selected_views_cuda;
	PatchMatchParams* params = helper->params;
	const Camera* cameras = helper->cameras_cuda;
	int num_images = params->num_images;

	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;

	// Adaptive Checkerboard Sampling
	float cost_array[8][32] = { 2.0f };
	bool flag[8] = { false };
	int num_valid_pixels = 0;

	int positions[8];

	if (helper->params->use_edge) {
		const int dir[EDGE_NEIGH_NUM][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1} };
		const unsigned offset = center * EDGE_NEIGH_NUM;
		const short2* edge_neigh = &(helper->edge_neigh_cuda[offset]);
		const uchar* edge = helper->edge_cuda;
		const float max_edge_dist = MAX(height, width) / 30.0f;
		const int min_step_len = 2;

		{
			for (int dir_index = 0; dir_index < EDGE_NEIGH_NUM; ++dir_index) {
				const int dx = dir[dir_index][0];
				const int dy = dir[dir_index][1];
				const int sx = 5 * dx;
				const int sy = 5 * dy;

				short2 edge_pt = edge_neigh[dir_index];
				float dist = std::sqrt(std::pow(edge_pt.x - p.x, 2) + std::pow(edge_pt.y - p.y, 2));
				if (dir_index >= 4) dist /= std::sqrt(2.0);
				if (edge[center]) {
					dist = 11 * min_step_len;
				}
				else if (!edge_pt.x == -1 || edge_pt.y == -1 || dist >= max_edge_dist) {
					dist = max_edge_dist;
					if (dir_index >= 4) dist /= std::sqrt(2.0);
				}
				int step_num = MIN(MAX(11, (int)(1.0f * dist / min_step_len)), 22);
				int step_len = MAX((int)(1.0f * dist / step_num), min_step_len);
				if (dir_index < 4 && step_len % 2 == 1) step_len -= 1;

				int2 min_cost_pt;
				float min_cost = FLT_MAX;
				for (int step = 0; step < step_num; ++step) {
					int fx = 0, fy = 0;
					if (dir_index > 4) {
						if (dir_index % 2) fx = dx; else fy = dy;
					}
					const int2 temp_pt = make_int2(p.x + sx + step * step_len * dx + fx, p.y + sy + step * step_len * dy + fy);
					if (!(temp_pt.x >= 0 && temp_pt.y >= 0 && temp_pt.x < width && temp_pt.y < height)) continue;
					const int pointTempCenter = temp_pt.x + temp_pt.y * width;
					if (min_cost > costs[pointTempCenter]) {
						min_cost_pt = temp_pt;
						min_cost = costs[pointTempCenter];
					}
				}
				if (min_cost < FLT_MAX) {
					flag[dir_index] = true;
					positions[dir_index] = min_cost_pt.x + min_cost_pt.y * width;
					ComputeMultiViewCostVectorOld(p, plane_hypotheses[positions[dir_index]], cost_array[dir_index], helper);
				}
			}
		}
		// �Ա�Ե���ػ�Ǳ�Ե���ؾ���һ��11�����ķǾֲ��������Ǳ�Ե���ػ������һ�α�Ե����Ӧ�Ĳ���
		if (!edge[center]) {
			const float good_threshold = 0.8f * expf((iter) * (iter) / (-90.0f));
			const float bad_threshold = 1.2f;
			for (int dir_index = 0; dir_index < EDGE_NEIGH_NUM; ++dir_index) {
				const int dx = dir[dir_index][0];
				const int dy = dir[dir_index][1];
				const int sx = 5 * dx;
				const int sy = 5 * dy;

				bool hasResBefore = flag[dir_index];

				float temp_cost_array[32] = { 2.0f };
				int temp_position;

				int2 min_cost_pt;
				float min_cost = FLT_MAX;
				for (int step = 0; step < 11; ++step) {
					int fx = 0, fy = 0;
					if (dir_index > 4) {
						if (dir_index % 2) fx = dx; else fy = dy;
					}
					const int2 temp_pt = make_int2(p.x + sx + step * min_step_len * dx + fx, p.y + sy + step * min_step_len * dy + fy);
					if (!(temp_pt.x >= 0 && temp_pt.y >= 0 && temp_pt.x < width && temp_pt.y < height)) continue;
					const int pointTempCenter = temp_pt.x + temp_pt.y * width;
					if (min_cost > costs[pointTempCenter]) {
						min_cost_pt = temp_pt;
						min_cost = costs[pointTempCenter];
					}
				}
				if (min_cost < FLT_MAX) {
					flag[dir_index] = true;
					temp_position = min_cost_pt.x + min_cost_pt.y * width;
					ComputeMultiViewCostVectorOld(p, plane_hypotheses[temp_position], temp_cost_array, helper);

					int good_num[2] = { 0, 0 }, bad_num[2] = { 0, 0 };
					for (int i = 0; i < 2; i++) {
						for (int j = 0; j < num_images - 1; j++) {
							float val = (i == 0 ? cost_array[dir_index][j] : temp_cost_array[j]);
							if (val < good_threshold) good_num[i]++;
							if (val > bad_threshold) bad_num[i]++;
						}
					}
					if (!hasResBefore || good_num[1] > good_num[0] || (good_num[1] == good_num[0] && bad_num[1] < bad_num[0])) {
						positions[dir_index] = temp_position;
						for (int j = 0; j < num_images - 1; j++) {
							cost_array[dir_index][j] = temp_cost_array[j];
						}
					}
				}
			}
		}
	}
	else {
		float costMin;
		int costMinPoint;

		int left_up = center - 5 - 6 * width;
		int right_up = center + 6 - 5 * width;
		int right_down = center + 5 + 6 * width;
		int left_down = center - 6 + 5 * width;

		int left_far = center - 5;
		int right_far = center + 5;
		int up_far = center - width * 5;
		int down_far = center + width * 5;
		// 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
		// up_far
		if (p.y > 4) {
			flag[1] = true;
			num_valid_pixels++;
			costMin = costs[up_far];
			costMinPoint = up_far;
#pragma omp parallel for
			for (int i = 1; i < 5; ++i) {
				if (p.y > 4 + 2 * i) {
					int pointTemp = up_far - 2 * i * width;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else {
					break;
				}
			}
			up_far = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[up_far], cost_array[1], helper);
		}

		// dwon_far
		if (p.y < height - 5) {
			flag[3] = true;
			num_valid_pixels++;
			costMin = costs[down_far];
			costMinPoint = down_far;
#pragma omp parallel for
			for (int i = 1; i < 5; ++i) {
				if (p.y < height - 5 - 2 * i) {
					int pointTemp = down_far + 2 * i * width;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else {
					break;
				}
			}
			down_far = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[down_far], cost_array[3], helper);
		}

		// left_far
		if (p.x > 4) {
			flag[5] = true;
			num_valid_pixels++;
			costMin = costs[left_far];
			costMinPoint = left_far;
#pragma omp parallel for
			for (int i = 1; i < 5; ++i) {
				if (p.x > 4 + 2 * i) {
					int pointTemp = left_far - 2 * i;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else {
					break;
				}
			}
			left_far = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_far], cost_array[5], helper);
		}

		// right_far
		if (p.x < width - 5) {
			flag[7] = true;
			num_valid_pixels++;
			costMin = costs[right_far];
			costMinPoint = right_far;
#pragma omp parallel for
			for (int i = 1; i < 5; ++i) {
				if (p.x < width - 5 - 2 * i) {
					int pointTemp = right_far + 2 * i;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else {
					break;
				}
			}
			right_far = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_far], cost_array[7], helper);
		}

		//checkerboard_version_1
		//left_up
		if (p.x > 4 && p.y > 5) {
			flag[0] = true;
			num_valid_pixels++;
			costMin = costs[left_up];
			costMinPoint = left_up;
			int position_x = p.x - 5;
			int position_y = p.y - 6;
#pragma omp parallel for
			for (int i = 0; i < 7; i++) {
				if (i % 2 == 0) {
					position_x = position_x - 2;
				}
				else
				{
					position_y = position_y - 2;
				}
				if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
					int pointTemp = position_y * width + position_x;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else
				{
					break;
				}
			}
			left_up = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_up], cost_array[0], helper);
		}

		//right_up
		if (p.x < width - 6 && p.y>4) {
			flag[2] = true;
			num_valid_pixels++;
			costMin = costs[right_up];
			costMinPoint = right_up;
			int position_x = p.x + 6;
			int position_y = p.y - 5;
#pragma omp parallel for
			for (int i = 0; i < 7; i++) {
				if (i % 2 == 0) {
					position_y = position_y - 2;
				}
				else
				{
					position_x = position_x + 2;
				}
				if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
					int pointTemp = position_y * width + position_x;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else
				{
					break;
				}
			}
			right_up = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_up], cost_array[2], helper);
		}

		//right_down
		if (p.x < width - 5 && p.y < height - 6) {
			flag[4] = true;
			num_valid_pixels++;
			costMin = costs[right_down];
			costMinPoint = right_down;
			int position_x = p.x + 5;
			int position_y = p.y + 6;
#pragma omp parallel for
			for (int i = 0; i < 7; i++) {
				if (i % 2 == 0) {
					position_x = position_x + 2;
				}
				else
				{
					position_y = position_y + 2;
				}
				if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
					int pointTemp = position_y * width + position_x;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else
				{
					break;
				}
			}
			right_down = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_down], cost_array[4], helper);
		}

		//left_down
		if (p.x > 5 && p.y < height - 5) {
			flag[6] = true;
			num_valid_pixels++;
			costMin = costs[left_down];
			costMinPoint = left_down;
			int position_x = p.x - 6;
			int position_y = p.y + 5;
#pragma omp parallel for
			for (int i = 0; i < 7; i++) {
				if (i % 2 == 0) {
					position_y = position_y + 2;
				}
				else
				{
					position_x = position_x - 2;
				}
				if (position_x >= 0 && position_x < width && position_y < height && position_y >= 0) {
					int pointTemp = position_y * width + position_x;
					float4 view = GetViewDirection(cameras[0], p, plane_hypotheses[pointTemp].w);
					if (costs[pointTemp] < costMin && plane_hypotheses[pointTemp].x * view.x + plane_hypotheses[pointTemp].y * view.y + plane_hypotheses[pointTemp].z * view.z <= 0.0f) {
						costMin = costs[pointTemp];
						costMinPoint = pointTemp;
					}
				}
				else {
					break;
				}
			}
			left_down = costMinPoint;
			ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_down], cost_array[6], helper);
		}

		int positions_tmp[8] = { left_up, up_far, right_up, down_far, right_down, left_far, left_down, right_far };
		//Extended eight orientations propagation
		int itertimes = 0;
		bool symbol_eight_orientations[10] = { true };
		while (itertimes < 3)
		{
			if (symbol_eight_orientations[0] == true) {
				if (JudgeExtend(iter, itertimes, 0, cost_array, *params, flag[0])) {
					ExtendedLeftUpPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[0] = false;
				}
			}
			if (symbol_eight_orientations[1] == true) {
				if (JudgeExtend(iter, itertimes, 1, cost_array, *params, flag[1])) {
					ExtendedUpFarPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[1] = false;
				}
			}
			if (symbol_eight_orientations[2] == true) {
				if (JudgeExtend(iter, itertimes, 2, cost_array, *params, flag[2])) {
					ExtendedRightUpPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[2] = false;
				}
			}
			if (symbol_eight_orientations[3] == true) {
				if (JudgeExtend(iter, itertimes, 3, cost_array, *params, flag[3])) {
					ExtendedDownFarPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[3] = false;
				}
			}
			if (symbol_eight_orientations[4] == true) {
				if (JudgeExtend(iter, itertimes, 4, cost_array, *params, flag[4])) {
					ExtendedRightDownPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[4] = false;
				}
			}
			if (symbol_eight_orientations[5] == true) {
				if (JudgeExtend(iter, itertimes, 5, cost_array, *params, flag[5])) {
					ExtendedLeftFarPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[5] = false;
				}
			}
			if (symbol_eight_orientations[6] == true) {
				if (JudgeExtend(iter, itertimes, 6, cost_array, *params, flag[6])) {
					ExtendedLeftDownPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[6] = false;
				}
			}
			if (symbol_eight_orientations[7] == true) {
				if (JudgeExtend(iter, itertimes, 7, cost_array, *params, flag[7])) {
					ExtendedRightFarPropagation(helper, cameras, plane_hypotheses, costs, selected_views, p, *params, itertimes, positions_tmp, cost_array, flag);
				}
				else {
					symbol_eight_orientations[7] = false;
				}
			}
			itertimes++;
		}
	}

	// Multi-hypothesis Joint View Selection
	uchar* view_weights = &(helper->view_weight_cuda[center * MAX_IMAGES]);
	for (int i = 0; i < MAX_IMAGES; ++i) {
		view_weights[i] = 0;
	}
	float view_selection_priors[32] = { 0.0f };

	int neighbor_positions[4] = { center - width, center + width, center - 1, center + 1 };
	for (int i = 0; i < 4; ++i) {
		if (flag[2 * i]) {
			for (int j = 0; j < num_images - 1; ++j) {
				if (isSet(selected_views[neighbor_positions[i]], j) == 1) {
					view_selection_priors[j] += 0.9f;
				}
				else {
					view_selection_priors[j] += 0.1f;
				}
			}
		}
	}

	float sampling_probs[32] = { 0.0f };
	float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));
	for (int i = 0; i < num_images - 1; i++) {
		float count = 0;
		int count_false = 0;
		float tmpw = 0;
		for (int j = 0; j < 8; j++) {
			if (cost_array[j][i] < cost_threshold) {
				tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
				count++;
			}
			if (cost_array[j][i] > 1.2f) {
				count_false++;
			}
		}
		if (count > 2 && count_false < 3) {
			sampling_probs[i] = tmpw / count;
		}
		else if (count_false < 3) {
			sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
		}
		sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
	}

	TransformPDFToCDF(sampling_probs, num_images - 1);
	for (int sample = 0; sample < 15; ++sample) {
		const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

		for (int image_id = 0; image_id < num_images - 1; ++image_id) {
			const float prob = sampling_probs[image_id];
			if (prob > rand_prob) {
				view_weights[image_id] += 1;
				break;
			}
		}
	}

	unsigned int temp_selected_views = 0;
	int num_selected_view = 0;
	float weight_norm = 0;

	for (int i = 0; i < num_images - 1; ++i) {
		if (view_weights[i] > 0) {
			setBit(&temp_selected_views, i);
			weight_norm += view_weights[i];
			num_selected_view++;
		}
	}

	float final_costs[8] = { 0.0f };

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				final_costs[i] += view_weights[j] * cost_array[i][j];
			}
		}

		final_costs[i] /= weight_norm;
	}

	const int min_cost_idx = FindMinCostIndex(final_costs, 8);

	float cost_vector_now[32] = { 2.0f };
	ComputeMultiViewCostVectorOld(p, plane_hypotheses[center], cost_vector_now, helper);
	float cost_now = 0.0f;

	for (int i = 0; i < num_images - 1; ++i) {
		cost_now += view_weights[i] * cost_vector_now[i];
	}
	cost_now /= weight_norm;
	costs[center] = cost_now;
	float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	float4 plane_hypotheses_now = plane_hypotheses[center];

	if (flag[min_cost_idx]) {
		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);

		if (depth_before >= params->depth_min && depth_before <= params->depth_max && final_costs[min_cost_idx] < cost_now) {
			depth_now = depth_before;
			plane_hypotheses_now = plane_hypotheses[positions[min_cost_idx]];
			cost_now = final_costs[min_cost_idx];
			selected_views[center] = temp_selected_views;
		}
	}

	float left_min = -1;
	float left_max = -1;
	float right_min = -1;
	float right_max = -1;

	//�㼶�� + ���ߴ���ѡ��С + �����㷨��
	if (false) {
		// params->num_images
		float all_left_min[25];
		float all_left_max[25];
		float all_right_min[25];
		float all_right_max[25];
		int lmin = 0; int lmax = 0;
		int rmin = 0; int rmax = 0;

		for (int src_idx = 1; src_idx < params->num_images; ++src_idx) {
			if (isSet(selected_views[center], src_idx - 1)) {
				// ���㼫��
				float R_At[9], R[9], R_[3];
				matTranspose3x3(cameras[0].R, R_At);
				matMul3x3(cameras[src_idx].R, R_At, R);
				matMul3x1(R, cameras[0].t, R_);
				float t[3];
				matSub3x1(t, cameras[src_idx].t, R_);
				float E[9];
				computeEssentialMatrix(R, t, E);
				float F[9];
				computeFundamentalMatrix(cameras[0].K, E, F);
				float x[3] = { p.x, p.y, 1.0f };
				float l[3];
				computeEpipolarLine(F, x, l);
				//�õ�ȫ���ĵ����ڻ����С����ͶӰ��
				float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, plane_hypotheses[center].w, cameras[0]);
				float2 src_pt;
				float src_d;
				ProjectonCamera_cu(forward_point, cameras[src_idx], src_pt, src_d);
				int2 src_pt_int = make_int2((int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);
				float dx = l[1];
				float dy = -l[0];
				float length = sqrt(dx * dx + dy * dy);
				dx /= length;
				dy /= length;
				const int Len = 160;
				int cnt = 0;
				int2 last_pt = make_int2(0, 0);
				float cost_min = 1000;
				int2 min_pt = make_int2(0, 0);
				const Camera ref_camera = helper->cameras_cuda[0];
				const Camera src_camera = helper->cameras_cuda[src_idx];
				const cudaTextureObject_t depth_image = helper->texture_depths_cuda[0].images[src_idx];

				int2 positive[Len];
				int2 negative[Len];
				int pos = 0; int neg = 0;

				for (int i = -Len; i <= Len; ++i) {
					int2 each_pt = make_int2(src_pt_int.x + static_cast<int>(dx * i), src_pt_int.y + static_cast<int>(dy * i));
					if (last_pt.x != each_pt.x || last_pt.y != each_pt.y) {
						last_pt.x = each_pt.x;
						last_pt.y = each_pt.y;
						const float src_depth = tex2D<float>(depth_image, last_pt.x, last_pt.y);
						float3 src_3D_pt = Get3DPointonWorld_cu(last_pt.x, last_pt.y, src_depth, src_camera);
						float2 backward_point;
						float ref_d;
						ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);
						const float diff_col = p.x - backward_point.x;
						const float diff_row = p.y - backward_point.y;
						float cost_this = sqrt(diff_col * diff_col + diff_row * diff_row);
						if (cost_this < cost_min) {
							min_pt.x = each_pt.x;
							min_pt.y = each_pt.y;
							cost_min = cost_this;
						}
						// ��¼���еĵ�
						if (i < 0) {
							negative[neg++] = make_int2(last_pt.x, last_pt.y);
						}
						if (i > 0) {
							positive[pos++] = make_int2(last_pt.x, last_pt.y);
						}
					}
				}

				//��С2�����10
				int min_val = 2;
				int max_val = 8;

				{
					int bound = MIN(min_val, neg - 1);
					const float src_depth = tex2D<float>(depth_image, negative[bound].x, negative[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(negative[bound].x, negative[bound].y, src_depth, src_camera);
					all_left_min[lmin++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}
				{
					int bound = MIN(min_val, pos - 1);
					const float src_depth = tex2D<float>(depth_image, positive[bound].x, positive[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(positive[bound].x, positive[bound].y, src_depth, src_camera);
					all_right_min[rmin++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}
				{
					int bound = MIN(max_val, neg - 1);
					const float src_depth = tex2D<float>(depth_image, negative[bound].x, negative[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(negative[bound].x, negative[bound].y, src_depth, src_camera);
					all_left_max[lmax++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}
				{
					int bound = MIN(max_val, pos - 1);
					const float src_depth = tex2D<float>(depth_image, positive[bound].x, positive[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(positive[bound].x, positive[bound].y, src_depth, src_camera);
					all_right_max[rmax++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}



				//����С�ĵ����ȴ���
				const float src_depth = tex2D<float>(depth_image, min_pt.x, min_pt.y);
				float3 pt_3D = Get3DPointonWorld_cu(min_pt.x, min_pt.y, src_depth, src_camera);
				float4 plane_hypotheses_epi = plane_hypotheses[center];
				plane_hypotheses_epi.w = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				float cost_vector_epi[32] = { 2.0f };
				ComputeMultiViewCostVectorOld(p, plane_hypotheses_epi, cost_vector_epi, helper);
				float cost_epi = 0.0f;
				for (int i = 0; i < num_images - 1; ++i) {
					cost_epi += view_weights[i] * cost_vector_epi[i];
				}
				if (flag[min_cost_idx]) {
					float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses_epi, p);
					if (depth_before >= params->depth_min && depth_before <= params->depth_max && cost_epi < cost_now) {
						depth_now = depth_before;
						plane_hypotheses_now = plane_hypotheses_epi;
						cost_now = cost_epi;
						selected_views[center] = temp_selected_views;
					}
				}
			}
		}

		int e = 3;
		int f = 2;
		//����������Ȼ��ѡ��eС��
		sort_small_k(all_left_min, lmin, e);
		sort_small_k(all_right_min, rmin, e);
		//����������Ȼ��ѡ��f���
		sort_large_k(all_left_max, lmax, f);
		sort_large_k(all_right_max, rmax, f);

		if (lmin >= e && lmax >= e) {
			left_min = all_left_min[e - 1];
			right_min = all_right_min[e - 1];
		}
		if (rmin >= f && rmax >= f) {
			left_max = all_left_max[f - 1];
			right_max = all_right_max[f - 1];
		}
	}

	PlaneHypothesisRefinementStrong(left_min, left_max, right_min, right_max, &plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], view_weights, weight_norm, p, helper);

	if (params->state == REFINE_INIT) {
		if (cost_now < costs[center] - 0.1) {
			costs[center] = cost_now;
			plane_hypotheses[center] = plane_hypotheses_now;
		}
	}
	else {
		costs[center] = cost_now;
		plane_hypotheses[center] = plane_hypotheses_now;
	}
}

__device__ void CheckerboardPropagationWeak(
	const int2 p,
	const int iter,
	DataPassHelper* helper
) {
	const int width = helper->width;
	const int height = helper->height;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	curandState* rand_states = helper->rand_states_cuda;
	unsigned int* selected_views = helper->selected_views_cuda;
	PatchMatchParams* params = helper->params;
	const Camera* cameras = helper->cameras_cuda;
	int num_images = params->num_images;

	if (p.x >= width || p.y >= height) {
		return;
	}

	const int center = p.y * width + p.x;

	// Adaptive Checkerboard Sampling
	float cost_array[8][32] = { 2.0f };
	bool flag[8] = { false };
	int num_valid_pixels = 0;

	int positions[8] = { 0 };
	float4 new_plane_hypothesis[8];

	for (int i = 0; i < 8; ++i) {
		const auto neighbour_pt = GetNeighbourPoint(p, i + 1, helper);
		if (neighbour_pt.x == -1 || neighbour_pt.y == -1 || helper->weak_info_cuda[neighbour_pt.x + neighbour_pt.y * width] != STRONG) {
			flag[i] = false;
			continue;
		}
		positions[i] = neighbour_pt.x + neighbour_pt.y * width;
		flag[i] = true;
		num_valid_pixels++;
		ComputeMultiViewCostVectorNew(p, plane_hypotheses[neighbour_pt.x + neighbour_pt.y * width], cost_array[i], helper);
		new_plane_hypothesis[i] = plane_hypotheses[neighbour_pt.x + neighbour_pt.y * width];
	}

	// Multi-hypothesis Joint View Selection
	uchar* view_weights = &(helper->view_weight_cuda[center * MAX_IMAGES]);
	for (int i = 0; i < MAX_IMAGES; ++i) {
		view_weights[i] = 0;
	}
	float view_selection_priors[32] = { 0.0f };

	for (int i = 0; i < 8; ++i) {
		const auto neighbour_pt = GetNeighbourPoint(p, i + 1, helper);
		if (neighbour_pt.x == -1 || neighbour_pt.y == -1) {
			continue;
		}
		for (int j = 0; j < num_images - 1; ++j) {
			if (isSet(selected_views[neighbour_pt.x + neighbour_pt.y * width], j) == 1) {
				view_selection_priors[j] += 0.9f;
			}
			else {
				view_selection_priors[j] += 0.1f;
			}
		}
	}

	float sampling_probs[32] = { 0.0f };
	float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));
	for (int i = 0; i < num_images - 1; i++) {
		float count = 0;
		int count_false = 0;
		float tmpw = 0;
		for (int j = 0; j < 8; j++) {
			if (cost_array[j][i] < cost_threshold) {
				tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
				count++;
			}
			if (cost_array[j][i] > 1.2f) {
				count_false++;
			}
		}
		if (count > 2 && count_false < 3) {
			sampling_probs[i] = tmpw / count;
		}
		else if (count_false < 3) {
			sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
		}
		sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
	}

	TransformPDFToCDF(sampling_probs, num_images - 1);
	for (int sample = 0; sample < 15; ++sample) {
		const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

		for (int image_id = 0; image_id < num_images - 1; ++image_id) {
			const float prob = sampling_probs[image_id];
			if (prob > rand_prob) {
				view_weights[image_id] += 1;
				break;
			}
		}
	}

	unsigned int temp_selected_views = 0;
	int num_selected_view = 0;
	float weight_norm = 0;

	for (int i = 0; i < num_images - 1; ++i) {
		if (view_weights[i] > 0) {
			setBit(&temp_selected_views, i);
			weight_norm += view_weights[i];
			num_selected_view++;
		}
	}

	float final_costs[8] = { 0.0f };

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < num_images - 1; ++j) {
			if (view_weights[j] > 0) {
				if (params->geom_consistency) {
					if (flag[i]) {
						final_costs[i] += view_weights[j] * (cost_array[i][j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, plane_hypotheses[positions[i]], helper));
					}
					else {
						final_costs[i] += view_weights[j] * (cost_array[i][j] + params->geom_factor * 3.0f);
					}
				}
				else {
					final_costs[i] += view_weights[j] * cost_array[i][j];
				}
			}
		}

		final_costs[i] /= weight_norm;
	}

	const int min_cost_idx = FindMinCostIndex(final_costs, 8);

	float cost_vector_now[32] = { 2.0f };
	ComputeMultiViewCostVectorNew(p, plane_hypotheses[center], cost_vector_now, helper);
	float cost_now = 0.0f;

	for (int i = 0; i < num_images - 1; ++i) {
		if (params->geom_consistency) {
			cost_now += view_weights[i] * (cost_vector_now[i] + params->geom_factor * ComputeGeomConsistencyCost(p, i + 1, plane_hypotheses[center], helper));
		}
		else {
			cost_now += view_weights[i] * cost_vector_now[i];
		}
	}
	cost_now /= weight_norm;
	costs[center] = cost_now;
	float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	float4 plane_hypotheses_now = plane_hypotheses[center];

	if (flag[min_cost_idx]) {
		float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], new_plane_hypothesis[min_cost_idx], p);
		if (depth_before >= params->depth_min && depth_before <= params->depth_max && final_costs[min_cost_idx] < cost_now) {
			depth_now = depth_before;
			plane_hypotheses_now = new_plane_hypothesis[min_cost_idx];
			cost_now = final_costs[min_cost_idx];
			selected_views[center] = temp_selected_views;
		}
	}

	float left_min = -1;
	float left_max = -1;
	float right_min = -1;
	float right_max = -1;

	//�㼶�� + ���ߴ���ѡ��С + �����㷨��
	if (false) {
		// params->num_images
		float all_left_min[25];
		float all_left_max[25];
		float all_right_min[25];
		float all_right_max[25];
		int lmin = 0; int lmax = 0;
		int rmin = 0; int rmax = 0;

		for (int src_idx = 1; src_idx < params->num_images; ++src_idx) {
			if (isSet(selected_views[center], src_idx - 1)) {
				// ���㼫��
				float R_At[9], R[9], R_[3];
				matTranspose3x3(cameras[0].R, R_At);
				matMul3x3(cameras[src_idx].R, R_At, R);
				matMul3x1(R, cameras[0].t, R_);
				float t[3];
				matSub3x1(t, cameras[src_idx].t, R_);
				float E[9];
				computeEssentialMatrix(R, t, E);
				float F[9];
				computeFundamentalMatrix(cameras[0].K, E, F);
				float x[3] = { p.x, p.y, 1.0f };
				float l[3];
				computeEpipolarLine(F, x, l);
				//�õ�ȫ���ĵ����ڻ����С����ͶӰ��
				float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, plane_hypotheses[center].w, cameras[0]);
				float2 src_pt;
				float src_d;
				ProjectonCamera_cu(forward_point, cameras[src_idx], src_pt, src_d);
				int2 src_pt_int = make_int2((int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);
				float dx = l[1];
				float dy = -l[0];
				float length = sqrt(dx * dx + dy * dy);
				dx /= length;
				dy /= length;
				const int Len = 160;
				int cnt = 0;
				int2 last_pt = make_int2(0, 0);
				float cost_min = 1000;
				int2 min_pt = make_int2(0, 0);
				const Camera ref_camera = helper->cameras_cuda[0];
				const Camera src_camera = helper->cameras_cuda[src_idx];
				const cudaTextureObject_t depth_image = helper->texture_depths_cuda[0].images[src_idx];

				int2 positive[Len];
				int2 negative[Len];
				int pos = 0; int neg = 0;

				for (int i = -Len; i <= Len; ++i) {
					int2 each_pt = make_int2(src_pt_int.x + static_cast<int>(dx * i), src_pt_int.y + static_cast<int>(dy * i));
					if (last_pt.x != each_pt.x || last_pt.y != each_pt.y) {
						last_pt.x = each_pt.x;
						last_pt.y = each_pt.y;
						const float src_depth = tex2D<float>(depth_image, last_pt.x, last_pt.y);
						float3 src_3D_pt = Get3DPointonWorld_cu(last_pt.x, last_pt.y, src_depth, src_camera);
						float2 backward_point;
						float ref_d;
						ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);
						const float diff_col = p.x - backward_point.x;
						const float diff_row = p.y - backward_point.y;
						float cost_this = sqrt(diff_col * diff_col + diff_row * diff_row);
						if (cost_this < cost_min) {
							min_pt.x = each_pt.x;
							min_pt.y = each_pt.y;
							cost_min = cost_this;
						}
						// ��¼���еĵ�
						if (i < 0) {
							negative[neg++] = make_int2(last_pt.x, last_pt.y);
						}
						if (i > 0) {
							positive[pos++] = make_int2(last_pt.x, last_pt.y);
						}
					}
				}

				//��С2�����10
				int min_val = 2;
				int max_val = 8;

				{
					int bound = MIN(min_val, neg - 1);
					const float src_depth = tex2D<float>(depth_image, negative[bound].x, negative[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(negative[bound].x, negative[bound].y, src_depth, src_camera);
					all_left_min[lmin++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}
				{
					int bound = MIN(min_val, pos - 1);
					const float src_depth = tex2D<float>(depth_image, positive[bound].x, positive[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(positive[bound].x, positive[bound].y, src_depth, src_camera);
					all_right_min[rmin++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}
				{
					int bound = MIN(max_val, neg - 1);
					const float src_depth = tex2D<float>(depth_image, negative[bound].x, negative[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(negative[bound].x, negative[bound].y, src_depth, src_camera);
					all_left_max[lmax++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}
				{
					int bound = MIN(max_val, pos - 1);
					const float src_depth = tex2D<float>(depth_image, positive[bound].x, positive[bound].y);
					float3 pt_3D = Get3DPointonWorld_cu(positive[bound].x, positive[bound].y, src_depth, src_camera);
					all_right_max[rmax++] = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				}



				//����С�ĵ����ȴ���
				const float src_depth = tex2D<float>(depth_image, min_pt.x, min_pt.y);
				float3 pt_3D = Get3DPointonWorld_cu(min_pt.x, min_pt.y, src_depth, src_camera);
				float4 plane_hypotheses_epi = plane_hypotheses[center];
				plane_hypotheses_epi.w = GetDepthFrom3DPoint(p.x, p.y, pt_3D, ref_camera);
				float cost_vector_epi[32] = { 2.0f };
				ComputeMultiViewCostVectorOld(p, plane_hypotheses_epi, cost_vector_epi, helper);
				float cost_epi = 0.0f;
				for (int i = 0; i < num_images - 1; ++i) {
					cost_epi += view_weights[i] * cost_vector_epi[i];
				}
				if (flag[min_cost_idx]) {
					float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses_epi, p);
					if (depth_before >= params->depth_min && depth_before <= params->depth_max && cost_epi < cost_now) {
						depth_now = depth_before;
						plane_hypotheses_now = plane_hypotheses_epi;
						cost_now = cost_epi;
						selected_views[center] = temp_selected_views;
					}
				}
			}
		}

		int e = 3;
		int f = 2;
		//����������Ȼ��ѡ��eС��
		sort_small_k(all_left_min, lmin, e);
		sort_small_k(all_right_min, rmin, e);
		//����������Ȼ��ѡ��f���
		sort_large_k(all_left_max, lmax, f);
		sort_large_k(all_right_max, rmax, f);

		if (lmin >= e && lmax >= e) {
			left_min = all_left_min[e - 1];
			right_min = all_right_min[e - 1];
		}
		if (rmin >= f && rmax >= f) {
			left_max = all_left_max[f - 1];
			right_max = all_right_max[f - 1];
		}
	}

	PlaneHypothesisRefinementWeak(left_min, left_max, right_min, right_max, &plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], view_weights, weight_norm, p, helper);

	if (params->state == REFINE_INIT) {
		if (cost_now < costs[center] - 0.1) {
			costs[center] = cost_now;
			plane_hypotheses[center] = plane_hypotheses_now;
		}
	}
	else {
		costs[center] = cost_now;
		plane_hypotheses[center] = plane_hypotheses_now;
	}

	{// update cost with old method
		float temp_radius = helper->params->strong_radius;
		if (helper->params->use_radius) {
			temp_radius = helper->radius_cuda[center];
			helper->radius_cuda[center] = helper->params->strong_radius;
		}
		cost_now = 0.0f;
		ComputeMultiViewCostVectorOld(p, plane_hypotheses[center], cost_vector_now, helper);
		for (int i = 0; i < num_images - 1; ++i) {
			cost_now += view_weights[i] * cost_vector_now[i];
		}
		cost_now /= weight_norm;
		costs[center] = cost_now;
		if (helper->params->use_radius) {
			helper->radius_cuda[center] = temp_radius;
		}
	}
}

__global__ void BlackPixelUpdateWeak(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		CheckerboardPropagationWeak(p, iter, helper);
	}
}

__global__ void RedPixelUpdateWeak(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		CheckerboardPropagationWeak(p, iter, helper);
	}
}

__global__ void BlackPixelUpdateStrong(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		return;
	}

	CheckerboardPropagationStrong(p, iter, helper);
}

__global__ void RedPixelUpdateStrong(const int iter, DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] == WEAK) {
		return;
	}

	CheckerboardPropagationStrong(p, iter, helper);
}

__global__ void GetDepthandNormal(
	DataPassHelper* helper
) {
	const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	Camera* cameras = helper->cameras_cuda;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	const int width = helper->width;;
	const int height = helper->height;

	if (p.x >= width || p.y >= height) {
		return;
	}
	const int center = p.y * width + p.x;
	plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
	plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__device__ void CheckerboardFilterStrong(
	const int2 p,
	DataPassHelper* helper
) {
	int width = helper->width;
	int height = helper->height;
	if (p.x >= width || p.y >= height) {
		return;
	}
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float* costs = helper->costs_cuda;
	const int center = p.y * width + p.x;

	float filter[21];
	int index = 0;

	filter[index++] = plane_hypotheses[center].w;

	// Left
	const int left = center - 1;
	const int leftleft = center - 3;

	// Up
	const int up = center - width;
	const int upup = center - 3 * width;

	// Down
	const int down = center + width;
	const int downdown = center + 3 * width;

	// Right
	const int right = center + 1;
	const int rightright = center + 3;

	if (costs[center] < 0.001f) {
		return;
	}

	if (p.y > 0 && helper->weak_info_cuda[up] == STRONG) {
		filter[index++] = plane_hypotheses[up].w;
	}
	if (p.y > 2 && helper->weak_info_cuda[upup] == STRONG) {
		filter[index++] = plane_hypotheses[upup].w;
	}
	if (p.y > 4 && helper->weak_info_cuda[upup - width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[upup - width * 2].w;
	}
	if (p.y < height - 1 && helper->weak_info_cuda[down] == STRONG) {
		filter[index++] = plane_hypotheses[down].w;
	}
	if (p.y < height - 3 && helper->weak_info_cuda[downdown] == STRONG) {
		filter[index++] = plane_hypotheses[downdown].w;
	}
	if (p.y < height - 5 && helper->weak_info_cuda[downdown + width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[downdown + width * 2].w;
	}
	if (p.x > 0 && helper->weak_info_cuda[left] == STRONG) {
		filter[index++] = plane_hypotheses[left].w;
	}
	if (p.x > 2 && helper->weak_info_cuda[leftleft] == STRONG) {
		filter[index++] = plane_hypotheses[leftleft].w;
	}
	if (p.x > 4 && helper->weak_info_cuda[leftleft - 2] == STRONG) {
		filter[index++] = plane_hypotheses[leftleft - 2].w;
	}
	if (p.x < width - 1 && helper->weak_info_cuda[right] == STRONG) {
		filter[index++] = plane_hypotheses[right].w;
	}
	if (p.x < width - 3 && helper->weak_info_cuda[rightright] == STRONG) {
		filter[index++] = plane_hypotheses[rightright].w;
	}
	if (p.x < width - 5 && helper->weak_info_cuda[rightright + 2] == STRONG) {
		filter[index++] = plane_hypotheses[rightright + 2].w;
	}
	if (p.y > 0 && p.x < width - 2 && helper->weak_info_cuda[up + 2] == STRONG) {
		filter[index++] = plane_hypotheses[up + 2].w;
	}
	if (p.y < height - 1 && p.x < width - 2 && helper->weak_info_cuda[down + 2] == STRONG) {
		filter[index++] = plane_hypotheses[down + 2].w;
	}
	if (p.y > 0 && p.x > 1 && helper->weak_info_cuda[up - 2] == STRONG)
	{
		filter[index++] = plane_hypotheses[up - 2].w;
	}
	if (p.y < height - 1 && p.x>1 && helper->weak_info_cuda[down - 2] == STRONG) {
		filter[index++] = plane_hypotheses[down - 2].w;
	}
	if (p.x > 0 && p.y > 2 && helper->weak_info_cuda[left - width * 2] == STRONG)
	{
		filter[index++] = plane_hypotheses[left - width * 2].w;
	}
	if (p.x < width - 1 && p.y>2 && helper->weak_info_cuda[right - width * 2] == STRONG)
	{
		filter[index++] = plane_hypotheses[right - width * 2].w;
	}
	if (p.x > 0 && p.y < height - 2 && helper->weak_info_cuda[left + width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[left + width * 2].w;
	}
	if (p.x < width - 1 && p.y < height - 2 && helper->weak_info_cuda[right + width * 2] == STRONG) {
		filter[index++] = plane_hypotheses[right + width * 2].w;
	}

	sort_small(filter, index);
	int median_index = index / 2;
	if (index % 2 == 0) {
		plane_hypotheses[center].w = (filter[median_index - 1] + filter[median_index]) / 2;
	}
	else {
		plane_hypotheses[center].w = filter[median_index];
	}
}

__global__ void BlackPixelFilterStrong(DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2;
	}
	else {
		p.y = p.y * 2 + 1;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] != WEAK) {
		CheckerboardFilterStrong(p, helper);
	}
}

__global__ void RedPixelFilterStrong(DataPassHelper* helper)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (threadIdx.x % 2 == 0) {
		p.y = p.y * 2 + 1;
	}
	else {
		p.y = p.y * 2;
	}
	if (p.x >= helper->width || p.y >= helper->height) {
		return;
	}
	if (helper->weak_info_cuda[p.x + p.y * helper->width] != WEAK) {
		CheckerboardFilterStrong(p, helper);
	}
}

__global__ void GenNeighbours(DataPassHelper* helper)
{
	int width = helper->width;
	int height = helper->height;
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (point.x >= width || point.y >= height) {
		return;
	}
	const unsigned center = point.x + point.y * width;
	const uchar* weak_info = helper->weak_info_cuda;

	if (weak_info[center] != WEAK) {
		// not weak point return
		return;
	}

	const int max_pt_num = 160;
	const int min_margin = 6;
	const float depth_diff = helper->params->depth_max - helper->params->depth_min;
	const int* neighbours_map = helper->neighbours_map_cuda;
	const PatchMatchParams* params = helper->params;
	const short2* weak_nearest_strong = helper->weak_nearest_strong;
	const Camera camera = helper->cameras_cuda[0];
	const unsigned offset = neighbours_map[center] * NEIGHBOUR_NUM;
	const float4* plane_hypotheses = helper->plane_hypotheses_cuda;

	const int* label_mask = helper->label_cuda;
	curandState* rand_state = &(helper->rand_states_cuda[center]);
	short2* neighbours = &(helper->neighbours_cuda[offset]);
	uchar* weak_reliable = &(helper->weak_reliable_cuda[center]);
	// init for invalid points
	for (int i = 0; i < NEIGHBOUR_NUM; ++i) {
		neighbours[i].x = -1;
		neighbours[i].y = -1;
	}
	neighbours[0] = make_short2(point.x, point.y); // the first point is the center point
	short2 strong_points[max_pt_num];
	bool dir_valid[max_pt_num];
	for (int i = 0; i < max_pt_num; ++i) {
		strong_points[i] = make_short2(-1, -1);
		dir_valid[i] = false;
	}
	int origin_direction_index = -1;
	int strong_point_size = 0;

	const int rotate_time = params->rotate_time; // max is 4 from [1, 2, 4] 
	const float angle = 45.0f / rotate_time;
	const float cos_angle = cos(angle * M_PI / 180.f);
	const float sin_angle = sin(angle * M_PI / 180.f);
	const float threshhold = cos((angle / 2.0f) * M_PI / 180.0f);
	const int shift_range = MAX((int)(tan((angle / 2.0f) * M_PI / 180.0f) * 20), 1);
	const float ransac_threshold = params->ransac_threshold;

	bool edge_limit = false;
	if (params->use_limit) {
		edge_limit = true;
		if (params->use_edge) {
			const unsigned complex_offset = neighbours_map[center];
			float complex_val = helper->complex_cuda[complex_offset];
			const float rand_prob = curand_uniform(rand_state) - FLT_EPSILON;
			if (rand_prob < complex_val) edge_limit = false;
		}
	}

	for (int origin_direction_x = -1; origin_direction_x <= 1; ++origin_direction_x) {
		for (int origin_direction_y = -1; origin_direction_y <= 1; ++origin_direction_y) {
			if (origin_direction_x == 0 && origin_direction_y == 0) {
				continue;
			}
			float2 origin_direction = make_float2(origin_direction_x, origin_direction_y);
			NormalizeVec2(&origin_direction);
			origin_direction_index++;
			for (int rotate_iter = 0; rotate_iter < rotate_time; ++rotate_iter) {
				int dir_index = origin_direction_index * 4 + rotate_iter;
				for (int radius = 2; radius <= MAX_SEARCH_RADIUS; radius = MIN(radius * 2, radius + 25)) {
					float2 test_pt = make_float2(point.x + origin_direction.x * radius, point.y + origin_direction.y * radius);
					if (test_pt.x < 0 || test_pt.y < 0 || test_pt.x >= width || test_pt.y >= height) {
						break;
					}
					for (int radius_iter = 0; radius_iter < 4; ++radius_iter) {
						int rand_x_shift = (curand(rand_state) % 2 == 0 ? 1 : -1) * curand(rand_state) % shift_range;
						int rand_y_shift = (curand(rand_state) % 2 == 0 ? 1 : -1) * curand(rand_state) % shift_range;
						float2 direction = make_float2(origin_direction.x * 20 + rand_x_shift, origin_direction.y * 20 + rand_y_shift);
						NormalizeVec2(&direction);
						short2 neighbour_pt = make_short2(point.x + direction.x * radius, point.y + direction.y * radius);
						if (neighbour_pt.x < min_margin || neighbour_pt.y < min_margin || neighbour_pt.x >= width - min_margin || neighbour_pt.y >= height - min_margin) {
							continue;
						}
						int neighbour_pt_center = neighbour_pt.x + neighbour_pt.y * width;
						if (weak_info[neighbour_pt_center] != STRONG) {
							neighbour_pt = weak_nearest_strong[neighbour_pt_center];
							if (neighbour_pt.x == -1 || neighbour_pt.y == -1) {
								continue;
							}
							neighbour_pt_center = neighbour_pt.x + neighbour_pt.y * width;
						}
						bool has_same_pt = false;
						for (int k = 0; k < dir_index; k++) {
							if (strong_points[k].x == neighbour_pt.x && strong_points[k].y == neighbour_pt.y) {
								has_same_pt = true;
								break;
							}
						}
						if (has_same_pt) continue;
						float2 test_direction = make_float2(neighbour_pt.x - point.x, neighbour_pt.y - point.y);
						NormalizeVec2(&test_direction);
						float cos_angle = Vec2DotVec2(test_direction, origin_direction);
						if (cos_angle > threshhold && (!edge_limit || !BresenhamLine(point, make_int2(neighbour_pt.x, neighbour_pt.y), helper))) {
							strong_points[dir_index] = neighbour_pt;
							dir_valid[dir_index] = true;
							strong_point_size++;
							break;
						}
					}
					if (dir_valid[dir_index]) {
						break;
					}
				}
				// rotate
				{
					float2 rotated_direction;
					rotated_direction.x = origin_direction.x * cos_angle - origin_direction.y * sin_angle;
					rotated_direction.y = origin_direction.x * sin_angle + origin_direction.y * cos_angle;
					NormalizeVec2(&rotated_direction);
					origin_direction = rotated_direction;
				}
			}
		}
	}

	int extend_index = 31;
	if (params->use_label && label_mask[center] > 0) {
		const int dir[LAB_BOUNDARY_NUM * 2][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1}, {1, 0.5}, {0.5, 1}, {-0.5, 1}, {-1, 0.5}, {-1, -0.5}, {-0.5, -1}, {0.5, -1}, {1, -0.5} };
		const int laboff = neighbours_map[center] * LAB_BOUNDARY_NUM;
		short2* lab_bound = &(helper->label_boundary_cuda[laboff]);
		float bound_dist[LAB_BOUNDARY_NUM * 2] = { 0 };
		int dir_step[LAB_BOUNDARY_NUM * 2] = { 0 };
		for (int i = 0; i < LAB_BOUNDARY_NUM; ++i) {
			short2 boundary_pt = lab_bound[i];
			float dist = 0.0f;
			if (boundary_pt.x != -1 && boundary_pt.y != -1) {
				dist = std::sqrt(std::pow(point.x - boundary_pt.x, 2) + std::pow(point.y - boundary_pt.y, 2));
				if (i >= 4) dist /= std::sqrt(2.0);
			}
			bound_dist[i] = dist;
			if (i % 2 == 1) {
				float opposite_dist = bound_dist[i - 1];
				int step = MIN(1, MAX(4 * rotate_time - 1, (int)(4 * rotate_time * dist / (dist + opposite_dist))));
				int opposite_step = 4 * rotate_time - step;
				dir_step[i - 1] = opposite_step;
				dir_step[i] = step;
			}
		}

		dir_step[8] = (dir_step[3] + dir_step[5]) / 2;
		bound_dist[8] = (bound_dist[3] + bound_dist[5]) / 2;

		dir_step[9] = (dir_step[1] + dir_step[5]) / 2;
		bound_dist[9] = (bound_dist[1] + bound_dist[5]) / 2;

		dir_step[10] = (dir_step[1] + dir_step[6]) / 2;
		bound_dist[10] = (bound_dist[1] + bound_dist[6]) / 2;

		dir_step[11] = (dir_step[2] + dir_step[6]) / 2;
		bound_dist[11] = (bound_dist[2] + bound_dist[6]) / 2;

		dir_step[12] = (dir_step[2] + dir_step[4]) / 2;
		bound_dist[12] = (bound_dist[2] + bound_dist[4]) / 2;

		dir_step[13] = (dir_step[4] + dir_step[0]) / 2;
		bound_dist[13] = (bound_dist[4] + bound_dist[0]) / 2;

		dir_step[14] = (dir_step[7] + dir_step[0]) / 2;
		bound_dist[14] = (bound_dist[7] + bound_dist[0]) / 2;

		dir_step[15] = (dir_step[7] + dir_step[3]) / 2;
		bound_dist[15] = (bound_dist[7] + bound_dist[3]) / 2;

		for (int i = 0; i < LAB_BOUNDARY_NUM * 2; ++i) {
			float dist = bound_dist[i];
			int gap_num = dir_step[i] + 1;
			int step_len = MAX(1, (int)floor(1.0 * dist / gap_num));
			for (int step = 1; step <= dir_step[i]; ++step) {
				short2 neighbour_pt = make_short2(point.x + step * step_len * dir[i][0], point.y + step * step_len * dir[i][1]);
				if (neighbour_pt.x < min_margin || neighbour_pt.y < min_margin || neighbour_pt.x >= width - min_margin || neighbour_pt.y >= height - min_margin) {
					continue;
				}
				int neighbour_pt_center = neighbour_pt.x + neighbour_pt.y * width;
				if (weak_info[neighbour_pt_center] != STRONG) {
					neighbour_pt = weak_nearest_strong[neighbour_pt_center];
					if (neighbour_pt.x == -1 || neighbour_pt.y == -1) {
						continue;
					}
					neighbour_pt_center = neighbour_pt.x + neighbour_pt.y * width;
				}
				bool has_same_pt = false;
				for (int k = 0; k <= extend_index; k++) {
					if (strong_points[k].x == neighbour_pt.x && strong_points[k].y == neighbour_pt.y) {
						has_same_pt = true;
						break;
					}
				}
				if (has_same_pt) continue;
				extend_index++;
				strong_points[extend_index] = neighbour_pt;
				dir_valid[extend_index] = true;
				strong_point_size++;
			}
		}
	}

	if (strong_point_size <= 3) {
		*weak_reliable = 0;
		return;
	}

	float4 best_plane;
	bool has_valid_plane = false;
	short2 strong_points_valid[max_pt_num];
	float3 strong_points_valid_3d[max_pt_num];
	float3 strong_points_valid_normals[max_pt_num];
	int valid_count = 0;
	float X[3];
	Get3DPoint(camera, point, plane_hypotheses[center].w, X);
	float3 center_point_world = make_float3(X[0], X[1], X[2]);
	for (int i = 0; i < max_pt_num; ++i) {
		strong_points_valid[i] = make_short2(-1, -1);
		if (dir_valid[i]) {
			const auto& strong_point = strong_points[i];
			int strong_point_center = strong_point.x + strong_point.y * width;
			strong_points_valid[valid_count] = strong_points[i];
			Get3DPoint(camera, strong_point, plane_hypotheses[strong_point_center].w, X);
			strong_points_valid_3d[valid_count] = make_float3(X[0], X[1], X[2]);
			float4 n4 = TransformNormal2RefCam(camera, plane_hypotheses[strong_point_center]);
			strong_points_valid_normals[valid_count] = make_float3(n4.x, n4.y, n4.z);
			valid_count++;
		}
	}

	{
		// RANSAC to find a good plane
		int iteration = 300, max_iter = 200;
		float min_cost = FLT_MAX;
		int max_count = 3;
		uchar edge_test[max_pt_num][max_pt_num] = { 0 };
		bool has_strong_plane = false;
		while (iteration > 0 && max_iter > 0) {
			max_iter--;
			int a_index = curand(rand_state) % valid_count;
			int b_index = curand(rand_state) % valid_count;
			int c_index = curand(rand_state) % valid_count;
			if (a_index == b_index || b_index == c_index || a_index == c_index) {
				continue;
			}
			// compute triangle
			if (!PointinTriangle(strong_points_valid[a_index], strong_points_valid[b_index], strong_points_valid[c_index], point)) {
				continue;
			}
			if (edge_limit) {
				if (edge_test[a_index][b_index] == 0) {
					edge_test[a_index][b_index] = edge_test[b_index][a_index] =
						(BresenhamLine(strong_points_valid[a_index], strong_points_valid[b_index], helper) ? 1 : 2);
				}
				if (edge_test[b_index][c_index] == 0) {
					edge_test[b_index][c_index] = edge_test[c_index][b_index] =
						(BresenhamLine(strong_points_valid[b_index], strong_points_valid[c_index], helper) ? 1 : 2);
				}
				if (edge_test[c_index][a_index] == 0) {
					edge_test[c_index][a_index] = edge_test[a_index][c_index] =
						(BresenhamLine(strong_points_valid[c_index], strong_points_valid[a_index], helper) ? 1 : 2);
				}
				if (edge_test[a_index][b_index] == 1 || edge_test[b_index][c_index] == 1 || edge_test[c_index][a_index] == 1) {
					continue;
				}
			}
			const float3& AN = strong_points_valid_normals[a_index];
			const float3& BN = strong_points_valid_normals[a_index];
			const float3& CN = strong_points_valid_normals[a_index];
			if (Vec3DotVec3(AN, BN) < 0.9f || Vec3DotVec3(AN, CN) < 0.9f || Vec3DotVec3(BN, CN) < 0.9f) {
				continue;
			}

			const float3& A = strong_points_valid_3d[a_index];
			const float3& B = strong_points_valid_3d[b_index];
			const float3& C = strong_points_valid_3d[c_index];
			float3 A_C = make_float3(A.x - C.x, A.y - C.y, A.z - C.z);
			float3 B_C = make_float3(B.x - C.x, B.y - C.y, B.z - C.z);
			float4 cross_vec;
			cross_vec.x = A_C.y * B_C.z - B_C.y * A_C.z;
			cross_vec.y = -(A_C.x * B_C.z - B_C.x * A_C.z);
			cross_vec.z = A_C.x * B_C.y - B_C.x * A_C.y;
			if ((cross_vec.x == 0 && cross_vec.y == 0 && cross_vec.z == 0) || isnan(cross_vec.x) || isnan(cross_vec.y) || isnan(cross_vec.z)) {
				continue;
			}
			iteration--;
			NormalizeVec3(&cross_vec);
			cross_vec.w = -(cross_vec.x * A.x + cross_vec.y * A.y + cross_vec.z * A.z);

			bool is_strong_plane = true;
			if (params->use_label && label_mask[center] > 0 && fabs(Vec3DotVec3(AN, cross_vec)) < 0.9f && fabs(Vec3DotVec3(BN, cross_vec)) < 0.9f && fabs(Vec3DotVec3(CN, cross_vec)) < 0.9f) {
				is_strong_plane = false;
			}
			if (has_strong_plane && !is_strong_plane) continue;

			int temp_count = 0;
			float strong_dist = 0.0f;
			for (int strong_index = 0; strong_index < valid_count; ++strong_index) {
				const float3& temp_point = strong_points_valid_3d[strong_index];
				const short2& temp_pos = strong_points_valid[strong_index];
				float factor_x = (temp_pos.x - camera.K[2]) / camera.K[0];
				float factor_y = (temp_pos.y - camera.K[5]) / camera.K[4];
				float fit_depth = -cross_vec.w / (cross_vec.x * factor_x + cross_vec.y * factor_y + cross_vec.z);
				float distance = fabs(fit_depth - temp_point.z);
				if (distance / depth_diff < ransac_threshold) {
					temp_count++;
					strong_dist += distance;
				}
			}
			if (temp_count < 6) {
				continue;
			}
			if (temp_count > max_count || (!has_strong_plane && is_strong_plane)) {
				if (!has_strong_plane && is_strong_plane) {
					has_strong_plane = true;
				}
				float factor_x = (point.x - camera.K[2]) / camera.K[0];
				float factor_y = (point.y - camera.K[5]) / camera.K[4];
				float fit_depth = -cross_vec.w / (cross_vec.x * factor_x + cross_vec.y * factor_y + cross_vec.z);
				const float center_distance = fabs(fit_depth - center_point_world.z);
				best_plane = cross_vec;
				max_count = temp_count;
				strong_dist /= temp_count;
				min_cost = center_distance;
				has_valid_plane = true;
			}
			else if (temp_count == max_count) {
				float factor_x = (point.x - camera.K[2]) / camera.K[0];
				float factor_y = (point.y - camera.K[5]) / camera.K[4];
				float fit_depth = -cross_vec.w / (cross_vec.x * factor_x + cross_vec.y * factor_y + cross_vec.z);
				const float center_distance = fabs(fit_depth - center_point_world.z);
				if (center_distance < min_cost) {
					best_plane = cross_vec;
					max_count = temp_count;
					strong_dist /= temp_count;
					min_cost = center_distance;
				}
			}
		}
	}

	float weight[max_pt_num];
	if (!has_valid_plane) {
		*weak_reliable = 0;
		return;
	}

	for (int i = 0; i < valid_count; ++i) {
		const float3& temp_point = strong_points_valid_3d[i];
		const short2& temp_pos = strong_points_valid[i];
		float factor_x = (temp_pos.x - camera.K[2]) / camera.K[0];
		float factor_y = (temp_pos.y - camera.K[5]) / camera.K[4];
		float fit_depth = -best_plane.w / (best_plane.x * factor_x + best_plane.y * factor_y + best_plane.z);
		float distance = fabs(fit_depth - temp_point.z);
		if (distance / depth_diff >= ransac_threshold) {
			strong_points_valid[i] = make_short2(-1, -1);
			weight[i] = FLT_MAX;
			continue;
		}
		if (distance / depth_diff >= ransac_threshold) {
			strong_points_valid[i] = make_short2(-1, -1);
			weight[i] = FLT_MAX;
			continue;
		}
		weight[i] = distance;
	}

	sort_small_weighted(strong_points_valid, weight, valid_count);
	for (int i = 1; i < NEIGHBOUR_NUM; ++i) {
		neighbours[i] = strong_points_valid[i - 1];
	}
	*weak_reliable = 1;
}

__global__ void NeigbourUpdate(
	DataPassHelper* helper
) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}
	const int center = point.x + point.y * width;
	if (helper->weak_info_cuda[center] != WEAK) {
		return;
	}
	if (helper->weak_reliable_cuda[center] != 1) {
		helper->weak_info_cuda[center] = UNKNOWN;
	}
}

__global__ void GenEdgeInform(
	DataPassHelper* helper
) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}

	const int center = point.x + point.y * width;
	const unsigned* selected_views = helper->selected_views_cuda;

	/////////////////////////////////////////////////////////////////////////////////////////////
	//��ȡÿ�����������
	for (int src_idx = 1; src_idx < helper->params->num_images; ++src_idx) {
		float4* plane_hypotheses = helper->plane_hypotheses_cuda;
		const cudaTextureObject_t ref_image = helper->texture_objects_cuda[0].images[0];

		int radius = helper->params->weak_radius;
		Point regions[12][20];
		int regionCounts[12] = { 0 }; // ÿ�������ڵĵ�������
		int idx = 0;
		const float ref_center_pix = tex2D<float>(ref_image, point.x + 0.5f, point.y + 0.5f);
		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++) {
				if (i == 0 && j == 0)
					continue;
				float angle = calculateAngle(i, j);
				const int2 ref_pt = make_int2(point.x + i, point.y + j);
				if (ref_pt.x >= 0 && ref_pt.x < width && ref_pt.y >= 0 && ref_pt.y < height) {
					int nei_center = ref_pt.x + ref_pt.y * width;
					if (isSet(selected_views[nei_center], src_idx - 1) == 1) {
						const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
						float weight = ComputeBilateralWeight_YZL(i, j, ref_pix, ref_center_pix, helper->params->sigma_spatial, helper->params->sigma_color);
						Point p; p.i = i; p.j = j; p.angle = angle; p.weight = weight;
						int region = getRegion(angle);
						regions[region][regionCounts[region]++] = p;
						idx++;
					}
					else {
						continue;
					}
				}
			}
		}

		Point min_regions[12];
		for (int i = 0; i < 12; ++i) {
			bubbleSort(regions[i], regionCounts[i]);
		}
		for (int i = 0; i < 12; ++i) {
			min_regions[i] = regions[i][0];
		}
		bubbleSort(min_regions, 12);

		short2* candidate = helper->candidate_cuda;

		int ind = src_idx - 1;
		for (int k = 0; k < LAB_BOUNDARY_NUM; k++) {
			candidate[center * LAB_BOUNDARY_NUM * NUM_IMAGES + ind * LAB_BOUNDARY_NUM + k].x = min_regions[k].i;
			candidate[center * LAB_BOUNDARY_NUM * NUM_IMAGES + ind * LAB_BOUNDARY_NUM + k].y = min_regions[k].j;
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////


	// ��ȡ��ǰ������Χ�����edge
	const unsigned offset = center * EDGE_NEIGH_NUM;
	const uchar* edge = helper->edge_cuda;
	const int dir[EDGE_NEIGH_NUM][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1} };
	if (helper->params->use_edge) {
		short2* edge_neigh = &(helper->edge_neigh_cuda[offset]);

		for (int i = 0; i < EDGE_NEIGH_NUM; i++) {
			edge_neigh[i].x = -1;
			edge_neigh[i].y = -1;

			int dx = dir[i][0], dy = dir[i][1];
			int nx = point.x + dx, ny = point.y + dy;
			while (true) {
				if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
					break;
				}
				if (edge[nx + ny * width]) {
					edge_neigh[i].x = nx;
					edge_neigh[i].y = ny;
					break;
				}
				nx += dx;
				ny += dy;
			}
		}

		if (helper->weak_info_cuda[center] == WEAK) {
			float* complex = helper->complex_cuda;
			int radius = helper->params->strong_radius;
			int edge_pix = 0, tot_pix = 0;
			for (int i = -radius; i <= radius; i++) {
				for (int j = -radius; j <= radius; j++) {
					int nx = point.x + i;
					int ny = point.y + j;
					if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
						continue;
					}
					if (edge[ny * width + nx]) edge_pix++;
					tot_pix++;
				}
			}
			float density = 1.0f * edge_pix / tot_pix;

			const unsigned complex_offset = helper->neighbours_map_cuda[center];
			complex[complex_offset] = 1.0f / (1.0f + exp(-25.0 * (density - 0.35))); // ��һ��λ��������������ĸ���
		}

		if (helper->params->state == REFINE_INIT && helper->params->use_detail && edge[center]) {
			if (helper->weak_info_cuda[center] != STRONG) helper->weak_info_cuda[center] = UNKNOWN;
		}
	}

	const unsigned laboff = helper->neighbours_map_cuda[center] * LAB_BOUNDARY_NUM;
	const int* label_mask = helper->label_cuda;
	const int labdir[LAB_BOUNDARY_NUM][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1} };
	if (helper->params->use_label && helper->weak_info_cuda[center] == WEAK) {
		short2* lab_bound = &(helper->label_boundary_cuda[laboff]);
		int center_label = label_mask[center];
		if (center_label > 0) {
			for (int i = 0; i < LAB_BOUNDARY_NUM; i++) {
				lab_bound[i].x = -1;
				lab_bound[i].y = -1;

				int dx = labdir[i][0], dy = labdir[i][1];
				int nx = point.x + dx, ny = point.y + dy;
				int last_x = -1, last_y = -1;
				while (true) {
					if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
						break;
					}
					int next_label = label_mask[nx + ny * width];
					if (next_label == center_label) {
						last_x = nx;
						last_y = ny;
					}
					else if (next_label == -1) {
						break;
					}
					nx += dx;
					ny += dy;
				}
				// �ڸ÷��������һ��ͬ�����������Ϊ�߽�
				lab_bound[i].x = last_x;
				lab_bound[i].y = last_y;
			}
		}
		if (helper->params->state == REFINE_INIT && helper->params->use_detail && label_mask[center] == 0) {
			if (helper->weak_info_cuda[center] != STRONG) helper->weak_info_cuda[center] = UNKNOWN;
		}
	}
}

__global__ void DepthToWeak(DataPassHelper* helper) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}

	const int min_margin = 6;
	const int center = point.x + point.y * width;

	if (helper->params->use_radius && helper->radius_cuda[center] == 0) {
		helper->radius_cuda[center] = helper->params->strong_radius;
	}

	if (point.x < min_margin || point.y < min_margin || point.x >= width - min_margin || point.y >= height - min_margin) {
		helper->weak_info_cuda[center] = UNKNOWN;
		return;
	}
	const auto& image = helper->texture_objects_cuda[0].images[0];
	const float center_pix = tex2D<float>(image, point.x + 0.5f, point.y + 0.5f);

	const Camera* cameras = helper->cameras_cuda;
	const unsigned* selected_views = helper->selected_views_cuda;
	const int num_images = helper->params->num_images;
	const uchar* view_weight = &(helper->view_weight_cuda[MAX_IMAGES * center]);
	float4 origin_plane_hypothesis;
	origin_plane_hypothesis = helper->plane_hypotheses_cuda[center];
	origin_plane_hypothesis = TransformNormal2RefCam(cameras[0], origin_plane_hypothesis);
	float origin_depth = origin_plane_hypothesis.w;
	if (origin_depth == 0) {
		helper->weak_info_cuda[center] = UNKNOWN;
		return;
	}
	// compute cost now and baseline

	float cost_now = 0.0f;
	float base_line = 0;
	int valid_neighbour = 0;
	float weight_normal = 0.0f;
	for (int src_index = 1; src_index < num_images; ++src_index) {
		int view_index = src_index - 1;
		if (isSet(selected_views[center], view_index)) {
			float4 temp_plane_hypothesis = origin_plane_hypothesis;
			temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], point, origin_depth, temp_plane_hypothesis);
			float temp_cost = ComputeBilateralNCCOld(point, src_index, temp_plane_hypothesis, helper);
			if (helper->params->geom_consistency) {
				temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_plane_hypothesis, helper);
			}
			cost_now += (temp_cost * view_weight[view_index]);
			weight_normal += view_weight[view_index];
			float c_dist[3];
			c_dist[0] = cameras[0].c[0] - cameras[src_index].c[0];
			c_dist[1] = cameras[0].c[1] - cameras[src_index].c[1];
			c_dist[2] = cameras[0].c[2] - cameras[src_index].c[2];
			double temp_val = c_dist[0] * c_dist[0] + c_dist[1] * c_dist[1] + c_dist[2] * c_dist[2];
			base_line += sqrtf(temp_val);
			valid_neighbour++;
		}
	}
	if (valid_neighbour == 0) {
		helper->weak_info_cuda[center] = UNKNOWN;
		return;
	}

	cost_now /= weight_normal;
	base_line /= valid_neighbour;

	float disp = cameras[0].K[0] * base_line / origin_depth;
	const int radius = 30;
	const int p_costs_size = 2 * radius + 1;
	float p_costs[p_costs_size];
	int increment = 1;

	for (int p_disp = -radius * increment; p_disp <= radius * increment; p_disp += increment) {

		float p_depth = cameras[0].K[0] * base_line / (disp + p_disp);
		if (p_depth < helper->params->depth_min || p_depth > helper->params->depth_max) {
			p_costs[p_disp + radius] = 2.0f;
			continue;
		}
		float4 temp_plane_hypothesis = origin_plane_hypothesis;
		temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], point, p_depth, temp_plane_hypothesis);
		float p_cost = 0.0f;
		for (int src_index = 1; src_index < num_images; ++src_index) {
			int view_index = src_index - 1;
			float temp_cost = 0.0f;
			if (isSet(selected_views[center], view_index)) {
				temp_cost += ComputeBilateralNCCOld(point, src_index, temp_plane_hypothesis, helper);
				if (helper->params->geom_consistency) {
					temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_plane_hypothesis, helper);
				}
				p_cost += (temp_cost * view_weight[view_index]);
			}
		}
		p_cost /= weight_normal;
		p_costs[p_disp + radius] = MIN(2.0f, p_cost);
	}
#ifdef DEBUG_COST_LINE
	{
		float* weak_ncc_cost = &(helper->weak_ncc_cost_cuda[(size_t)center * 61]);
		for (int i = 0; i < p_costs_size; ++i) {
			weak_ncc_cost[i] = p_costs[i];
		}
	}
#endif // DEBUG_COST_LINE
	// find peaks
	bool is_peak[p_costs_size];
	for (int i = 0; i < p_costs_size; ++i) {
		is_peak[i] = false;
	}

	int peak_count = 0;
	int min_peak = 0;
	float min_cost = 2.0f;
	for (int i = 2; i < p_costs_size - 2; ++i) {
		if (p_costs[i - 1] > p_costs[i] && p_costs[i + 1] > p_costs[i]) {
			is_peak[i] = true;
			peak_count++;
			if (p_costs[i] < min_cost) {
				min_peak = i;
				min_cost = p_costs[i];
			}
		}
	}

	if (abs(min_peak - radius) > helper->params->weak_peak_radius || p_costs[min_peak] > 0.5f) {
		helper->weak_info_cuda[center] = WEAK;
		return;
	}

	if (peak_count == 1) {
		if (p_costs[min_peak] <= 0.15f) {
			helper->weak_info_cuda[center] = STRONG;
		}
		else {
			helper->weak_info_cuda[center] = WEAK;
		}
		return;
	}

	float var = 0.0f;
	for (int i = 2; i < p_costs_size - 2; ++i) {
		if (is_peak[i] && i != min_peak) {
			float dist = p_costs[i] - min_cost;
			var += dist * dist;

		}
	}
	var = sqrtf(var);
	var /= (peak_count - 1);

	if (var > 0.2f) {
		helper->weak_info_cuda[center] = STRONG;
	}
	else {
		helper->weak_info_cuda[center] = WEAK;
	}

}

__global__ void LocalRefine(DataPassHelper* helper) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}

	const int center = point.x + point.y * width;

	const Camera* cameras = helper->cameras_cuda;
	const unsigned* selected_views = helper->selected_views_cuda;
	const int num_images = helper->params->num_images;
	const uchar* view_weight = &(helper->view_weight_cuda[MAX_IMAGES * center]);
	float4 origin_plane_hypothesis;
	origin_plane_hypothesis = helper->plane_hypotheses_cuda[center];
	origin_plane_hypothesis = TransformNormal2RefCam(cameras[0], origin_plane_hypothesis);
	float origin_depth = origin_plane_hypothesis.w;
	if (origin_depth == 0) {
		return;
	}

	// compute cost now and baseline
	float cost_now = 0.0f;
	float base_line = 0;
	int valid_neighbour = 0;
	float weight_normal = 0.0f;
	for (int src_index = 1; src_index < num_images; ++src_index) {
		int view_index = src_index - 1;
		if (isSet(selected_views[center], view_index)) {
			float4 temp_plane_hypothesis = origin_plane_hypothesis;
			temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], point, origin_depth, temp_plane_hypothesis);
			float temp_cost = ComputeBilateralNCCOld(point, src_index, temp_plane_hypothesis, helper);
			if (helper->params->geom_consistency) {
				temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_plane_hypothesis, helper);
			}
			cost_now += (temp_cost * view_weight[view_index]);
			weight_normal += view_weight[view_index];
			float c_dist[3];
			c_dist[0] = cameras[0].c[0] - cameras[src_index].c[0];
			c_dist[1] = cameras[0].c[1] - cameras[src_index].c[1];
			c_dist[2] = cameras[0].c[2] - cameras[src_index].c[2];
			double temp_val = c_dist[0] * c_dist[0] + c_dist[1] * c_dist[1] + c_dist[2] * c_dist[2];
			base_line += sqrtf(temp_val);
			valid_neighbour++;
		}
	}

	if (weight_normal == 0 || valid_neighbour == 0) {
		return;
	}

	cost_now /= weight_normal;
	base_line /= valid_neighbour;

	float disp = cameras[0].K[0] * base_line / origin_depth;
	const int radius = 5;

	float min_cost = 2.0f;
	float best_depth = origin_depth;
	for (int p_disp = -radius; p_disp <= radius; ++p_disp) {
		float p_depth = cameras[0].K[0] * base_line / (disp + p_disp);
		if (p_depth < helper->params->depth_min || p_depth > helper->params->depth_max) {
			continue;
		}
		float4 temp_plane_hypothesis = origin_plane_hypothesis;
		temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], point, p_depth, temp_plane_hypothesis);
		float temp_cost = 0.0f;
		for (int src_index = 1; src_index < num_images; ++src_index) {
			int view_index = src_index - 1;
			if (isSet(selected_views[center], view_index)) {
				temp_cost += (ComputeBilateralNCCOld(point, src_index, temp_plane_hypothesis, helper) * view_weight[view_index]);
				if (helper->params->geom_consistency) {
					temp_cost += (helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_plane_hypothesis, helper) * view_weight[view_index]);
				}
			}
		}
		temp_cost /= weight_normal;
		if (temp_cost < min_cost) {
			min_cost = temp_cost;
			best_depth = p_depth;
		}
	}
	if (cost_now - min_cost > 0.1) {
		helper->plane_hypotheses_cuda[center].w = best_depth;
	}
}

__global__ void FindNearestStrongPointInit(DataPassHelper* helper) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}
	const uchar* weak_info = helper->weak_info_cuda;
	short2* weak_nearest_strong = helper->weak_nearest_strong;
	const int center = point.x + point.y * width;
	weak_nearest_strong[center].x = -1;
	weak_nearest_strong[center].y = -1;
	if (weak_info[center] != WEAK) {
		return;
	}
}


__global__ void FindNearestStrongPoint(DataPassHelper* helper) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}
	const uchar* weak_info = helper->weak_info_cuda;
	short2* weak_nearest_strong = helper->weak_nearest_strong;
	const int center = point.x + point.y * width;
	weak_nearest_strong[center].x = -1;
	weak_nearest_strong[center].y = -1;
	if (weak_info[center] != WEAK) {
		return;
	}

	const int max_radius = 100; // ETH 100
	for (int radius = 0; radius <= max_radius; ++radius) {
		for (int x = -radius; x <= radius; ++x) {
			for (int y = -radius; y <= radius; ++y) {
				if (abs(x) != radius && abs(y) != radius) continue;
				const int2 neighbour_pt = make_int2(point.x + x, point.y + y);
				if (neighbour_pt.x < 0 || neighbour_pt.y < 0 || neighbour_pt.x >= width || neighbour_pt.y >= height) {
					continue;
				}
				const int neighbour_center = neighbour_pt.x + neighbour_pt.y * width;
				if (weak_info[neighbour_center] == STRONG) {
					weak_nearest_strong[center].x = neighbour_pt.x;
					weak_nearest_strong[center].y = neighbour_pt.y;
					return;
				}
			}
		}
	}
}

__global__ void RANSACToGetFitPlane(DataPassHelper* helper) {
	const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int width = helper->width;
	const int height = helper->height;
	if (point.x >= width || point.y >= height) {
		return;
	}
	const uchar* weak_info = helper->weak_info_cuda;
	const int center = point.x + point.y * width;
	const PatchMatchParams* params = helper->params;
	float4* plane_hypotheses = helper->plane_hypotheses_cuda;
	float4* fit_plane_hypothese = helper->fit_plane_hypotheses_cuda;

	if (weak_info[center] != WEAK) {
		fit_plane_hypothese[center] = plane_hypotheses[center];
		return;
	}
	// make sure that the plane is in the ref camera coord
	curandState* rand_state = &(helper->rand_states_cuda[center]);
	const auto& camera = helper->cameras_cuda[0];

	bool edge_limit = false;
	if (params->use_limit) {
		edge_limit = true;
		if (params->use_edge) {
			const unsigned complex_offset = helper->neighbours_map_cuda[center];
			float complex_val = helper->complex_cuda[complex_offset];
			const float rand_prob = curand_uniform(rand_state) - FLT_EPSILON;
			if (rand_prob < complex_val) edge_limit = false;
		}
	}

	short2 strong_points[NEIGHBOUR_NUM - 1];
	float3 strong_points_3d[NEIGHBOUR_NUM - 1];
	float3 strong_points_normals[NEIGHBOUR_NUM - 1];
	int strong_count = 0;
	float X[3];

	for (int i = 1; i < NEIGHBOUR_NUM; ++i) {
		short2 temp_point = GetNeighbourPoint(point, i, helper);
		if (temp_point.x == -1 || temp_point.y == -1) {
			continue;
		}
		strong_points[strong_count].x = temp_point.x;
		strong_points[strong_count].y = temp_point.y;
		// get 3d point in ref camera coord
		const int temp_center = temp_point.x + temp_point.y * width;
		float depth = ComputeDepthfromPlaneHypothesis(camera, plane_hypotheses[temp_center], make_int2(temp_point.x, temp_point.y));
		Get3DPoint(camera, strong_points[strong_count], depth, X);
		strong_points_3d[strong_count].x = X[0];
		strong_points_3d[strong_count].y = X[1];
		strong_points_3d[strong_count].z = X[2];
		float4 n4 = plane_hypotheses[temp_center];
		strong_points_normals[strong_count] = make_float3(n4.x, n4.y, n4.z);
		strong_count++;
	}
	if (strong_count < 3) {
		fit_plane_hypothese[center] = plane_hypotheses[center];
		return;
	}

	int iteration = 50;
	int use_a_index = -1, use_b_index = -1, use_c_index = -1;
	float min_cost = FLT_MAX;
	float4 best_plane;
	bool has_best_plane = false;
	uchar edge_test[NEIGHBOUR_NUM - 1][NEIGHBOUR_NUM - 1] = { 0 };
	while (iteration--)
	{
		int a_index = curand(rand_state) % strong_count;
		int b_index = curand(rand_state) % strong_count;
		int c_index = curand(rand_state) % strong_count;

		if (a_index == b_index || b_index == c_index || a_index == c_index) {
			continue;
		}

		const float3& AN = strong_points_normals[a_index];
		const float3& BN = strong_points_normals[b_index];
		const float3& CN = strong_points_normals[c_index];
		if (Vec3DotVec3(AN, BN) < 0.9f || Vec3DotVec3(AN, CN) < 0.9f || Vec3DotVec3(BN, CN) < 0.9f) {
			continue;
		}

		if (!PointinTriangle(strong_points[a_index], strong_points[b_index], strong_points[c_index], point)) {
			continue;
		}

		if (edge_limit) {
			if (edge_test[a_index][b_index] == 0) {
				edge_test[a_index][b_index] = edge_test[b_index][a_index] =
					(BresenhamLine(strong_points[a_index], strong_points[b_index], helper) ? 1 : 2);
			}
			if (edge_test[b_index][c_index] == 0) {
				edge_test[b_index][c_index] = edge_test[c_index][b_index] =
					(BresenhamLine(strong_points[b_index], strong_points[c_index], helper) ? 1 : 2);
			}
			if (edge_test[c_index][a_index] == 0) {
				edge_test[c_index][a_index] = edge_test[a_index][c_index] =
					(BresenhamLine(strong_points[c_index], strong_points[a_index], helper) ? 1 : 2);
			}
			if (edge_test[a_index][b_index] == 1 || edge_test[b_index][c_index] == 1 || edge_test[c_index][a_index] == 1) {
				continue;
			}
		}

		const float3& A = strong_points_3d[a_index];
		const float3& B = strong_points_3d[b_index];
		const float3& C = strong_points_3d[c_index];
		float3 A_C = make_float3(A.x - C.x, A.y - C.y, A.z - C.z);
		float3 B_C = make_float3(B.x - C.x, B.y - C.y, B.z - C.z);
		float4 cross_vec;
		cross_vec.x = A_C.y * B_C.z - B_C.y * A_C.z;
		cross_vec.y = -(A_C.x * B_C.z - B_C.x * A_C.z);
		cross_vec.z = A_C.x * B_C.y - B_C.x * A_C.y;
		if ((cross_vec.x == 0 && cross_vec.y == 0 && cross_vec.z == 0) || isnan(cross_vec.x) || isnan(cross_vec.y) || isnan(cross_vec.z)) {
			continue;
		}
		NormalizeVec3(&cross_vec);
		cross_vec.w = -(cross_vec.x * A.x + cross_vec.y * A.y + cross_vec.z * A.z);

		float temp_cost = 0.0f;
		for (int strong_index = 0; strong_index < strong_count; ++strong_index) {
			if (strong_index == a_index || strong_index == b_index || strong_index == c_index) {
				continue;
			}
			const float3& temp_point = strong_points_3d[strong_index];
			const short2& temp_pix = strong_points[strong_index];
			float factor_x = (temp_pix.x - camera.K[2]) / camera.K[0];
			float factor_y = (temp_pix.y - camera.K[5]) / camera.K[4];
			float fit_depth = -cross_vec.w / (cross_vec.x * factor_x + cross_vec.y * factor_y + cross_vec.z);
			float distance = fabs(fit_depth - temp_point.z);
			temp_cost += distance;
		}
		if (temp_cost < min_cost) {
			min_cost = temp_cost;
			best_plane = cross_vec;
			has_best_plane = true;
		}
	}

	if (has_best_plane) {
		float depth = ComputeDepthfromPlaneHypothesis(camera, plane_hypotheses[center], point);
		float4 view_direction = GetViewDirection(camera, point, depth);
		float dot_product = best_plane.x * view_direction.x + best_plane.y * view_direction.y + best_plane.z * view_direction.z;
		if (dot_product > 0) {
			best_plane.x = -best_plane.x;
			best_plane.y = -best_plane.y;
			best_plane.z = -best_plane.z;
			best_plane.w = -best_plane.w;
		}
		fit_plane_hypothese[center] = best_plane;

		if (helper->params->use_radius) {
			const short2& A = strong_points[use_a_index];
			const short2& B = strong_points[use_b_index];
			const short2& C = strong_points[use_c_index];
			float a = sqrtf((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
			float b = sqrtf((B.x - C.x) * (B.x - C.x) + (B.y - C.y) * (B.y - C.y));
			float c = sqrtf((C.x - A.x) * (C.x - A.x) + (C.y - A.y) * (C.y - A.y));
			float p = (a + b + c) / 2.0;
			float S = sqrtf(p * (p - a) * (p - b) * (p - c));
			int radius = (int)floor(sqrtf(S) / 2.0);
			float A_dis = sqrtf((A.x - point.x) * (A.x - point.x) + (A.y - point.y) * (A.y - point.y));
			float B_dis = sqrtf((B.x - point.x) * (B.x - point.x) + (B.y - point.y) * (B.y - point.y));
			float C_dis = sqrtf((C.x - point.x) * (C.x - point.x) + (C.y - point.y) * (C.y - point.y));
			float min_dis = MIN(MIN(A_dis, B_dis), C_dis);
			if (2.5 * min_dis < radius) radius = (int)min_dis;

			if (edge_limit) {
				if (params->use_edge) {
					float min_edge_dist = FLT_MAX;
					const unsigned offset = center * EDGE_NEIGH_NUM;
					const short2* edge_neigh = &(helper->edge_neigh_cuda[offset]);
					for (int dir_index = 0; dir_index < EDGE_NEIGH_NUM; ++dir_index) {
						short2 edge_pt = edge_neigh[dir_index];
						if (edge_pt.x == -1 || edge_pt.y == -1) continue;
						float dist = sqrtf((edge_pt.x - point.x) * (edge_pt.x - point.x) + (edge_pt.y - point.y) * (edge_pt.y - point.y));
						min_edge_dist = MIN(min_edge_dist, dist);
					}
					if (min_edge_dist < radius) radius = (int)min_edge_dist;
				}
				if (params->use_label) {
					float min_boundary_dist = FLT_MAX;
					const unsigned laboff = helper->neighbours_map_cuda[center] * LAB_BOUNDARY_NUM;
					const short2* lab_bound = &(helper->label_boundary_cuda[laboff]);
					for (int dir_index = 0; dir_index < LAB_BOUNDARY_NUM; ++dir_index) {
						short2 boundary_pt = lab_bound[dir_index];
						if (boundary_pt.x == -1 || boundary_pt.y == -1) continue;
						float dist = std::sqrt(std::pow(point.x - boundary_pt.x, 2) + std::pow(point.y - boundary_pt.y, 2));
						min_boundary_dist = MIN(min_boundary_dist, dist);
					}
					if (min_boundary_dist < radius) radius = (int)min_boundary_dist;
				}
			}

			while ((radius << 1) % 5 != 0) radius--;

			helper->radius_cuda[center] = radius < helper->params->strong_radius ? 0 : radius;
		}
	}
	else {
		fit_plane_hypothese[center] = make_float4(0, 0, 0, 0);
		if (helper->params->use_radius) {
			helper->radius_cuda[center] = helper->params->strong_radius;
		}
	}


}

void APD::RunPatchMatch() {
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	int BLOCK_W = 32;
	int BLOCK_H = (BLOCK_W / 2);

	dim3 grid_size_full;
	grid_size_full.x = (width + 16 - 1) / 16;
	grid_size_full.y = (height + 16 - 1) / 16;
	grid_size_full.z = 1;
	dim3 block_size_full;
	block_size_full.x = 16;
	block_size_full.y = 16;
	block_size_full.z = 1;

	dim3 grid_size_half;
	grid_size_half.x = (width + BLOCK_W - 1) / BLOCK_W;
	grid_size_half.y = ((height / 2) + BLOCK_H - 1) / BLOCK_H;
	grid_size_half.z = 1;
	dim3 block_size_half;
	block_size_half.x = BLOCK_W;
	block_size_half.y = BLOCK_H;
	block_size_half.z = 1;

	InitRandomStates << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	GenEdgeInform << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
#ifdef DEBUG_COMPLEX
	path complex_path = problem.result_folder / path("complex.jpg");
	cv::Mat complex_host(height, width, CV_32F);
	cudaMemcpy(complex_host.ptr<float>(0), complex_cuda, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	cv::Mat visual_complex_mat;
	complex_host.convertTo(visual_complex_mat, CV_8U, 255.0);
	cv::imwrite(complex_path.string(), visual_complex_mat);
#endif

	FindNearestStrongPoint << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	GenNeighbours << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	NeigbourUpdate << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	if (problem.show_medium_result) { // write neighbour for visualization
#ifdef DEBUG_NEIGHBOUR
		path neighbour_map_path = problem.result_folder / path("neighbour_map.bin");
		path neighbour_path = problem.result_folder / path("neighbour.bin");
		WriteBinMat(neighbour_map_path, neighbours_map_host);
		short2* neighbours_host = new short2[weak_count * NEIGHBOUR_NUM];
		cudaMemcpy(neighbours_host, neighbours_cuda, sizeof(short2) * weak_count * NEIGHBOUR_NUM, cudaMemcpyDeviceToHost);
		{
			ofstream out(neighbour_path, std::ios_base::binary);
			int neighbour_sample_num = NEIGHBOUR_NUM;
			out.write((char*)&weak_count, sizeof(int));
			out.write((char*)&neighbour_sample_num, sizeof(int));
			out.write((char*)neighbours_host, sizeof(short2) * weak_count * NEIGHBOUR_NUM);
			out.close();
		}
		delete[] neighbours_host;
#endif // DEBUG_NEIGHBOUR
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Generate neighbours done. Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	RandomInitialization << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	for (int i = 0; i < params_host.max_iterations; ++i) {
		BlackPixelUpdateStrong << <grid_size_half, block_size_half >> > (i, helper_cuda);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		RedPixelUpdateStrong << <grid_size_half, block_size_half >> > (i, helper_cuda);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		std::cout << "Iteration " << i << " strong done\n";
		RANSACToGetFitPlane << <grid_size_full, block_size_full >> > (helper_cuda);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		std::cout << "Compute normal done\n";
		BlackPixelUpdateWeak << <grid_size_half, block_size_half >> > (i, helper_cuda);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		RedPixelUpdateWeak << <grid_size_half, block_size_half >> > (i, helper_cuda);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		std::cout << "Iteration " << i << " -weak- done\n";
	}

	GetDepthandNormal << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	BlackPixelFilterStrong << <grid_size_half, block_size_half >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	RedPixelFilterStrong << <grid_size_half, block_size_half >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	DepthToWeak << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	LocalRefine << <grid_size_full, block_size_full >> > (helper_cuda);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
#ifdef DEBUG_COST_LINE
	{
		// export for test
		float* weak_ncc_cost_host = new float[width * height * 61];
		cudaMemcpy(weak_ncc_cost_host, weak_ncc_cost_cuda, width * height * sizeof(float) * 61, cudaMemcpyDeviceToHost);
		path weak_ncc_cost_path = problem.result_folder / path("weak_ncc_cost.bin");
		{
			ofstream out(weak_ncc_cost_path, std::ios_base::binary);
			int p_cost_count = 61;
			out.write((char*)&width, sizeof(int));
			out.write((char*)&height, sizeof(int));
			out.write((char*)&p_cost_count, sizeof(int));
			out.write((char*)weak_ncc_cost_host, sizeof(float) * width * height * p_cost_count);
			out.close();
		}
		delete[] weak_ncc_cost_host;
	}
#endif // DEBUG_COST_LINE
	cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(weak_info_host.ptr<uchar>(0), weak_info_cuda, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaMemcpy(selected_views_host.ptr<unsigned int>(0), selected_views_cuda, width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (problem.params.use_radius) {
		cudaMemcpy(radius_host.ptr<int>(0), radius_cuda, width * height * sizeof(int), cudaMemcpyDeviceToHost);
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}