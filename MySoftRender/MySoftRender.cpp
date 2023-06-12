#include <map>
#include <time.h>
#include <conio.h>
#include <vector>
#include <Windows.h>
#include <fstream>
#include <string>
#include <iostream>
#include <graphics.h>

#define PI 3.1415926535
#define DEBUG 1
const int screenWidth = 800, screenHeight = 600;

template<typename T>
struct Vector3
{
	T x, y, z, w;

	Vector3<T>() :x(0), y(0), z(0), w(1) {}
	Vector3<T>(T vx, T vy, T vz) : x(vx), y(vy), z(vz), w(1) {}
	Vector3<T>(T vx, T vy, T vz, T vw) :x(vx), y(vy), z(vz), w(vw) {}
	
	Vector3<T> operator*(const T right)const
	{
		return Vector3<T>(x * right, y * right, z * right);
	}
	T operator*(const Vector3<T> right)const
	{
		return this->x * right.x + this->y * right.y + this->z * right.z;
	}
	T operator[](int index)const
	{
		return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w));
	}
	Vector3<T> operator+(const Vector3<T>& right)const
	{
		return Vector3<T>(this->x + right.x, this->y + right.y, this->z + right.z);
	}
	Vector3<T> operator-()const
	{
		return Vector3<T>(-x, -y, -z);
	}
	Vector3<T> operator-(const Vector3<T>& right)const
	{
		return Vector3<T>(this->x - right.x, this->y - right.y, this->z - right.z);
	}
	static Vector3<T> Dot(const Vector3<T>& left, const Vector3<T>& right)
	{
		return left.x * right.x + left.y * right.y + left.z * right.z;
	}
	static Vector3<T> Cross(const Vector3<T>& left, const Vector3<T>& right)
	{
		return Vector3<T>(
			left.y * right.z - left.z * right.y,
			left.z * right.x - left.x * right.z,
			left.x * right.y - left.y * right.x
		);
	}

	float getLength()
	{
		return sqrt(x * x + y * y + z * z);
	}
	void normalize()
	{
		float length = getLength();
		if(length==0)
			return;
		x /= length;
		y /= length;
		z /= length;
	}
	void standard()
	{
		if (!w)
			return;
		x /= w;
		y /= w;
		z /= w;
		w = 1;
	}
	void print()
	{
		std::cout << "[" << x << "," << y << "," << z << "]";
	}
};
typedef Vector3<float> Color;

template<typename T>
struct Vector2
{
	Vector2<T>() :x(0), y(0) {}
	Vector2<T>(T vx, T vy) : x(vx), y(vy) {}
	T x, y;
	Vector2<T> operator+(const Vector2& right)
	{
		return Vector2(x + right.x, y + right.y);
	}
	Vector2<T> operator*(float value)
	{
		return Vector2(x * value, y * value);
	}
	void print()
	{
		std::cout << "[" << x << "," << y << "]";
	}
};

struct Mesh
{
	std::vector<Vector3<float>> positionBuffer;
	std::vector<Vector2<float>> uvBuffer;
	std::vector<Vector3<float>> normalBuffer;
	std::vector<Vector3<float>> colorBuffer;
	std::vector<Vector3<int>> indexBuffer;

	void stringSplit(std::string s, char splitchar, std::vector<std::string>& vec)
	{
		if (vec.size() > 0)
			vec.clear();
		int length = s.length();
		int start = s[0] == splitchar ? 1 : 0;
		for (int i = 0; i < length; ++i)
		{
			if (s[i] == splitchar)
			{
				vec.push_back(s.substr(start, i - start));
				start = i + 1;
			}
			else if (i == length - 1)
				vec.push_back(s.substr(start, i + 1 - start));
		}
	}

	void readObjFile(std::string path)
	{
		std::ifstream in(path);
		std::string txt = "";

		if (in)
		{
			while (std::getline(in, txt))
			{
				if (txt[0] == 'v' && txt[1] == ' ')
				{
					std::vector<std::string> num;
					txt.erase(0, 2);
					stringSplit(txt, ' ', num);
					Vector3<float> pos;
					pos = Vector3<float>((float)atof(num[0].c_str()), (float)atof(num[1].c_str()), (float)atof(num[2].c_str()));
					this->positionBuffer.push_back(pos);
				}
				else if (txt[0] == 'v' && txt[1] == 'n')
				{
					std::vector<std::string> num;
					txt.erase(0, 3);
					stringSplit(txt, ' ', num);
					Vector3<float> n = Vector3<float>((float)atof(num[0].c_str()), (float)atof(num[1].c_str()), (float)atof(num[2].c_str()), 0.0);
					this->normalBuffer.push_back(n);
				}
				else if (txt[0] == 'v' && txt[1] == 't')
				{
					std::vector<std::string> num;
					txt.erase(0, 3);
					stringSplit(txt, ' ', num);
					this->uvBuffer.push_back(Vector2<float>((float)atof(num[0].c_str()), (float)atof(num[1].c_str())));
				}
				else if (txt[0] == 'f' && txt[1] == ' ')
				{
					std::vector<std::string> num;
					txt.erase(0, 2);
					stringSplit(txt, ' ', num);
					for (int i = 0; i < num.size(); ++i)
					{
						std::vector<std::string> threeIndex;
						stringSplit(num[i], '/', threeIndex);
						Vector3<int> indexes = { atoi(threeIndex[0].c_str()) - 1, atoi(threeIndex[1].c_str()) - 1, atoi(threeIndex[2].c_str()) - 1 };
						this->indexBuffer.push_back(indexes);
					}
				}
			}
		}
		else
			std::cout << "no file" << std::endl;
	}

	void print()
	{
		std::cout << "Mesh data:" << std::endl;
		for (int i = 0; i < positionBuffer.size(); ++i)
		{
			std::cout << "v ";
			positionBuffer[i].print();
			std::cout << std::endl;
		}
		std::cout << std::endl;
		for (int i = 0; i < uvBuffer.size(); ++i)
		{
			std::cout << "vt ";
			uvBuffer[i].print();
			std::cout << std::endl;
		}
		std::cout << std::endl;
		for (int i = 0; i < normalBuffer.size(); ++i)
		{
			std::cout << "vn ";
			normalBuffer[i].print();
			std::cout << std::endl;
		}
		std::cout << std::endl;
		for (int i = 0; i <= indexBuffer.size() - 3; i += 3)
		{
			std::cout << "f ";
			indexBuffer[i].print();
			std::cout << " ";
			indexBuffer[i + 1].print();
			std::cout << " ";
			indexBuffer[i + 2].print();
			std::cout << std::endl;
		}
		std::cout << "end" << std::endl;
	}
};

template<typename T>
struct Matrix4
{
	Matrix4() {  }
	Matrix4(const std::initializer_list<float>& list)
	{
		auto begin = list.begin();
		auto end = list.end();
		int i = 0, j = 0;
		while (begin != end)
		{
			data[i][j++] = *begin;
			if (j > 3)
			{
				++i;
				j = 0;
			}
			++begin;
		}
	}

	T data[4][4] = {};

	void Identity()
	{
		for (int i = 0; i < 4; ++i)
			data[i][i] = 1;
	}

	Matrix4<T> operator * (const Matrix4<T>& right) const
	{
		Matrix4 res;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				for (int k = 0; k < 4; ++k)
				{
					res.data[i][j] += this->data[i][k] * right.data[k][j];
				}
			}
		}
		return res;
	}

	Vector3<T> operator * (const Vector3<T>& v) const
	{
		float x = v.x * data[0][0] + v.y * data[0][1] + v.z * data[0][2] + v.w * data[0][3];
		float y = v.x * data[1][0] + v.y * data[1][1] + v.z * data[1][2] + v.w * data[1][3];
		float z = v.x * data[2][0] + v.y * data[2][1] + v.z * data[2][2] + v.w * data[2][3];
		float w = v.x * data[3][0] + v.y * data[3][1] + v.z * data[3][2] + v.w * data[3][3];
		Vector3<float> returnValue(x, y, z, w);
		return returnValue;
	}

	static Matrix4<T> get_model_matrix_translation(const Matrix4& model, const Vector3<float>& trs)
	{
		Matrix4 trsModel;
		trsModel.Identity();
		for (int i = 0; i < 3; i++)
			trsModel.data[i][3] = trs[i];
		return trsModel * model;
	}
	static Matrix4 get_model_matrix_scale(const Matrix4& model, const Vector3<float>& scale)
	{
		Matrix4 scaModel;
		scaModel.Identity();
		for (int i = 0; i < 3; i++)
			scaModel.data[i][i] = scale[i];
		return scaModel * model;
	}
	static Matrix4 get_model_matrix_rotateX(const Matrix4& model, float rotation_angle)
	{
		rotation_angle = rotation_angle / 180 * PI;
		Matrix4 rotateModel;
		rotateModel.Identity();
		rotateModel.data[1][1] = cos(rotation_angle);
		rotateModel.data[1][2] = -sin(rotation_angle);
		rotateModel.data[2][1] = -rotateModel.data[1][2];
		rotateModel.data[2][2] = rotateModel.data[1][1];
		return rotateModel * model;
	}
	static Matrix4 get_model_matrix_rotateY(const Matrix4& model, float rotation_angle)
	{
		rotation_angle = rotation_angle / 180 * PI;
		Matrix4 rotateModel;
		rotateModel.Identity();
		rotateModel.data[0][0] = cos(rotation_angle);
		rotateModel.data[0][2] = sin(rotation_angle);
		rotateModel.data[2][0] = -rotateModel.data[0][2];
		rotateModel.data[2][2] = rotateModel.data[0][0];
		return rotateModel * model;
	}
	static Matrix4 get_model_matrix_rotateZ(const Matrix4& model, float rotation_angle)
	{
		rotation_angle = rotation_angle / 180 * PI;
		Matrix4 rotateModel;
		rotateModel.Identity();
		rotateModel.data[0][0] = cos(rotation_angle);
		rotateModel.data[0][1] = -sin(rotation_angle);
		rotateModel.data[1][0] = -rotateModel.data[0][1];
		rotateModel.data[1][1] = rotateModel.data[0][0];
		return rotateModel * model;
	}
	static Matrix4 get_model_matrix_Rotate(const Matrix4& model,
		const Vector3<float>& axisT, float rotation_angle)
	{
		rotation_angle = rotation_angle / 180 * PI;

		Vector3<float> axis = axisT;
		axis.normalize();

		Matrix4 rotateModel;
		rotateModel.Identity();
		rotateModel = { (1 - cos(rotation_angle)) * (axis.x * axis.x) + cos(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.x * axis.y) - axis.z * sin(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.x * axis.z) + axis.y * sin(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.x * axis.y) + axis.z * sin(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.y * axis.y) + cos(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.y * axis.z) - axis.x * sin(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.x * axis.z) - axis.y * sin(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.y * axis.z) + axis.x * sin(rotation_angle),
			(1 - cos(rotation_angle)) * (axis.z * axis.z) + cos(rotation_angle) };
		return rotateModel * model;
	}

	static Matrix4 get_view_matrix(const Vector3<float>& eye_pos, const Vector3<float>& front,
		const Vector3<float>& up)
	{
		Matrix4 view;

		Matrix4 translate;
		translate = { 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
			-eye_pos[2], 0, 0, 0, 1 };

		Matrix4 rotate;
		Vector3<float> gxt = Vector3<float>::Cross(front, up);

		rotate = { gxt[0], gxt[1], gxt[2], 0,
			up[0], up[1], up[2], 0,
			-front[0], -front[1], -front[2], 0,
			0, 0, 0, 1 };

		return rotate * translate;
	}
	static Matrix4 get_projection_matrix(float eye_fov, float aspect_ratio,
		float zNear, float zFar)
	{
		Matrix4 projection;
		float f, n, l, r, b, t, fov;
		fov = eye_fov / 180 * PI;
		n = -zNear; 
		f = zFar;
		t = tan(fov / 2) * zNear;
		b = -t;
		r = t * aspect_ratio;
		l = -r;

		Matrix4 pertoorth;
		pertoorth = { n, 0, 0, 0,
			0, n, 0, 0,
			0, 0, n + f, -n * f,
			0, 0, 1, 0 };

		Matrix4 orth1;
		orth1 = { 1, 0, 0, -(r + l) / 2,
			0, 1, 0, -(t + b) / 2,
			0, 0, 1, -(n + f) / 2,
			0, 0, 0, 1 };

		Matrix4 orth2;
		orth2 = { 2 / (r - l), 0, 0, 0,
			0, 2 / (t - b), 0, 0,
			0, 0, 2 / (n - f), 0,
			0, 0, 0, 1 };
		projection = orth2 * orth1 * pertoorth;
		return projection;
	}

	static Matrix4 get_viewport_matrix(float width, float height)
	{
		Matrix4 viewport;
		viewport.Identity();
		viewport.data[0][0] = width / 2;
		viewport.data[1][1] = height / 2;
		viewport.data[0][3] = width / 2;
		viewport.data[1][3] = height / 2;
		return viewport;
	}
	void Print()
	{
		std::cout << "-----------------Matrix Begin--------------" << std::endl;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				std::cout << "[" << data[i][j] << "]   ";
			}
			std::cout << std::endl;
		}
		std::cout << "-----------------Matrix End----------------" << std::endl;
	}
};

struct VertexData
{
	VertexData() {}
	VertexData(Vector3<float> pos, Vector2<float> texCoor = Vector2<float>(0, 0), Vector3<float> nor = Vector3<float>(0, 0, 0, 1), Vector3<float> colr = Vector3<float>(255, 255, 255, 1)) :
		position(pos), uv(texCoor), normal(nor), color(colr) {}
	Vector3<float> position;
	Vector2<float> uv;
	Vector3<float> normal;
	Vector3<float> color;
};

struct Triangle
{
	VertexData vertex[3];
	Vector3<float> getNormal()
	{
		Vector3<float> v1 = vertex[1].position - vertex[0].position;
		Vector3<float> v2 = vertex[2].position - vertex[1].position;
		return Vector3<float>::Cross(v2, v1);
	}
};

struct Pixel
{
	Pixel(VertexData data, Vector2<int> posi) :
		verdata(data), pos(posi) {}
	VertexData verdata;
	Vector2<int> pos;
};

struct Myth
{
	static float clampe(float x, float mi, float ma)
	{
		if (x < mi)x = mi;
		if (x > mi)x = ma;
		return x;
	}

	template<typename T>
	static Vector3<T> centerOfGravity(const Vector3<T>& v1, const Vector3<T>& v2,
		const Vector3<T>& v3, const Vector2<int>& p)
	{
		if ((-(v1.x - v2.x) * (v3.y - v2.y) + (v1.y - v2.y) * (v3.x - v2.x)) == 0)
			return Vector3<T>(1, 0, 0);
		if (-(v2.x - v3.x) * (v1.y - v3.y) + (v2.y - v3.y) * (v1.x - v3.x) == 0)
			return Vector3<T>(1, 0, 0);
		float alpha = (-(p.x - v2.x) * (v3.y - v2.y) + (p.y - v2.y) * (v3.x - v2.x)) / (-(v1.x - v2.x) * (v3.y - v2.y) + (v1.y - v2.y) * (v3.x - v2.x));
		float beta = (-(p.x - v3.x) * (v1.y - v3.y) + (p.y - v3.y) * (v1.x - v3.x)) / (-(v2.x - v3.x) * (v1.y - v3.y) + (v2.y - v3.y) * (v1.x - v3.x));
		float gamma = 1 - alpha - beta;
		return Vector3<T>(alpha, beta, gamma);
	}

	template<typename T>
	static Vector2<float> get_leftTop(const Vector3<T>& v0, const Vector3<T>& v1, const Vector3<T>& v2)
	{
		return Vector2<float>(min(v0.x, min(v1.x, v2.x)), max(v0.y, max(v1.y, v2.y)));
	}

	template<typename T>
	static Vector2<float> get_rightBottom(const Vector3<T>& v0, const Vector3<T>& v1, const Vector3<T>& v2)
	{
		return Vector2<float>(max(v0.x, max(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)));
	}

	static bool isInTriangle(const Vector3<float>& pos, const Vector3<float>& pos0,
		const Vector3<float>& pos1, const Vector3<float>& pos2)
	{

		// 三次叉乘
		Vector3<float> res1 = Vector3<float>::Cross((pos - pos0), (pos1 - pos0));

		Vector3<float> res2 = Vector3<float>::Cross((pos - pos1), (pos2 - pos1));

		Vector3<float> res3 = Vector3<float>::Cross((pos - pos2), (pos0 - pos2));

		// 要求叉积同向
		if (res1.z * res2.z > 0 && res1.z * res3.z > 0 && res2.z * res3.z > 0)
			return true;
		else
			return false;
	}

	static bool isInNDC(const Vector3<float>& pos)
	{
		return ((abs(pos.x) > 1) + (abs(pos.y) > 1) + (abs(pos.z) > 1)) != 3;
	}
};
class Camera
{
public:
	Camera() {}
	Camera(Vector3<float> posT, Vector3<float> frontT,
		Vector3<float> upT) :pos(posT), front(frontT), up(upT) {}
	Vector3<float> pos;
	Vector3<float> front;
	Vector3<float> up;
};
class Render
{
public:
	int width, height;
	Render(int screenWidth, int screenHeight) :width(screenWidth), height(screenHeight) {}

	std::vector<Triangle> in_triangle;
	void assemblingElements(const Mesh& mesh)
	{
		for (int i = 0; i <= mesh.indexBuffer.size() - 3; i += 3)
		{
			Triangle trian;
			for (int j = 0; j < 3; ++j)
			{
				trian.vertex[j].position = mesh.positionBuffer[mesh.indexBuffer[i + j].x];
				trian.vertex[j].uv = mesh.uvBuffer[mesh.indexBuffer[i + j].y];
				trian.vertex[j].normal = mesh.normalBuffer[mesh.indexBuffer[i + j].z];
				trian.vertex[j].color = mesh.colorBuffer[mesh.indexBuffer[i + j].x];
			}
			in_triangle.push_back(trian);
		}
	}


	Matrix4<float> model, view, projection, viewport;
	void setMatrix(Matrix4<float> modelT, Matrix4<float> viewT,
		Matrix4<float> projectionT, Matrix4<float> viewportT)
	{
		model = modelT;
		view = viewT;
		projection = projectionT;
		viewport = viewportT;
	}

	std::vector<Triangle> out_triangle;
	Camera camera;
	bool isBackCulling;
	void backfaceCulling()
	{
#if DEBUG
		std::cout << std::endl << "-----------------backfaceCulling Begin-----------------" << std::endl;
#endif
		std::vector<Triangle>::iterator it = out_triangle.begin();
		while (it != out_triangle.end())
		{
			Vector3<float> v = camera.front;
			Vector3<float> n = (*it).getNormal();
			float value = v * n;
			if (value < 0)
				it = out_triangle.erase(it);
			else
				++it;
		}
#if DEBUG
		std::cout << std::endl << "out:" << out_triangle.size() << std::endl;
		for (int i = 0; i < out_triangle.size(); ++i)
		{
			std::cout << "Triangle " << i << " :" << std::endl;
			out_triangle[i].vertex[0].position.print();
			std::cout  << std::endl;
			out_triangle[i].vertex[1].position.print();
			std::cout << std::endl;
			out_triangle[i].vertex[2].position.print();
			std::cout << std::endl;
		}
		std::cout << std::endl << "-----------------backfaceCulling End-----------------" << std::endl;
#endif
	}

	bool isViewClipping;
	void viewFrustumClipping()
	{
#if DEBUG
		std::cout << std::endl << "-----------------viewFrustumClipping Begin-----------------" << std::endl;
#endif
		std::vector<Triangle>::iterator it = out_triangle.begin();
		while (it != out_triangle.end())
		{
			int index = 0;
			for (int j = 0; j < 3; ++j)
				index += Myth::isInNDC((*it).vertex[j].position);
			if (!index)
				it = out_triangle.erase(it);
			else
				++it;
		}
#if DEBUG
		std::cout << std::endl << "out:" << out_triangle.size() << std::endl;
		for (int i = 0; i < out_triangle.size(); ++i)
		{
			std::cout << "Triangle " << i << " :" << std::endl;
			out_triangle[i].vertex[0].position.print();
			std::cout << std::endl;
			out_triangle[i].vertex[1].position.print();
			std::cout << std::endl;
			out_triangle[i].vertex[2].position.print();
			std::cout << std::endl;
		}
		std::cout << std::endl << "-----------------viewFrustumClipping End-----------------" << std::endl;
#endif
	}

	void vertexShader()
	{
#if DEBUG
		std::cout << std::endl << "-----------------vertexShader Begin-----------------" << std::endl;
		std::cout << std::endl << "in:" << in_triangle.size() << std::endl;
		for (int i = 0; i < in_triangle.size(); ++i)
		{
			std::cout << "Triangle " << i << " :" << std::endl;
			in_triangle[i].vertex[0].position.print();
			std::cout << std::endl;
			in_triangle[i].vertex[1].position.print();
			std::cout << std::endl;
			in_triangle[i].vertex[2].position.print();
			std::cout << std::endl;
		}
#endif
		out_triangle.clear();
		for (int i = 0; i < in_triangle.size(); ++i)
		{
			Triangle trans;
			trans = in_triangle[i];
			for (int j = 0; j < 3; ++j)
			{
				trans.vertex[j].position = projection * view * model * in_triangle[i].vertex[j].position;
				trans.vertex[j].position.standard();
			}
				
			out_triangle.push_back(trans);
		}


#if DEBUG
		std::cout << std::endl << "out:" << out_triangle.size() << std::endl;
		for (int i = 0; i < out_triangle.size(); ++i)
		{
			std::cout << "Triangle " << i << " :" << std::endl;
			out_triangle[i].vertex[0].position.print();
			std::cout << std::endl;
			out_triangle[i].vertex[1].position.print();
			std::cout << std::endl;
			out_triangle[i].vertex[2].position.print();
			std::cout << std::endl;
		}
		std::cout << std::endl << "-----------------vertexShader endmodel-----------------" << std::endl;
#endif
		if(isViewClipping)
			viewFrustumClipping();
		if(isBackCulling)
			backfaceCulling();
		for (int i = 0; i < out_triangle.size(); i++)
			for (int j = 0; j < 3; ++j)
				out_triangle[i].vertex[j].position =
				viewport * out_triangle[i].vertex[j].position;
#if DEBUG
		std::cout << std::endl << "out:" << out_triangle.size() << std::endl;
		for (int i = 0; i < out_triangle.size(); ++i)
		{
			std::cout << "Triangle " << i << " :" << std::endl;
			out_triangle[i].vertex[0].position.print();
			std::cout << std::endl;
			out_triangle[i].vertex[1].position.print();
			std::cout << std::endl;
			out_triangle[i].vertex[2].position.print();
			std::cout << std::endl;
		}
		std::cout << std::endl << "-----------------vertexShader end-----------------" << std::endl;
#endif
	}

	std::vector<Pixel> pixels;
	class map_key_comp
	{
	public:
		bool operator()(const Vector2<int>& lhs, const Vector2<int>& rhs)const
		{
			return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
		}
	};
	std::map<Vector2<int>, float, map_key_comp> zBuffer;
	
	bool isTestZ;
	void rasterization()
	{
#if DEBUG
		std::cout << std::endl << "-----------------rasterization Begin-----------------" << std::endl;
		std::cout << out_triangle.size() << std::endl;
#endif
		pixels.clear();
		zBuffer.clear();
		
		for (int i = 0; i < out_triangle.size(); ++i)
		{
			//if (i==2 || i == 3 || i==4 || i==5 || i==8 || i==9  )
				//continue;
			Vector3<float> posArr[3] = {
				out_triangle[i].vertex[0].position ,
				out_triangle[i].vertex[1].position ,
				out_triangle[i].vertex[2].position };

			Vector2<float> leftTop = Myth::get_leftTop(posArr[0],posArr[1], posArr[2]);
			Vector2<float> rightBottom = Myth::get_rightBottom(posArr[0],posArr[1], posArr[2]);

			for (int x = leftTop.x; x <= rightBottom.x; ++x)
			{
				for (int y = leftTop.y; y >= rightBottom.y; --y)
				{
					const Vector2<int> pixPos(x, y);
					bool isInTriangle = Myth::isInTriangle(Vector3<float>(x, y, 0), posArr[0], posArr[1], posArr[2]);
					if (isInTriangle)
					{
						Vector3<float> abg = Myth::centerOfGravity(
							posArr[0], posArr[1], posArr[2], pixPos);
						float z = posArr[0].z * abg.x +
							posArr[1].z * abg.y + posArr[2].z * abg.z;

						if (zBuffer.count(pixPos))
						{
							if (z <= zBuffer[pixPos])
								;
							else
								zBuffer[pixPos] = z;
						}
						else
							zBuffer[pixPos] = z;
						
						Vector3<float> color = out_triangle[i].vertex[0].color * abg.x +
							out_triangle[i].vertex[1].color * abg.y + out_triangle[i].vertex[2].color * abg.z;
						Vector3<float> normal = out_triangle[i].vertex[0].normal * abg.x +
							out_triangle[i].vertex[1].normal * abg.y + out_triangle[i].vertex[2].normal * abg.z;
						Vector2<float> uv = out_triangle[i].vertex[0].uv * abg.x +
							out_triangle[i].vertex[1].uv * abg.y + out_triangle[i].vertex[2].uv * abg.z;

						VertexData verdata(Vector3<float>(0, 0, z), uv, normal, color);
						pixels.push_back(Pixel(verdata, Vector2<int>(x, y)));
					}
				}
			}
		}
#if DEBUG
		std::cout << std::endl << "-----------------rasterization End-----------------" << std::endl;
		std::cout << pixels.size() << std::endl;
		std::cout << zBuffer.size() << std::endl;
#endif
	}

	void testZ()
	{
#if DEBUG
		std::cout << std::endl << "-----------------testZ Begin-----------------" << std::endl;
#endif
		std::vector<Pixel>::iterator it = pixels.begin();
		while (it != pixels.end())
		{
			if ((*it).verdata.position.z < zBuffer[(*it).pos])
				it = pixels.erase(it);
			else
				++it;
		}
#if DEBUG
		std::cout << pixels.size() << std::endl;
		std::cout << std::endl << "-----------------testZ End-----------------" << std::endl;
#endif
	}

	void renderingPipeline()
	{
		vertexShader();
		rasterization();
		if(isTestZ)
			testZ();
	}
};

struct ScreenWindow
{
	const int width, height;
	Color clearBack;
	bool isClearBack = false;
	std::vector<Pixel> pixels;

	ScreenWindow(int screenWidth, int screenHeight) :width(screenWidth), height(screenHeight)
	{
		initgraph(screenWidth, screenHeight);

	}
	~ScreenWindow()
	{
		closegraph();
	}

	void update(const std::vector<Pixel>& input, bool clearBackColor = false, Color color = Color(1, 1, 1, 1))
	{
		this->pixels = input;
		if (clearBackColor)
		{
			isClearBack = true;
			clearBack = color;
			setbkcolor(RGB(clearBack.x, clearBack.y, clearBack.z));
		}

	}
	void show()
	{
		setorigin(0, 600);
		setaspectratio(1, -1);
		if (isClearBack)
			cleardevice();
		BeginBatchDraw();
		for (int i = 0; i < pixels.size(); ++i)
		{
			//putpixel(pixels[i].pos.x, pixels[i].pos.y, RGB(255, 0, 0));
			putpixel(pixels[i].pos.x, pixels[i].pos.y, RGB(pixels[i].verdata.color.x, pixels[i].verdata.color.y, pixels[i].verdata.color.z));
		}
		FlushBatchDraw();
	}

};


int main()
{
	initgraph(screenWidth, screenHeight, SHOWCONSOLE);
	std::string meshlLocation = "C:\\Users\\32156\\source\\repos\\SoftRender\\OBJ\\Cube.txt";
	Mesh mesh;
	mesh.readObjFile(meshlLocation);
	mesh.colorBuffer = {
		Color(255,255,255),
		Color(255,0,0),
		Color(0,255,0),
		Color(0,0,255),
		Color(255,255,255),
		Color(255,0,0),
		Color(0,255,0),
		Color(0,0,255),
	};

	Vector3<float> scale(1, 1, 1), position(0, 0, -2);
	Matrix4<float> model;
	model.Identity();
	model = Matrix4<float>::get_model_matrix_scale(model, scale);
	model = Matrix4<float>::get_model_matrix_translation(model, position);
	model.Print();

	float fov = 90, aspecet = 1, n = 1, f = 2;
	Matrix4<float> projection;
	projection.Identity();
	projection = Matrix4<float>::get_projection_matrix(fov, aspecet, n, f);
	projection.Print();

	Matrix4<float> viewport = Matrix4<float>::get_viewport_matrix(screenWidth, screenHeight);

	ScreenWindow window(screenWidth, screenHeight);

	Vector3<float> cameraPos(0.0f, 0.0f, 0.0f), cameraFront(0, 0, -1), cameraUp(0, 1, 0);
	cameraPos.normalize(); cameraFront.normalize(); cameraUp.normalize();
	Matrix4<float> view = Matrix4<float>::get_view_matrix(cameraPos, cameraFront, cameraUp);

	Render render(screenWidth, screenHeight);
	render.assemblingElements(mesh);

	render.camera = Camera(cameraPos, cameraFront, cameraUp);
	render.isViewClipping = true;
	render.isBackCulling = true;
	render.isTestZ = true;
	render.model = model;
	render.view = view;
	render.projection = projection;
	render.viewport = viewport;

#if DEBUG
	std::cout << "While out:" << std::endl;
	mesh.print();
	std::cout << std::endl;
	render.model.Print();
	std::cout << std::endl;
	render.view.Print();
	std::cout << std::endl;
	render.projection.Print();
	std::cout << std::endl;
	render.viewport.Print();
	std::cout << std::endl;
#endif
	clock_t begin = clock();
	while (true)
	{
		char key = _getch();
		switch (key)
		{
		case 'w':
			cameraPos.y += 0.1;
			break;
		case 's':
			cameraPos.y -= 0.1;
			break;
		case 'a':
			cameraPos.x += 0.1;
			break;
		case 'd':
			cameraPos.x -= 0.1;
			break;
		default:
			break;
		}
		clock_t now = clock();
		float deletime = static_cast<float>(now - begin) / CLOCKS_PER_SEC * 1000;
		begin = clock();
		int fps = 1.0 / deletime;
		std::cout << "FPS:" << fps << std::endl;

		cameraFront.normalize();
		render.view = Matrix4<float>::get_view_matrix(cameraPos, cameraFront, cameraUp);
		cameraPos.print();
		std::cout << std::endl;
		render.view.Print();
		std::cout << std::endl;

		render.renderingPipeline();

		window.update(render.pixels, true, Color(0, 0, 0));
		window.show();
	}
}