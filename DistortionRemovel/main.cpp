#include <iostream>
#include "Eigen\Dense"

using namespace std;
using namespace Eigen;
int RemoveDistortion(double &distX, double &distY, double srcX, double srcY, double *Kx, double *Ky);

int main()
{
	// 生成模拟数据，x和y为无畸变坐标，f_xy为x方向的畸变量，g_xy为y方向的畸变量，畸变模型为双线性（x和y坐标需要是归一化坐标）
	double x = 0.5;
	double y = 0.5;
	double kx[4] = { 0.025, 0.087, 0.089, -0.012 };
	double ky[4] = { -0.116, -0.103, -0.122, -0.103 };
	double f_xy = (1 - x - y + x*y)*kx[0] + (x - x*y)*kx[1] + (y - x*y)*kx[2] + x*y*kx[3];
	double g_xy = (1 - x - y + x*y)*ky[0] + (x - x*y)*ky[1] + (y - x*y)*ky[2] + x*y*ky[3];
	double distor_x = x + f_xy;
	double distor_y = y + g_xy;

	// 高斯牛顿法去畸变, 理论上(dDist_x, dDist_y）与（x,y)是相等的
	double dDist_x, dDist_y;
	cout << "有畸变坐标：" << distor_x << " " << distor_y << endl;
	RemoveDistortion(dDist_x, dDist_y, distor_x, distor_y, kx, ky);
	cout << "去畸变坐标：" << dDist_x << " " << dDist_y << endl;
	cout << "理论无畸变坐标: " << x << " " << y << endl;

	system("pause");
	return 0;
}

int RemoveDistortion(double &distX, double &distY, double srcX, double srcY, double *Kx, double *Ky)
{
	// 去畸变坐标的初始值设置为有畸变的坐标
	distX = srcX;
	distY = srcY;

	int iMaxItetNum = 5;
	int iIterCnt = 0;
	Eigen::MatrixXd delta;
	delta.setOnes(2, 1);

	// 迭代去畸变
	do{
		// 雅可比矩阵
		Eigen::Matrix2d Jacobi;
		Jacobi(0, 0) = 1 + (-1 + distY)*Kx[0]
			+ (1 - distY)*Kx[1]
			+ (-distY)*Kx[2]
			+ distY*Kx[3];
		Jacobi(0, 1) = (-1 + distX)*Kx[0]
			+ (-distX)*Kx[1]
			+ (1 - distX)*Kx[2]
			+ distX*Kx[3];
		Jacobi(1, 0) = (-1+distY)*Ky[0]
			+ (1 - distY)*Ky[1]
			+ (-distY)*Ky[2]
			+ distY*Ky[3];
		Jacobi(1, 1) = 1 + (-1 + distX)*Ky[0]
			+ (-distX)*Ky[1]
			+ (1 - distX)*Ky[2]
			+ distX*Ky[3];

		// 残差矩阵
		Eigen::MatrixXd R;
		R.setZero(2, 1);
		R(0, 0) = distX + (1 - distX - distY + distX*distY)*Kx[0]
			+ (distX-distX*distY)*Kx[1]
			+ (distY-distX*distY)*Kx[2]
			+ distX*distY*Kx[3]
			- srcX;
		R(1, 0) = distY + (1 - distX - distY + distX*distY)*Ky[0]
			+ (distX - distX*distY)*Ky[1]
			+ (distY - distX*distY)*Ky[2]
			+ distX*distY*Ky[3]
			- srcY;

		// 求解增量
		delta = (Jacobi.transpose() * Jacobi).inverse();
		delta *= Jacobi.transpose() * R *(-1);

		// 下一次迭代
		iIterCnt++;
		distX = distX + delta(0, 0);
		distY = distY + delta(1, 0);
	} while (iIterCnt < iMaxItetNum && (abs(delta(0, 0)) > 1e-3 || abs(delta(1, 0)) > 1e-3));

	return 0;
}