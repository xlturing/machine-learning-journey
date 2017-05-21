#include <vector>
#include <fstream>
#include <float.h>
#include <cmath>

#include "matplotlibcpp.h"
#include "Point.h"

using namespace std;

namespace plt = matplotlibcpp;

vector<Point> readData(string file) {
	ifstream inf;
	inf.open(file);
	vector<Point> trainData;
	string line;
	while (getline(inf, line)) {
		trainData.push_back(Point(line));
	}
	cout << "read train data num:" << trainData.size() << endl;

	return trainData;
}

void print(vector<double> v) {
	for (size_t i = 0; i < v.size(); i++)
		cout << v[i] << " ";
	cout << endl;
}

int main() {
	// parameters
	double ALPHA = 0.001;
	double MAX_ITERATIONS = 500;
	int N = 3;	// the number of theta

	// read data from file
	vector<Point> trainData = readData("train_data.txt");

	int m = trainData.size(); // the number of samples

	// init parameters
	vector<double> thetas;
	for (int i = 0; i < N; i++) {
		thetas.push_back(1.0);
	}

	// batch gradient descent
	double err = DBL_MAX;
	int iter = 0;
	vector<double> tmp;
	while (iter++ < MAX_ITERATIONS) {
		double sum = 0.0;
		for (int i = 0; i < m; i++) {
			double h = trainData[i].hypothesis(thetas);
			sum +=
					(log(h) * trainData[i].y + (1 - trainData[i].y) * log(1 - h));
		}
		err = sum / 2 / m;

		// update theta
		tmp = thetas;
		for (size_t n = 0; n < thetas.size(); n++) {
			sum = 0.0;
			for (int i = 0; i < m; i++) {
				double h = trainData[i].hypothesis(thetas);
				sum += ((trainData[i].y - h) * trainData[i].X[n]);
			}
			tmp[n] += (ALPHA * sum);
		}
		thetas = tmp;
		cout << "after " << iter << " >>> error is " << err
				<< " iteration thetas is:" << endl;
		print(thetas);
	}

	// plot data
	std::vector<double> x0, y0, x1, y1, x, y;
	for (size_t i = 0; i < trainData.size(); i++) {
		Point p = trainData[i];
		if (p.y == 0) {
			x0.push_back(p.X[1]);
			y0.push_back(p.X[2]);
		} else {
			x1.push_back(p.X[1]);
			y1.push_back(p.X[2]);
		}
	}
	for (double i = -3; i < 3; i += 0.1) {
		x.push_back(i);
		y.push_back((-thetas[0] - thetas[1] * i) / thetas[2]);
	}
	// Plot line from given x and y data. Color is selected automatically.
	plt::plot(x0, y0, "r+");
	plt::plot(x1, y1, "b*");
	plt::plot(x, y);
	// Set x-axis to interval [0,1000000]
	plt::xlim(-4, 4);
	plt::ylim(-20, 20);
	// Enable legend.
	plt::legend();
	// Save the image (file format is determined by the extension)
	plt::show();
}
