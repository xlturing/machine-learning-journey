#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include <float.h>
#include "Point.h"

using namespace std;

void print(vector<double> v) {
	for (size_t i = 0; i < v.size(); i++)
		cout << v[i] << " ";
	cout << endl;
}

int main() {
	// parameters
	double ALPHA = 0.0005;
	double MAX_ITERATIONS = 1000;
	double MIN_ERR = 10;
	int N = 4;	// the number of theta

	// training data
	vector<Point> trainData;
	int m = 0; // the number of samples

	// read data from files
	ifstream inf;
	inf.open("train_data.txt");
	string line;
	while (getline(inf, line)) {
		trainData.push_back(Point(line));
		++m;
	}
	cout << "read train data num:" << trainData.size() << endl;

	// init parameters
	vector<double> thetas;
	for (int i = 0; i < N; i++) {
		thetas.push_back(1.0);
	}

	// batch gradient descent
	double err = DBL_MAX;
	int iter = 0;
	vector<double> tmp;
	while (err >= MIN_ERR && iter++ < MAX_ITERATIONS) {
		double sum = 0.0;
		for (int i = 0; i < m; i++) {
			double h = trainData[i].hypothesis(thetas);
			sum += ((h - trainData[i].y) * (h - trainData[i].y));
		}
		err = sum / 2 / m;

		// update theta
		tmp = thetas;
		for (size_t n = 0; n < thetas.size(); n++) {
			sum = 0.0;
			for (int i = 0; i < m; i++) {
				double h = trainData[i].hypothesis(thetas);
				sum += ((h - trainData[i].y) * trainData[i].X[n]);
			}
			tmp[n] -= (ALPHA * (sum / m));
		}
		thetas = tmp;
		cout << "after " << iter << " >>> error is " << err
				<< " iteration thetas is:" << endl;
		print(thetas);
	}

	cout << "error:" << err << " thetas:" << endl;
	print(thetas);
	return 0;
}
