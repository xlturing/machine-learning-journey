/*
 * Point.cpp
 *
 *  Created on: 2017年5月18日
 *      Author: litaoxiao
 */

#include "Point.h"

Point::Point(std::string line) {
	// TODO Auto-generated constructor stub
	std::vector<std::string> tmp;
	boost::split_regex(tmp, line, boost::regex("\t"));
	X.push_back(1); // add x0
	for (size_t n = 0; n < tmp.size() - 1; n++) {
		X.push_back(std::stod(tmp[n]));
	}
	y = std::stod(tmp[tmp.size() - 1]);
}

Point::~Point() {
	// TODO Auto-generated destructor stub
	Point::X.clear();
	std::vector<double>().swap(Point::X);
}

double Point::hypothesis(std::vector<double> thetas) {
	double rs = 0.0;
	for (size_t n = 0; n < thetas.size(); n++) {
		rs += (thetas[n] * X[n]);
	}

	return rs;
}

int Point::getX(int index) {
	return X[index];
}

