/*
 * Point.h
 *
 *  Created on: 2017年5月18日
 *      Author: litaoxiao
 */

#ifndef POINT_H_
#define POINT_H_

#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <vector>
#include <string>
#include <math.h>

class Point {
public:
	Point(std::string X);
	virtual ~Point();
	double hypothesis(std::vector<double> thetas);
	int getX(int index);
	std::vector<double> X; //坐标
	double y;
	double sigmod(int x);
};

#endif /* POINT_H_ */
