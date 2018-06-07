#pragma once

namespace vj {
	class Point {
	public:
		unsigned int x;
		unsigned int y;

		Point() {}
		Point(unsigned int x, unsigned int y) {
			this->x = x;
			this->y = y;
		}
	};
}