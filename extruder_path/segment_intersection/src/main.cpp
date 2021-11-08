#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define abs(x) (((x) > 0) ? (x) : (-(x)))

bool intersect(double xa, double ya, double xb, double yb, double xc, double yc, double xd, double yd, double eps) {
    double det = (xb - xa) * (yc - yd) - (xc - xd) * (yb - ya);
    if (abs(det) < eps)
        return false;
    double alpha = ((yc - yd) * (xc - xa) + (xd - xc) * (yc - ya)) / det, \
           beta = ((ya - yb) * (xc - xa) + (xb - xa) * (yc - ya)) / det;
    if (alpha > eps && alpha < 1 - eps && beta > eps && beta < 1 - eps)
        return true;
    else
        return false;
}

namespace py = pybind11;

bool query(py::array_t<double> segmentsLeftX, py::array_t<double> segmentsLeftY, py::array_t<double> segmentsRightX, py::array_t<double> segmentsRightY, double eps) {
    py::buffer_info segmentsLeftXBuf = segmentsLeftX.request();
    py::buffer_info segmentsLeftYBuf = segmentsLeftY.request();
    py::buffer_info segmentsRightXBuf = segmentsRightX.request();
    py::buffer_info segmentsRightYBuf = segmentsRightY.request();
    int numSegments = segmentsLeftXBuf.shape[0];
    assert (numSegments == segmentsLeftYBuf.shape[0] && numSegments == segmentsRightXBuf.shape[0] && numSegments == segmentsRightYBuf.shape[0]);
    double* segmentsLeftXPtr = (double*) segmentsLeftXBuf.ptr;
    double* segmentsLeftYPtr = (double*) segmentsLeftYBuf.ptr;
    double* segmentsRightXPtr = (double*) segmentsRightXBuf.ptr;
    double* segmentsRightYPtr = (double*) segmentsRightYBuf.ptr;
    for (int i = 0; i < numSegments; ++i)
        for (int j = i + 1; j < numSegments; ++j) {
            double xa = segmentsLeftXPtr[i], ya = segmentsLeftYPtr[i], \
                   xb = segmentsRightXPtr[i], yb = segmentsRightYPtr[i], \
                   xc = segmentsLeftXPtr[j], yc = segmentsLeftYPtr[j], \
                   xd = segmentsRightXPtr[j], yd = segmentsRightYPtr[j];
            if (intersect(xa, ya, xb, yb, xc, yc, xd, yd, eps))
                return true;
        }
    return false;
}

PYBIND11_MODULE(segment_intersection, m) {
    m.doc() = R"pbdoc(
        Check whether there is/are intersection(s) between any pair of input segments.
        Ignore intersections at endpoints or when segments are colinear.
    )pbdoc";

    m.def("query", &query, R"pbdoc(
        query
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}