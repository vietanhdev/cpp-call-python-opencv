#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>

#define TEST_EXECUTION_TIME false

using namespace std;
namespace py = boost::python;
namespace np = boost::python::numpy;

void Init() {
    // Set your python location.
    // wchar_t str[] = L"/home/vietanhdev/miniconda3/envs/example_env";
    // Py_SetPythonHome(str);

    setenv("PYTHONPATH", ".", 1);

    Py_Initialize();
    np::initialize();
}


// Function to convert from cv::Mat to numpy array
np::ndarray ConvertMatToNDArray(const cv::Mat& mat) {
    py::tuple shape = py::make_tuple(mat.rows, mat.cols, mat.channels());
    py::tuple stride =
        py::make_tuple(mat.channels() * mat.cols * sizeof(uchar),
                       mat.channels() * sizeof(uchar), sizeof(uchar));
    np::dtype dt = np::dtype::get_builtin<uchar>();
    np::ndarray ndImg =
        np::from_data(mat.data, dt, shape, stride, py::object());

    return ndImg;
}


// Function to convert from numpy array to cv::Mat
cv::Mat ConvertNDArrayToMat(const np::ndarray& ndarr) {
    int length =
        ndarr.get_nd();  // get_nd() returns num of dimensions. this is used as
                         // a length, but we don't need to use in this case.
                         // because we know that image has 3 dimensions.
    const Py_intptr_t* shape =
        ndarr.get_shape();  // get_shape() returns Py_intptr_t* which we can get
                            // the size of n-th dimension of the ndarray.
    char* dtype_str = py::extract<char*>(py::str(ndarr.get_dtype()));

    // Variables for creating Mat object
    int rows = shape[0];
    int cols = shape[1];
    int channel = length == 3 ? shape[2] : 1;
    int depth;

    // Find corresponding datatype in C++
    if (!strcmp(dtype_str, "uint8")) {
        depth = CV_8U;
    } else if (!strcmp(dtype_str, "int8")) {
        depth = CV_8S;
    } else if (!strcmp(dtype_str, "uint16")) {
        depth = CV_16U;
    } else if (!strcmp(dtype_str, "int16")) {
        depth = CV_16S;
    } else if (!strcmp(dtype_str, "int32")) {
        depth = CV_32S;
    } else if (!strcmp(dtype_str, "float32")) {
        depth = CV_32F;
    } else if (!strcmp(dtype_str, "float64")) {
        depth = CV_64F;
    } else {
        std::cout << "Wrong dtype error" << std::endl;
        return cv::Mat();
    }

    int type = CV_MAKETYPE(
        depth, channel);  // Create specific datatype using channel information

    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, ndarr.get_data(), sizeof(uchar) * rows * cols * channel);

    return mat;
}

int main(int argc, char const* argv[]) {
    setlocale(LC_ALL, "");

    try {
        // Initialize boost python and numpy
        Init();

        // Import module
        py::object main_module = py::import("__main__");

        // Load the dictionary for the namespace
        py::object mn = main_module.attr("__dict__");

        // Import the module into the namespace
        py::exec("import image_processing", mn);

        // Create the locally-held object
        py::object image_processor =
            py::eval("image_processing.SimpleImageProccessor()", mn);
        py::object process_img = image_processor.attr("process_img");

        // Get image. Image from:
        // https://github.com/opencv/opencv/blob/master/samples/data/baboon.jpg
        cv::Mat img = cv::imread("baboon.jpg", cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "can't getting image" << std::endl;
            return -1;
        }

        cv::Mat clone_img = img.clone();  

        float total_time = 0;
        for (size_t i = 0; i < (TEST_EXECUTION_TIME ? 1000 : 1); i++) {
            const clock_t begin_time = clock();

            np::ndarray nd_img = ConvertMatToNDArray(clone_img);
            np::ndarray output_img = py::extract<np::ndarray>(process_img(nd_img));
            cv::Mat mat_img = ConvertNDArrayToMat(output_img);

            float instance_time = float(clock() - begin_time) / CLOCKS_PER_SEC;
            total_time += instance_time;
            cout << "Instance time: " << instance_time << endl;

            // Show image
            if (!TEST_EXECUTION_TIME) {
                cv::namedWindow("Original image", cv::WINDOW_NORMAL);
                cv::namedWindow("Output image", cv::WINDOW_NORMAL);
                cv::imshow("Original image", img);
                cv::imshow("Output image", mat_img);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
        }

        cout << "Avg. time: " << total_time / 1000 << endl;

    } catch (py::error_already_set&) {
        PyErr_Print();
    }

    return 0;
}
