#include <vector>
#include <array>
#include "image.hpp"
#include "common.hpp"

namespace emc
{
    inline intvec2d find_max_drift(size_t n, size_t window_size)
    {
        // Given a 1D window with size window_size and a 1D array with size n,
        // this function gives the range that the window can be drifted on the n-array.

        auto left_over = n - window_size;
        int x_max = (int) (left_over/2);
        int x_min = left_over%2 == 0? -x_max : -x_max-1;
        return intvec2d{{x_min, x_max}};
    }
    class Drift
    {
        private:
            shape2d m_model_shape;
            shape2d m_frame_shape;
        public:
            Drift(shape2d model_size, shape2d frame_size):
                m_model_shape(model_size), m_frame_shape(frame_size) {}
            static Image<emcfloat> forward_one(const Image<emcfloat>& model,  // the model to be drifted 
                                               std::array<int,2> drift,       // the drift vector 
                                               int x_min, int y_min,          // the minimal drift 
                                               shape2d frame_size);
            ImageStack<emcfloat> forward(const Image<emcfloat>& model, 
                                         std::vector<intvec2d> drift_table);
            Image<emcfloat> backward(const ImageStack<emcfloat>& images, 
                                     std::vector<intvec2d> drift_table);

    };

}
