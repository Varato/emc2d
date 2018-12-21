#include <cstddef>
#include <iostream>
#include <vector>
#include "image.hpp"
#include "common.hpp"
#include "transforms.hpp"

int main()
{
    std::vector<float> data(25);
    for (auto i=0; i<25; ++i) data[i] = i;
    shape2d shape = {5, 5};

    emc::Image<float> model(data, shape);
    std::cout << model;

    emc::Drift trans(shape, shape2d{{3,3}});
    std::vector<intvec2d> drifts;
    drifts.push_back(intvec2d{{0, 0}});
    drifts.push_back(intvec2d{{0, 1}});
    drifts.push_back(intvec2d{{1, 1}});
    drifts.push_back(intvec2d{{-1, -1}});

    emc::ImageStack<float> stk = trans.forward(model, drifts);
    std::cout << stk;

    emc::Image<float> res = trans.backward(stk, drifts);

    std::cout << "backward result: \n" << res;


}
