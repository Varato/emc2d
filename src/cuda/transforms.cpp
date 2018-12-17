#include "transforms.hpp"
#include <iostream>

#include <stdio.h>

emc::Image<emcfloat> emc::Drift::forward_one(const Image<emcfloat>& model,  // the model to be drifted 
                                             intvec2d drift,                // the drift vector 
                                             int x_min, int y_min,          // the minimal drift 
                                             shape2d image_size)
{
    int x = -drift[0];
    int y = -drift[1];
    return model.crop(-x_min+x, -x_min+x+image_size[0], -y_min+y, -y_min+y+image_size[1]);
}

emc::ImageStack<emcfloat> emc::Drift::forward(const Image<emcfloat>& model, std::vector<intvec2d> drift_table)
{
    auto mm = model.getRows();
    auto nn = model.getCols();
    if (mm <= m_frame_shape[0] || nn <= m_frame_shape[1])
        throw std::invalid_argument("emc::Drift::forward: model size must be larger than image_size.");

    intvec2d x_range = emc::find_max_drift(mm, m_frame_shape[0]);
    intvec2d y_range = emc::find_max_drift(nn, m_frame_shape[1]);
    ImageStack<emcfloat> img_stk(m_frame_shape[0], m_frame_shape[1]);

    std::vector<intvec2d>::iterator iter;

    for(iter=drift_table.begin(); iter!=drift_table.end(); ++iter){
        if ((*iter)[0] < x_range[0] || (*iter)[0] > x_range[1])
            throw std::invalid_argument("x drift out of range.");
        if ((*iter)[1] < y_range[0] || (*iter)[1] > y_range[1])
            throw std::invalid_argument("y drift out of range.");

        img_stk.addImage(emc::Drift::forward_one(model, *iter, x_range[0], y_range[0], m_frame_shape));
    }
    return img_stk;

}

emc::Image<emcfloat> emc::Drift::backward(const ImageStack<emcfloat>& images, std::vector<intvec2d> drift_table)
{
    auto m = images.getRows();
    auto n = images.getCols();
    if (m_model_shape[0] < m || m_model_shape[1] < n)
        throw std::invalid_argument("emc::Drift::backward: model size must be larger than image_size.");
    if (images.size() != drift_table.size())
        throw std::invalid_argument("emc::Drift::backward: size of drift_table and image_stack not match.");

    intvec2d x_range = emc::find_max_drift(m_model_shape[0], m);
    intvec2d y_range = emc::find_max_drift(m_model_shape[1], n);

    std::vector<emcfloat> canvas(m_model_shape[0]*m_model_shape[1], 0);

    size_t k;
    int lx,hx,ly,hy;
    for (size_t p=0; p<images.size(); ++p){
        if (drift_table[p][0] < x_range[0] || drift_table[p][0] > x_range[1])
            throw std::invalid_argument("x drift out of range.");
        if (drift_table[p][1] < y_range[0] || drift_table[p][1] > y_range[1])
            throw std::invalid_argument("y drift out of range.");
        lx = -x_range[0] - drift_table[p][0];
        ly = -y_range[0] - drift_table[p][1];
        hx = lx + m;
        hy = ly + n;
        k = 0;  
        for (auto i=lx; i<hx; ++i)
            for (auto j=ly; j<hy; ++j)
                canvas[j + i*m_model_shape[0]] += images(p, k++);
    }
    emc::Image<emcfloat> ret(std::move(canvas), m_model_shape);
    return ret;
}



