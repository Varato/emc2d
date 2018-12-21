#include <iostream>
#include <stdio.h>
#include "image.hpp"

/************************
 * Image implementation *
 ************************/

template <typename T>
emc::Image<T>::Image(T *data, size_t shape[2])                       // for contigious array
{
    m_data.resize(shape[0]*shape[1]);
    std::memcpy(m_data.data(), data, shape[0]*shape[1]*sizeof(T));
    m_shape[0] = shape[0];
    m_shape[1] = shape[1];
}

template <typename T>
emc::Image<T>::Image(T *data, size_t shape[2], size_t strides[2])  // for non-contigious array
{
    m_data.resize(shape[0]*shape[1]);
    for (size_t i=0; i<shape[0]; ++i)
        for (size_t j=0; j<shape[1]; ++j)
            m_data[j+i*shape[0]] = *(data + i*strides[0] + j*strides[1]);
    m_shape[0] = shape[0];
    m_shape[1] = shape[1];
}

template <typename T>
emc::Image<T>::Image(std::vector<T> data, shape2d shape)
{
    // data is deep-copied when passing in.
    // So use std::move to avoid copying again in assignment,
    m_shape = shape;
    m_data = std::move(data);
}


template <typename T>
emc::Image<T> emc::Image<T>::crop(size_t x0, size_t x1, size_t y0, size_t y1) const
{
    shape2d shape = {x1-x0, y1-y0};
    if (x1 > m_shape[0] || y1 > m_shape[1])
        throw std::invalid_argument("crop coordinates out of boundary.");
    if (shape[0] <=0 || shape[1] <=0)
        throw std::invalid_argument( "Invalid crop coordinates. Negative or zero cropping shape detected." );

    std::vector<T> data(shape[0] * shape[1]);
    size_t k = 0;
    for (auto i=x0; i<x1; ++i)
        for (auto j=y0; j<y1; ++j){
            data[k] = (*this)(i,j);
            k += 1;
        }
    Image<T> ret(data, shape);
    return ret;
}

template <typename T>
std::ostream& operator<< (std::ostream& os, const emc::Image<T>& img){
    for (auto i=0; i<img.getRows(); ++i)
        for (auto j=0; j<img.getCols(); ++j){
            if (j<img.getCols()-1)
                os << img(i,j) << ", ";
            else
                os << img(i,j) << std::endl;
        }
    return os;
}


/*****************************
 * ImageStack implementation *
 *****************************/

template <typename T>
emc::ImageStack<T>::ImageStack(T* data, size_t shape[3])
{
    m_shape[0] = shape[0];  // frame index
    m_shape[1] = shape[1];  // image shape0
    m_shape[2] = shape[2];  // image shape1
    auto image_size = shape[1]*shape[2];

    for (auto n=0; n<m_shape[0]; ++n){
        std::vector<T> img(image_size);
        std::memcpy(img.data(), data+n*image_size, image_size*sizeof(T));
        m_data.push_back(img);
    }
}

template <typename T>
emc::ImageStack<T>::ImageStack(T* data, size_t shape[3], size_t strides[3])
{
    m_shape[0] = shape[0];  // frame index
    m_shape[1] = shape[1];  // image shape0
    m_shape[2] = shape[2];  // image shape1
    auto image_size = shape[1]*shape[2];

    for (auto n=0; n<m_shape[0]; ++n){
        std::vector<T> img(image_size);
        for (auto i=0; i<m_shape[1]; ++i)
            for (auto j=0; j<m_shape[2]; ++j){
                img[j+i*m_shape[1]] = *(data + n*strides[0] +i*strides[1] * j*strides[2]);
            }
        m_data.push_back(std::move(img));
    }
}

template <typename T>
emc::ImageStack<T>::ImageStack(std::vector<T> data, shape3d shape)
{
    m_shape = shape;
    m_data = std::move(data);
}

template <typename T>
emc::Image<T> emc::ImageStack<T>::getImage(size_t n) const
{
    shape2d shape = {m_shape[1], m_shape[2]};
    Image<T> img(m_data[n], shape);
    return img;
}

template <typename T>
void emc::ImageStack<T>::addImage(Image<T> img)
{
    if (img.getRows() != m_shape[1] || img.getCols() != m_shape[2])
        throw std::invalid_argument("img shape is not suitable for this stack.");
    m_data.push_back(img.getData());
    m_shape[0] += 1;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const emc::ImageStack<T>& stk){
    os << stk.size() << "-image stack" <<std::endl;
    for (auto i=0; i<stk.size(); ++i){
        os << "image " << i << std::endl;
        os << stk.getImage(i); 
    }
    return os;
}
