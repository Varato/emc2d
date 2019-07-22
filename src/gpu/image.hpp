/* The classes in this header file provide representations of frame sets 
 * Image can be dense or spase
 */
# pragma once
#ifndef _FRAMES_H_
#define _FRAMES_H_

#include <cstddef> // for unsigned
#include <vector>
#include "common.hpp"

namespace emc {
    template <typename T> class Image; 
    template <typename T> class ImageStack;}

template <typename T>
std::ostream& operator<< (std::ostream& os, const emc::Image<T>&);

template <typename T>
std::ostream& operator<< (std::ostream& os, const emc::ImageStack<T>&);

namespace emc
{
    template <typename T>
    class Image
    {
        private:
            shape2d m_shape;
            std::vector<T> m_data;
        public:
            Image() = default;
            // The following two constructors deep-copy the data.
            // Thus the original T* data must manage its own memory.
            Image(T *data, unsigned shape[2]);                       // for contigious array
            Image(T *data, unsigned shape[2], unsigned strides[2]);    // for non-contigious array
            Image(std::vector<T> data, shape2d shape); 

            inline unsigned rows() const { return m_shape[0]; }
            inline unsigned cols() const { return m_shape[1]; }
            inline std::vector<T> const & getData() const { return m_data; }

            T operator ()(unsigned i, unsigned j) const { return m_data[j + i*m_shape[0]];}

            // [x0, x1) X [y0, y1) region will be cropped and return
            // Throw exceptions if invalide crop coordinates.
            Image<T> crop(unsigned x0, unsigned x1, unsigned y0, unsigned y1) const;

            Image operator +(const Image &);
            Image operator -(const Image &);
            Image operator *(const Image &);
            Image operator /(const Image &);
            Image operator +(emcfloat);
            Image operator -(emcfloat);
            Image operator *(emcfloat);
            Image operator /(emcfloat);

    };
}

namespace emc 
{
    template <typename T>
    class ImageStack
    {
        private:
            shape3d m_shape;
            std::vector<std::vector<T>> m_data;
        public:
            ImageStack() = default;
            ImageStack(unsigned rows, unsigned cols) {
                m_shape[0] = 0;
                m_shape[1] = rows;
                m_shape[2] = cols;
            }
            // The following two constructors deep-copy the data.
            // Thus the original T* data must manage its own memory.
            ImageStack(T* data, unsigned shape[3]);
            ImageStack(T* data, unsigned shape[3], unsigned strids[3]);
            ImageStack(std::vector<T> data, shape3d shape);

            inline unsigned size() const {return m_shape[0];}
            inline unsigned rows() const {return m_shape[1];}
            inline unsigned cols() const {return m_shape[2];}

            // element access
            //T operator() (unsigned n, unsigned i, unsigned j) const {return m_data[n][j+i*m_shape[1]];}
            T operator() (unsigned n, unsigned k) const {return m_data[n][k];}

            Image<T> getImage(unsigned) const;
            void addImage(Image<T>);
    };
}

#include "image.cpp"
#endif





