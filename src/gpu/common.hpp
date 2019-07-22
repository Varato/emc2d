#pragma once
#ifndef _PRECISION_H_
#define _PRECISION_H_

#include <array>

namespace emc {
    typedef float emcfloat;
    typedef unsigned short ecount;
    typedef std::array<size_t, 2> shape2d;
    typedef std::array<size_t, 3> shape3d;
    typedef std::array<int, 2> intvec2d;

    enum InitModelMethod {
        RANDOM,
        FRAMES
    };
}


#endif
