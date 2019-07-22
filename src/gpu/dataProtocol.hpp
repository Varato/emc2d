#include <vector>
#include "common.hpp"


namespace emc{
    struct Frames {
        std::vector<ecount> data;
        unsigned numFrames;
        unsigned w, h;
        //TODO: constructor from numpy array
    };

    struct FramesSparse {
        // CSR format encoding a numFrames by w*h sparse matrix
        std::vector<ecount> data;
        std::vector<unsigned> iA; // indptr in scipy.sparse.csr_matrix
        std::vector<unsigned> jA; // index in scipy.sparse.csr_matrix
        unsigned numFrames;
        unsigned w, h;
        //TODO: constructor from scipy.sparse.csr_matrix
    };

    struct Model {
        std::vector<emcfloat> data;
        unsigned w, h;
        //TODO: constructor from numpy array
    };
}
