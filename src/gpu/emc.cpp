#include "emc.hpp"
#include "dataProtocol.hpp"

namespace emc {
    DriftCorrector::DriftCorrector(Frames const &frames, shape2d halfDriftRange){
        this->frames = frames;
        this->halfDriftRange = halfDriftRange;
    }
}
