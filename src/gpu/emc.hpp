#include <vector>
#include "common.hpp"
#include "dataProtocol.hpp"

namespace emc {
    class DriftCorrector {
        private:
            emcfloat *modelOnDevice;
            Model modelOnHost;
            unsigned *framesOnDevice;
            Frames frames;
            FramesSparse framesSparse;
            shape2d modelDims;
            shape2d patternDims;
            shape2d halfDriftRange;
            unsigned iterNum = 0;
        public:
            DriftCorrector(Frames const &frames, shape2d halfDriftRange);
            DriftCorrector(FramesSparse const &framesSparse, shape2d halfDriftRange);

            // initialize the model by option
            void setModel(InitModelMethod flag);
            void setModel(Model model);


            void expand();
            void compress();
            void maximize();
            void iterate(unsigned maxIter);

    };
}
