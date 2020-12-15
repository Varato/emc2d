find_package(OpenMP)
if (OPENMP_FOUND)
    message("-- Found OpenMP library ${OpenMP_CXX_LIBRARIES}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(WIN32)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
endif(WIN32)

if (UNIX)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
endif(UNIX)

pybind11_add_module(emc_kernel src/emc_kernel.cpp)