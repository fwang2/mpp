include_directories("../include")
set(matrix_SRC 
    distance.cpp 
    eigen.cpp 
    fileIO.cpp 
    Matrix.cpp 
    matrixProcessing.cpp 
    preprocessing.cpp
    mpp.cpp)
add_library(matrix STATIC ${matrix_SRC})
