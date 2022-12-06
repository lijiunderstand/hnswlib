cmake_minimum_required (VERSION 3.5)
project(hnsw_lib
    LANGUAGES CXX)

add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE .) 
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
    endif()

    add_executable(test_updates examples/updates_test.cpp)
    target_link_libraries(test_updates hnswlib)

    add_executable(searchKnnCloserFirst_test examples/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(main main.cpp sift_1b.cpp)
    target_link_libraries(main hnswlib)

    add_executable(sift_1M_L2 sift_1M_L2.cpp)
    target_link_libraries(sift_1M_L2 hnswlib OpenMP::OpenMP_CXX)

    add_executable(sift_1M_IP sift_1M_IP.cpp)
    target_link_libraries(sift_1M_IP hnswlib)
    add_executable(sift_1M_IP_normalize sift_1M_IP_normalize.cpp)
    target_link_libraries(sift_1M_IP_normalize hnswlib)
   
    add_executable(sift1M_demo sift1M_demo.cpp)
    target_link_libraries(sift1M_demo hnswlib)

    add_executable(sift1B_demo sift1B_demo.cpp)
    target_link_libraries(sift1B_demo hnswlib)
    
    add_executable(siftsmall siftsmall.cpp)
    target_link_libraries(siftsmall hnswlib)

endif()
