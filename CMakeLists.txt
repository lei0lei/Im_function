cmake_minimum_required(VERSION 3.16)
project(Im_function LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(TracyOptional)

# 若未设置 OpenCV_DIR，默认使用本机常见安装路径
if(NOT OpenCV_DIR AND EXISTS "C:/tools/opencv/build")
    set(OpenCV_DIR "C:/tools/opencv/build" CACHE PATH "OpenCV cmake config directory")
endif()

find_package(OpenCV REQUIRED)

add_executable(Im_function
    main.cpp
    ImageConcat.cpp
    ImageBlend.cpp
    ImageBlendAvx2.cpp
)

if(MSVC)
    set_source_files_properties(ImageBlendAvx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(ImageBlendAvx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

target_include_directories(Im_function PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(Im_function PRIVATE ${OpenCV_LIBS})

# Tracy 默认关闭；开启: cmake -DIM_FUNCTION_ENABLE_TRACY=ON ...
im_function_link_tracy(Im_function)

# Windows: 运行前把 OpenCV 的 bin 目录加入 PATH，或复制 DLL 到 exe 旁
if(WIN32)
    get_filename_component(_opencv_root "${OpenCV_DIR}" DIRECTORY)
    set(_opencv_bin "${_opencv_root}/x64/vc16/bin")
    if(EXISTS "${_opencv_bin}")
        message(STATUS "OpenCV bin: ${_opencv_bin}")
        message(STATUS "Add to PATH before running, or copy opencv_world*.dll next to the exe.")
    endif()
endif()
