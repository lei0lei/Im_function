# 可选 Tracy：IM_FUNCTION_ENABLE_TRACY=OFF 时不链接任何 Tracy 依赖。
# 开启时仅使用 IM_FUNCTION_TRACY_PATH 或 third_party/tracy；两者皆无则自动关闭。
option(IM_FUNCTION_ENABLE_TRACY "Enable Tracy profiler (optional)" OFF)

set(IM_FUNCTION_TRACY_PATH "" CACHE PATH "Tracy source root (default: third_party/tracy)")

function(im_function_link_tracy target_name)
    if(NOT IM_FUNCTION_ENABLE_TRACY)
        message(STATUS "Tracy profiler: disabled")
        return()
    endif()

    set(_tracy_src "")
    if(IM_FUNCTION_TRACY_PATH AND EXISTS "${IM_FUNCTION_TRACY_PATH}/CMakeLists.txt")
        set(_tracy_src "${IM_FUNCTION_TRACY_PATH}")
    elseif(IM_FUNCTION_TRACY_PATH)
        message(WARNING "IM_FUNCTION_TRACY_PATH is invalid: ${IM_FUNCTION_TRACY_PATH}")
    endif()

    if(NOT _tracy_src AND EXISTS "${PROJECT_SOURCE_DIR}/third_party/tracy/CMakeLists.txt")
        set(_tracy_src "${PROJECT_SOURCE_DIR}/third_party/tracy")
    endif()

    if(NOT _tracy_src)
        message(STATUS "Tracy profiler: requested but no local source found, disabled")
        return()
    endif()

    message(STATUS "Tracy profiler: enabled (${_tracy_src})")
    # Client 须与 Tracy GUI 同版本（协议号一致）。GitHub 预编译 GUI 目前为 v0.13.1。
    # third_party/tracy 请保持在 v0.13.1；若用更新源码，需自行编译同版本 profiler。

    set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)

    add_subdirectory("${_tracy_src}" "${CMAKE_BINARY_DIR}/tracy" EXCLUDE_FROM_ALL)

    if(NOT TARGET TracyClient)
        message(WARNING "TracyClient target not found, Tracy disabled")
        return()
    endif()

    target_compile_definitions(${target_name} PRIVATE IM_FUNCTION_USE_TRACY TRACY_ENABLE)
    target_link_libraries(${target_name} PRIVATE TracyClient)
endfunction()
