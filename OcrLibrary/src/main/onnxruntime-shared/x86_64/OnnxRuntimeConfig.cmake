set(OnnxRuntime_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")
include_directories(${OnnxRuntime_INCLUDE_DIRS})
link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
set(OnnxRuntime_LIBS onnxruntime)
add_library(${OnnxRuntime_LIBS} SHARED IMPORTED)
set_target_properties(${OnnxRuntime_LIBS} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/lib/libonnxruntime.so)
