include_directories("../include")

add_executable(testMpp testMpp.cpp)
target_link_libraries(testMpp matrix)
