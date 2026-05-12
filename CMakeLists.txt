cmake_minimum_required(VERSION 3.10)
project(DroneDetector_YOLOv11 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
qt_standard_project_setup()

# OpenCV
find_package(OpenCV REQUIRED)

# Исполняемый файл
add_executable(${PROJECT_NAME} main.cpp)

# Линковка
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    ${OpenCV_LIBS}
)

# Копирование ресурсов в папку сборки
file(COPY ${CMAKE_SOURCE_DIR}/dron.jpg 
          ${CMAKE_SOURCE_DIR}/best.onnx
     DESTINATION ${CMAKE_BINARY_DIR})

# Вывод информации для отладки
message(STATUS "=== Сборка ===")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")