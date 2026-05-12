# recognition
распознавание дронов opencv

https://chat.qwen.ai

# содержит архитектуру + обученные параметры + метаданные в едином бинарном формате
https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/tree/main/weight
# YOLOv11x

yolo export model=best.pt format=onnx opset=12 simplify=true


/////////////////////////////////////////////////////////////////////
# 1. Экспортируйте лёгкую модель в Python
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"

# 2. В C++ коде убедитесь:
net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);
cv::setNumThreads(6);  // или 4, если ноутбук слабее

# 3. Соберите и запустите
cd build && make -j$(nproc) && ./DroneDetector_YOLOv11
/////////////////////////////////////////////////////////////////////

# Ubuntu/Debian: собрать из исходников
cd ~
git clone -b 4.10.0 https://github.com/opencv/opencv.git
git clone -b 4.10.0 https://github.com/opencv/opencv_contrib.git
cd opencv && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DOPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -DWITH_CUDA=ON \          # опционально: поддержка GPU
  -DWITH_CUDNN=ON \         # опционально
  -DOPENCV_DNN_CUDA=ON \    # опционально
  -DBUILD_opencv_python3=OFF

make -j$(nproc)
sudo make install
sudo ldconfig


sudo apt install build-essential cmake git
sudo apt install qt6-base-dev libqt6gui6 libqt6widgets6

sudo apt install libopencv-dev

#best.pt
pip install ultralytics onnx onnxsim

cd /home/alex/recognition/
sudo apt update && sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install -U ultralytics
pip show ultralytics

# yolo export model=best.pt format=onnx opset=12 simplify=true
python3 export_model.py
deactivate

sudo apt update
sudo apt install build-essential cmake libopencv-dev

mkdir build
cd build
cmake ..
make clean
cmake --build .

./DroneDetector_YOLOv11


# 1. Создаём и активируем окружение
mkdir ~/netron-tools && cd ~/netron-tools
python3 -m venv .
source bin/activate
# 2. Устанавливаем
pip install netron
# 3. Запускаем
netron ~/путь/к/best.onnx
