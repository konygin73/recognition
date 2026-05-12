from ultralytics import YOLO

model = YOLO("best.pt")
# format="onnx" — стандарт
# opset=12 — стабильная версия
# simplify=True — убирает сложные узлы
# dynamic=False — СТРОГО фиксируем размеры (OpenCV 4.5 не любит динамику)
# model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
model.export(
    format="onnx", 
    imgsz=640, 
    opset=12, 
    simplify=True, 
    dynamic=False
)
