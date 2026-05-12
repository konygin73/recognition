// main.cpp
#include <QApplication>
#include <QLabel>
#include <QImage>
#include <QPixmap>
#include <QScreen>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// ============================================================================
// Конвертация Mat -> QImage (безопасная, с копированием данных)
// ============================================================================
QImage matToQImage(const Mat& mat) {
    if (mat.type() == CV_8UC3) {
        Mat rgb;
        cvtColor(mat, rgb, COLOR_BGR2RGB);
        return QImage((const uchar*)rgb.data, rgb.cols, rgb.rows, 
                     rgb.step, QImage::Format_RGB888).copy();
    }
    return QImage();
}

// ============================================================================
// Структура детекции
// ============================================================================
struct Detection {
    Rect box;
    float confidence;
    int classId;
    
    Detection(Rect r, float conf, int cls) 
        : box(r), confidence(conf), classId(cls) {}
};

// ============================================================================
// Letterbox: подготовка изображения под модель (как в Ultralytics)
// ============================================================================
Mat letterbox(const Mat& source, const Size& targetSize, 
              const Scalar& padColor = Scalar(114, 114, 114)) {
    Size srcSize = source.size();
    
    // Расчёт масштабирования с сохранением пропорций
    float ratio = min((float)targetSize.width / srcSize.width,
                      (float)targetSize.height / srcSize.height);
    
    int newW = cvRound(srcSize.width * ratio);
    int newH = cvRound(srcSize.height * ratio);
    
    // Ресайз
    Mat resized;
    resize(source, resized, Size(newW, newH), 0, 0, INTER_LINEAR);
    
    // Паддинг до targetSize
    Mat padded(targetSize.height, targetSize.width, source.type(), padColor);
    int dx = (targetSize.width - newW) / 2;
    int dy = (targetSize.height - newH) / 2;
    
    resized.copyTo(padded(Rect(dx, dy, newW, newH)));
    
    return padded;
}

#include <cmath> // <-- Добавьте в начало файла

// ============================================================================
// Декодирование выхода YOLOv11 (авто-определение: logits vs probabilities)
// ============================================================================
vector<Detection> decodeYOLOv11Output(const Mat& rawOutput, 
                                       const Size& imgSize,
                                       const Size& inputSize,
                                       float confThreshold) {
    vector<Detection> detections;
    if (rawOutput.dims != 3) return detections;

    // [1, 5, 8400] -> [5, 8400]
    Mat output = rawOutput.reshape(1, rawOutput.size[1]);
    if (output.rows != 5) {
        cerr << "⚠ Ошибка формата: ожидается 5 каналов, получено " << output.rows << endl;
        return detections;
    }

    int anchors = output.cols;
    const float* cx_ptr    = output.ptr<float>(0);
    const float* cy_ptr    = output.ptr<float>(1);
    const float* w_ptr     = output.ptr<float>(2);
    const float* h_ptr     = output.ptr<float>(3);
    const float* score_ptr = output.ptr<float>(4);

    // 🔍 Определяем формат выхода: логиты (>1.0) или вероятности (0..1)
    bool isLogits = false;
    for (int i = 0; i < min(anchors, 100); ++i) {
        if (score_ptr[i] > 1.0f) { isLogits = true; break; }
    }
    cout << "  [DEBUG] Format: " << (isLogits ? "Logits (apply sigmoid)" : "Probabilities [0,1]") << endl;

    // Коэффициенты letterbox
    float gain = min((float)inputSize.width / imgSize.width,
                     (float)inputSize.height / imgSize.height);
    float padX = (inputSize.width - imgSize.width * gain) / 2.0f;
    float padY = (inputSize.height - imgSize.height * gain) / 2.0f;

    int debugShown = 0;
    for (int i = 0; i < anchors; ++i) {
        float score = score_ptr[i];
        if (isLogits) {
            score = 1.0f / (1.0f + std::exp(-score)); // Только если это логиты
        }

        if (debugShown < 3 && score > confThreshold) {
            cout << "  [DEBUG] Anchor " << i << ": conf=" << score 
                 << " | cx=" << cx_ptr[i] << " cy=" << cy_ptr[i] 
                 << " w=" << w_ptr[i] << " h=" << h_ptr[i] << endl;
            debugShown++;
        }

        if (score < confThreshold) continue;

        // Обратное преобразование координат
        float cx_o = (cx_ptr[i] - padX) / gain;
        float cy_o = (cy_ptr[i] - padY) / gain;
        float w_o  = w_ptr[i] / gain;
        float h_o  = h_ptr[i] / gain;

        int left   = cvRound(max(0.0f, cx_o - w_o * 0.5f));
        int top    = cvRound(max(0.0f, cy_o - h_o * 0.5f));
        int width  = cvRound(min((float)imgSize.width - left, w_o));
        int height = cvRound(min((float)imgSize.height - top, h_o));

        left   = max(0, min(left, (int)imgSize.width - 1));
        top    = max(0, min(top, (int)imgSize.height - 1));
        width  = max(1, min(width, (int)imgSize.width - left));
        height = max(1, min(height, (int)imgSize.height - top));

        detections.emplace_back(Rect(left, top, width, height), score, 0);
    }
    return detections;
}

// ============================================================================
// Отрисовка детекций
// ============================================================================
void drawDetections(Mat& image, const vector<Detection>& detections,
                    const vector<string>& classNames = {}) {
    static const Scalar colors[] = {
        Scalar(0, 128, 255),    // Оранжевый для дронов
        Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
        Scalar(255, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255)
    };
    
    for (const auto& det : detections) {
        const Scalar& color = colors[det.classId % 7];
        
        // Bounding box
        rectangle(image, det.box, color, 3);
        
        // Подпись
        string label;
        if (!classNames.empty() && det.classId < (int)classNames.size()) {
            label = format("%s: %.2f", classNames[det.classId].c_str(), det.confidence);
        } else {
            label = format("Drone: %.2f", det.confidence);
        }
        
        int baseLine = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
        int top = max(det.box.y, labelSize.height + 8);
        
        // Фон под текстом
        rectangle(image, 
                 Point(det.box.x, top - labelSize.height - 5),
                 Point(det.box.x + labelSize.width + 5, top + baseLine - 3),
                 color, FILLED);
        
        // Текст
        putText(image, label, Point(det.box.x + 3, top - 3), 
               FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[]) {
#ifdef Q_OS_LINUX
    qputenv("QT_QPA_PLATFORM", "xcb");
#endif

    QApplication app(argc, argv);

    // === Пути к файлам ===
    const string imagePath = "dron.jpg";
    const string modelPath = "best.onnx";
    
    cout << "🚁 YOLOv11x Drone Detector" << endl;
    cout << "=========================" << endl;
    
    // === 1. Загрузка ===
    cout << "\n[1/5] Загрузка данных..." << endl;
    
    //Читает dron.jpg в память как cv::Mat формата CV_8UC3 (BGR, 3 канала)
    Mat originalFrame = imread(imagePath);
    if (originalFrame.empty()) {
        cerr << "❌ Ошибка: не найдено изображение '" << imagePath << "'" << endl;
        return -1;
    }
    cout << "✓ Изображение: " << originalFrame.cols << "x" << originalFrame.rows << endl;
    
    //Парсит Protobuf-файл, строит вычислительный граф (624 слоя), аллоцирует буферы под веса и активации
    Net net = readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "❌ Ошибка: не загружена модель '" << modelPath << "'" << endl;
        return -1;
    }
    cout << "✓ Модель YOLOv11x загружена" << endl;
    
    // Настройка инференса
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    //Указывает OpenCV использовать стандартные CPU-ядра (SSE/AVX инструкции)
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    // === 2. Предобработка (letterbox как в Ultralytics) ===
    cout << "\n[2/5] Предобработка..." << endl;
    
    const Size inputSize(640, 640);
    Mat inputFrame = letterbox(originalFrame, inputSize);
    
    // Нормализация и создание blob (swapRB=true для BGR→RGB)
    Mat blob = blobFromImage(inputFrame, 1.0/255.0, Size(), Scalar(), true, false);
    cout << "✓ Blob: [1, " << blob.size[1] << ", " 
         << blob.size[2] << ", " << blob.size[3] << "]" << endl;
    
    // === Диагностика (совместимая с OpenCV 4.6+) ===
    cout << "\n🔍 Диагностика:" << endl;
    cout << "OpenCV версия: " << CV_VERSION << endl;
    
    auto layerNames = net.getLayerNames();
    cout << "Слоёв в модели: " << layerNames.size() << endl;
    
    auto outNames = net.getUnconnectedOutLayersNames();
    cout << "Выходных слоёв: " << outNames.size() << endl;
    for (const auto& name : outNames) {
        cout << "  - " << name << endl;
    }
    
    // === 3. Инференс ===
    cout << "\n[3/5] Инференс..." << endl;
    
    net.setInput(blob);
    
    auto t1 = getTickCount();
    vector<Mat> outs;
    
    // 🔧 Безопасный forward с двумя вариантами для совместимости
    bool forwardOk = false;
    
    // Вариант 1: стандартный вызов с именами слоёв
    if (!outNames.empty()) {
        try {
            net.forward(outs, outNames);
            forwardOk = true;
            cout << "✓ forward(outs, names) успешен" << endl;
        } catch (const cv::Exception& e) {
            cout << "⚠ forward(outs, names) не сработал: " << e.what() << endl;
        }
    }
    
    // Вариант 2: упрощённый вызов (fallback)
    if (!forwardOk) {
        try {
            //Вход RGB → Conv2D → SiLU → C2f → Concat → Upsample → ... → Detect Head
            cout << "🔄 Пробуем упрощённый forward()..." << endl;
            Mat singleOut = net.forward();
            outs.push_back(singleOut);
            forwardOk = true;
            cout << "✓ forward() успешен" << endl;
        } catch (const cv::Exception& e) {
            cerr << "\n❌ Критическая ошибка инференса: " << e.what() << endl;
            cerr << "💡 Требуется OpenCV >= 4.10.0 для полной совместимости с YOLOv11" << endl;
            return -1;
        }
    }
    
    if (outs.empty()) {
        cerr << "❌ Нет выходных данных от модели" << endl;
        return -1;
    }
    
    auto t2 = getTickCount();
    double ms = (t2 - t1) / getTickFrequency() * 1000;
    cout << "✓ Инференс: " << ms << " мс" << endl;
    
    // === 4. Пост-обработка ===
    cout << "\n[4/5] Декодирование..." << endl;
    
    // Объединение выходов (если несколько)
    Mat output = outs[0];
    for (size_t i = 1; i < outs.size(); ++i) {
        output.push_back(outs[i]);
    }
    
    cout << "✓ Форма выхода: ";
    for (int d = 0; d < output.dims; ++d) cout << output.size[d] << " ";
    cout << endl;
    
    // Параметры детекции
    const float confThreshold = 0.35f;
    const float nmsThreshold = 0.45f;
    const int numClasses = 1;  // Только "drone"
    
    // Декодирование
    vector<Detection> detections = decodeYOLOv11Output(
        output, originalFrame.size(), inputSize, confThreshold);
    
    cout << "✓ Найдено: " << detections.size() << " детекций (до NMS)" << endl;
    
    // NMS
    vector<Rect> boxes;
    vector<float> scores;
    for (const auto& d : detections) {
        boxes.push_back(d.box);
        scores.push_back(d.confidence);
    }
    
    vector<int> indices;
    NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);
    
    vector<Detection> finalDetections;
    for (int idx : indices) {
        finalDetections.push_back(detections[idx]);
    }
    cout << "✓ После NMS: " << finalDetections.size() << " детекций" << endl;
    
    // === 5. Отрисовка и вывод ===
    cout << "\n[5/5] Отрисовка..." << endl;
    
    // Имена классов
    vector<string> classNames = {"drone"};
    
    drawDetections(originalFrame, finalDetections, classNames);
    
    // Статистика
    cout << "\n📊 Результаты:" << endl;
    if (finalDetections.empty()) {
        cout << "  (нет детекций выше порога)" << endl;
    }
    for (size_t i = 0; i < finalDetections.size(); ++i) {
        const auto& d = finalDetections[i];
        string clsName = (!classNames.empty() && d.classId < (int)classNames.size()) 
                        ? classNames[d.classId] 
                        : format("Class_%d", d.classId);
        cout << "  [" << i+1 << "] " << clsName 
             << " @ " << d.confidence * 100 << "% "
             << "[" << d.box.x << "," << d.box.y 
             << "," << d.box.width << "x" << d.box.height << "]" << endl;
    }
    
    // Отображение в окне
    QLabel label;
    label.setWindowTitle("🚁 Drone Detection - YOLOv11x");
    label.setPixmap(QPixmap::fromImage(matToQImage(originalFrame)));
    label.setScaledContents(true);
    
    // Адаптивный размер окна
    QSize screen = QGuiApplication::primaryScreen()->size();
    int w = min(originalFrame.cols, (int)(screen.width() * 0.9));
    int h = min(originalFrame.rows, (int)(screen.height() * 0.9));
    label.resize(w, h);
    
    label.show();
    
    cout << "\n✅ Готово! Закройте окно для выхода." << endl;
    
    return app.exec();
}
