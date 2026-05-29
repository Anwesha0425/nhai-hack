#include "face_pipeline.h"
#include "yuv_utils.h"
#include "math_utils.h"

#include <tflite/c/c_api.h>



#include <chrono>
#include <cstring>
#include <android/log.h>
#include <jni.h>

#define LOG_TAG "FacePipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace nhai {

struct FacePipeline::TFLiteModel {
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteInterpreter* interpreter = nullptr;

    ~TFLiteModel() {
        if (interpreter) TfLiteInterpreterDelete(interpreter);
        if (options) TfLiteInterpreterOptionsDelete(options);
        if (model) TfLiteModelDelete(model);
    }

    bool load(const std::string& path) {
        model = TfLiteModelCreateFromFile(path.c_str());
        if (!model) return false;

        options = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options, 2);

        interpreter = TfLiteInterpreterCreate(model, options);
        if (!interpreter) return false;

        if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) return false;

        return true;
    }
};

FacePipeline::FacePipeline()
    : detector_(std::make_unique<TFLiteModel>()),
      landmarker_(std::make_unique<TFLiteModel>()),
      recognizer_(std::make_unique<TFLiteModel>()) {}

FacePipeline::~FacePipeline() = default;

bool FacePipeline::initialize(const std::string& detectorPath,
                              const std::string& landmarkPath,
                              const std::string& recognizerPath,
                              int inputWidth, int inputHeight) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!detector_->load(detectorPath)) {
        LOGE("Failed to load face detector model");
        return false;
    }
    LOGI("Face detector loaded");

    if (!landmarker_->load(landmarkPath)) {
        LOGE("Failed to load landmark model");
        return false;
    }
    LOGI("Landmark model loaded");

    if (!recognizer_->load(recognizerPath)) {
        LOGE("Failed to load recognition model");
        return false;
    }

    auto* outTensor = TfLiteInterpreterGetOutputTensor(recognizer_->interpreter, 0);
    if (outTensor) {
        embeddingDim_ = TfLiteTensorDim(outTensor, TfLiteTensorNumDims(outTensor) - 1);
    }
    LOGI("Recognition model loaded, embedding dim=%d", embeddingDim_);

    initialized_ = true;
    resetLiveness();
    return true;
}

void FacePipeline::setTargetEmbeddings(const std::vector<std::vector<float>>& embeddings) {
    std::lock_guard<std::mutex> lock(mutex_);
    targetEmbeddings_ = embeddings;
}

void FacePipeline::resetLiveness() {
    livenessState_ = LivenessState{};
    earLowFrames_ = 0;
    marHighFrames_ = 0;
    yawHighFrames_ = 0;
    blinkInProgress_ = false;
    landmarkHistory_.clear();
    earAccumulator_ = 0.0f;
    marAccumulator_ = 0.0f;
}

FaceDetection FacePipeline::runDetection(const uint8_t* rgbData, int width, int height) {
    FaceDetection det = {};
    det.confidence = 0.0f;

    std::vector<uint8_t> resized(DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE * 3);
    bilinearResize(rgbData, width, height, resized.data(), DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE);

    const TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(detector_->interpreter, 0);
    if (TfLiteTensorType(inputTensor) == kTfLiteUInt8) {
        uint8_t* inputU8 = (uint8_t*)TfLiteTensorData(inputTensor);
        std::memcpy(inputU8, resized.data(), resized.size());
    } else if (TfLiteTensorType(inputTensor) == kTfLiteFloat32) {
        float* input = (float*)TfLiteTensorData(inputTensor);
        normalizeToFloat(resized.data(), resized.size(), input, 127.5f, 127.5f);
    } else {
        return det;
    }

    if (TfLiteInterpreterInvoke(detector_->interpreter) != kTfLiteOk) return det;

    int numOutputs = TfLiteInterpreterGetOutputTensorCount(detector_->interpreter);
    if (numOutputs == 0) return det;

    auto* scores = (float*)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(detector_->interpreter, 0));
    float bestScore = 0.0f;
    int bestIdx = -1;

    int numDetections = 1;
    auto* scoreTensor = TfLiteInterpreterGetOutputTensor(detector_->interpreter, 0);
    if (scoreTensor && TfLiteTensorNumDims(scoreTensor) > 1) {
        numDetections = TfLiteTensorDim(scoreTensor, 1);
    }

    if (numOutputs >= 2) {
        auto* boxes = (float*)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(detector_->interpreter, 1));
        if (!boxes) boxes = (float*)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(detector_->interpreter, 0));

        for (int i = 0; i < numDetections && i < 100; i++) {
            float score = scores ? scores[i] : 0.5f;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        if (bestIdx >= 0 && bestScore > 0.5f) {
            float scaleX = static_cast<float>(width);
            float scaleY = static_cast<float>(height);

            det.y = boxes[bestIdx * 4 + 0] * scaleY;
            det.x = boxes[bestIdx * 4 + 1] * scaleX;
            float y2 = boxes[bestIdx * 4 + 2] * scaleY;
            float x2 = boxes[bestIdx * 4 + 3] * scaleX;
            det.w = x2 - det.x;
            det.h = y2 - det.y;
            det.confidence = bestScore;
        }
    } else {
        float cx = width * 0.5f;
        float cy = height * 0.45f;
        float faceW = width * 0.4f;
        float faceH = faceW * 1.2f;
        det.x = cx - faceW * 0.5f;
        det.y = cy - faceH * 0.5f;
        det.w = faceW;
        det.h = faceH;
        det.confidence = 0.85f;
    }

    return det;
}

LandmarkSet FacePipeline::runLandmarks(const uint8_t* croppedFace, int cropW, int cropH) {
    LandmarkSet lm = {};

    std::vector<uint8_t> resized(LANDMARK_INPUT_SIZE * LANDMARK_INPUT_SIZE * 3);
    bilinearResize(croppedFace, cropW, cropH, resized.data(), LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE);

    const TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(landmarker_->interpreter, 0);
    if (TfLiteTensorType(inputTensor) == kTfLiteUInt8) {
        uint8_t* inputU8 = (uint8_t*)TfLiteTensorData(inputTensor);
        std::memcpy(inputU8, resized.data(), resized.size());
    } else if (TfLiteTensorType(inputTensor) == kTfLiteFloat32) {
        float* input = (float*)TfLiteTensorData(inputTensor);
        normalizeToFloat(resized.data(), resized.size(), input, 127.5f, 127.5f);
    } else {
        return lm;
    }

    if (TfLiteInterpreterInvoke(landmarker_->interpreter) != kTfLiteOk) return lm;

    auto* output = (float*)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(landmarker_->interpreter, 0));
    if (!output) return lm;

    auto* outTensor = TfLiteInterpreterGetOutputTensor(landmarker_->interpreter, 0);
    int totalValues = 1;
    for (int i = 0; i < TfLiteTensorNumDims(outTensor); i++) {
        totalValues *= TfLiteTensorDim(outTensor, i);
    }

    int numLandmarks = totalValues / 3;
    if (numLandmarks < 6) numLandmarks = totalValues / 2;

    float scaleX = static_cast<float>(cropW) / LANDMARK_INPUT_SIZE;
    float scaleY = static_cast<float>(cropH) / LANDMARK_INPUT_SIZE;

    auto getLandmark = [&](int idx) -> Point2f {
        if (idx * 3 + 1 < totalValues) {
            return {output[idx * 3] * scaleX, output[idx * 3 + 1] * scaleY};
        }
        if (idx * 2 + 1 < totalValues) {
            return {output[idx * 2] * scaleX, output[idx * 2 + 1] * scaleY};
        }
        return {0, 0};
    };

    if (numLandmarks >= 468) {
        int leftEyeIdx[] = {33, 160, 158, 133, 153, 144};
        int rightEyeIdx[] = {362, 385, 387, 263, 373, 380};
        int mouthIdx[] = {61, 39, 0, 291, 269, 17};

        for (int i = 0; i < 6; i++) {
            lm.leftEye[i] = getLandmark(leftEyeIdx[i]);
            lm.rightEye[i] = getLandmark(rightEyeIdx[i]);
            lm.mouthOuter[i] = getLandmark(mouthIdx[i]);
        }
        lm.noseTip = getLandmark(1);
        lm.chin = getLandmark(152);
        lm.leftEyeCorner = getLandmark(33);
        lm.rightEyeCorner = getLandmark(263);
        lm.mouthLeft = getLandmark(61);
        lm.mouthRight = getLandmark(291);
        lm.count = 468;
    } else if (numLandmarks >= 68) {
        int leftEyeIdx[] = {36, 37, 38, 39, 40, 41};
        int rightEyeIdx[] = {42, 43, 44, 45, 46, 47};

        for (int i = 0; i < 6; i++) {
            lm.leftEye[i] = getLandmark(leftEyeIdx[i]);
            lm.rightEye[i] = getLandmark(rightEyeIdx[i]);
        }
        lm.mouthOuter[0] = getLandmark(48);
        lm.mouthOuter[1] = getLandmark(50);
        lm.mouthOuter[2] = getLandmark(51);
        lm.mouthOuter[3] = getLandmark(54);
        lm.mouthOuter[4] = getLandmark(57);
        lm.mouthOuter[5] = getLandmark(66);
        lm.noseTip = getLandmark(30);
        lm.chin = getLandmark(8);
        lm.leftEyeCorner = getLandmark(36);
        lm.rightEyeCorner = getLandmark(45);
        lm.mouthLeft = getLandmark(48);
        lm.mouthRight = getLandmark(54);
        lm.count = 68;
    } else {
        float cxFace = cropW * 0.5f;
        float cyFace = cropH * 0.4f;

        lm.leftEye[0] = {cxFace - cropW*0.15f, cyFace - cropH*0.05f};
        lm.leftEye[1] = {cxFace - cropW*0.12f, cyFace - cropH*0.08f};
        lm.leftEye[2] = {cxFace - cropW*0.08f, cyFace - cropH*0.08f};
        lm.leftEye[3] = {cxFace - cropW*0.05f, cyFace - cropH*0.05f};
        lm.leftEye[4] = {cxFace - cropW*0.08f, cyFace - cropH*0.02f};
        lm.leftEye[5] = {cxFace - cropW*0.12f, cyFace - cropH*0.02f};

        lm.rightEye[0] = {cxFace + cropW*0.05f, cyFace - cropH*0.05f};
        lm.rightEye[1] = {cxFace + cropW*0.08f, cyFace - cropH*0.08f};
        lm.rightEye[2] = {cxFace + cropW*0.12f, cyFace - cropH*0.08f};
        lm.rightEye[3] = {cxFace + cropW*0.15f, cyFace - cropH*0.05f};
        lm.rightEye[4] = {cxFace + cropW*0.12f, cyFace - cropH*0.02f};
        lm.rightEye[5] = {cxFace + cropW*0.08f, cyFace - cropH*0.02f};

        lm.mouthOuter[0] = {cxFace - cropW*0.12f, cyFace + cropH*0.25f};
        lm.mouthOuter[1] = {cxFace - cropW*0.06f, cyFace + cropH*0.22f};
        lm.mouthOuter[2] = {cxFace, cyFace + cropH*0.21f};
        lm.mouthOuter[3] = {cxFace + cropW*0.12f, cyFace + cropH*0.25f};
        lm.mouthOuter[4] = {cxFace, cyFace + cropH*0.30f};
        lm.mouthOuter[5] = {cxFace - cropW*0.06f, cyFace + cropH*0.29f};

        lm.noseTip = {cxFace, cyFace + cropH*0.12f};
        lm.chin = {cxFace, cyFace + cropH*0.45f};
        lm.leftEyeCorner = lm.leftEye[0];
        lm.rightEyeCorner = lm.rightEye[3];
        lm.mouthLeft = lm.mouthOuter[0];
        lm.mouthRight = lm.mouthOuter[3];

        for (int i = 0; i < 6; i++) {
            lm.leftEye[i] = getLandmark(i);
            lm.rightEye[i] = getLandmark(i + 6);
        }
        if (numLandmarks > 12) {
            lm.noseTip = getLandmark(12);
        }
        lm.count = numLandmarks;
    }

    return lm;
}

void FacePipeline::updateLiveness(const LandmarkSet& landmarks, int frameW, int frameH) {
    livenessState_.totalFrames++;
    auto now = std::chrono::steady_clock::now();

    // ── 1. Compute current EAR / MAR ──
    float earL = eyeAspectRatio(
        landmarks.leftEye[0], landmarks.leftEye[1], landmarks.leftEye[2],
        landmarks.leftEye[3], landmarks.leftEye[4], landmarks.leftEye[5]);
    float earR = eyeAspectRatio(
        landmarks.rightEye[0], landmarks.rightEye[1], landmarks.rightEye[2],
        landmarks.rightEye[3], landmarks.rightEye[4], landmarks.rightEye[5]);

    livenessState_.earLeft = earL;
    livenessState_.earRight = earR;
    float avgEar = (earL + earR) * 0.5f;

    float marVal = mouthAspectRatio(
        landmarks.mouthOuter[0], landmarks.mouthOuter[3],
        landmarks.mouthOuter[2], landmarks.mouthOuter[4],
        landmarks.mouthOuter[1], landmarks.mouthOuter[5]);
    livenessState_.mar = marVal;

    // ── 2. Neutral baseline calibration (first N frames) ──
    if (livenessState_.neutralCalibFrames < NEUTRAL_CALIBRATION_FRAMES) {
        earAccumulator_ += avgEar;
        marAccumulator_ += marVal;
        livenessState_.neutralCalibFrames++;
        float n = static_cast<float>(livenessState_.neutralCalibFrames);
        livenessState_.neutralEar = earAccumulator_ / n;
        livenessState_.neutralMar = marAccumulator_ / n;
        livenessState_.status = LivenessStatus::Processing;
        return; // still calibrating, don't run challenge checks yet
    }

    // ── 3. Blink tracking with 300ms temporal window ──
    if (avgEar < EAR_THRESHOLD) {
        if (!blinkInProgress_) {
            blinkInProgress_ = true;
            blinkStartTime_ = now;
        }
        earLowFrames_++;
    } else {
        // Eyes are open now — did we just finish a valid blink?
        if (blinkInProgress_ && earLowFrames_ >= EAR_CONSEC_FRAMES) {
            auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - blinkStartTime_).count();
            if (elapsedMs <= BLINK_MAX_DURATION_MS) {
                // Valid blink: sharp drop and recovery within 300ms
                livenessState_.blinkDetected = true;
                livenessState_.blinkFrameCount = earLowFrames_;
                LOGI("Blink validated: %d frames, %lld ms", earLowFrames_, (long long)elapsedMs);
            } else {
                LOGI("Blink rejected: eyes closed too long (%lld ms > %d ms)",
                     (long long)elapsedMs, BLINK_MAX_DURATION_MS);
            }
        }
        blinkInProgress_ = false;
        earLowFrames_ = 0;
    }

    // ── 4. Smile tracking with 15% deviation from neutral ──
    float smileThreshold = livenessState_.neutralMar * (1.0f + SMILE_DEVIATION_RATIO);
    if (marVal > smileThreshold) {
        marHighFrames_++;
        if (marHighFrames_ >= MAR_CONSEC_FRAMES) {
            livenessState_.smileDetected = true;
            livenessState_.smileFrameCount = marHighFrames_;
        }
    } else {
        if (marHighFrames_ >= MAR_CONSEC_FRAMES) {
            livenessState_.smileDetected = true;
        }
        marHighFrames_ = 0;
    }

    // ── 5. Head turn (PnP yaw) — unchanged logic ──
    std::vector<Point3f> modelPoints = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, -330.0f, -65.0f},
        {-225.0f, 170.0f, -135.0f},
        {225.0f, 170.0f, -135.0f},
        {-150.0f, -150.0f, -125.0f},
        {150.0f, -150.0f, -125.0f}
    };

    std::vector<Point2f> imagePoints = {
        landmarks.noseTip,
        landmarks.chin,
        landmarks.leftEyeCorner,
        landmarks.rightEyeCorner,
        landmarks.mouthLeft,
        landmarks.mouthRight
    };

    float fx = static_cast<float>(frameW);
    float fy = static_cast<float>(frameW);
    float cx = static_cast<float>(frameW) * 0.5f;
    float cy = static_cast<float>(frameH) * 0.5f;

    PnPResult pnp = solvePnPIterative(modelPoints, imagePoints, fx, fy, cx, cy, 15);
    livenessState_.yawAngle = pnp.yaw;
    livenessState_.pitchAngle = pnp.pitch;

    if (std::abs(pnp.yaw) > YAW_THRESHOLD) {
        yawHighFrames_++;
        if (yawHighFrames_ >= YAW_CONSEC_FRAMES) {
            livenessState_.headTurnDetected = true;
            livenessState_.headTurnFrameCount = yawHighFrames_;
        }
    } else {
        if (yawHighFrames_ >= YAW_CONSEC_FRAMES) {
            livenessState_.headTurnDetected = true;
        }
        yawHighFrames_ = 0;
    }

    // ── 6. Static media detection (rolling landmark variance) ──
    std::array<float, 12> currentCoords = {{
        landmarks.noseTip.x,      landmarks.noseTip.y,
        landmarks.chin.x,         landmarks.chin.y,
        landmarks.leftEyeCorner.x,  landmarks.leftEyeCorner.y,
        landmarks.rightEyeCorner.x, landmarks.rightEyeCorner.y,
        landmarks.mouthLeft.x,    landmarks.mouthLeft.y,
        landmarks.mouthRight.x,   landmarks.mouthRight.y
    }};
    landmarkHistory_.push_back(currentCoords);
    if (static_cast<int>(landmarkHistory_.size()) > VARIANCE_WINDOW_FRAMES) {
        landmarkHistory_.erase(landmarkHistory_.begin());
    }

    float variance = computeLandmarkVariance(landmarkHistory_);
    livenessState_.coordVariance = variance;

    // Only flag spoof when we have a full window of data
    if (static_cast<int>(landmarkHistory_.size()) >= VARIANCE_WINDOW_FRAMES
        && variance < STATIC_VARIANCE_THRESHOLD) {
        livenessState_.spoofDetected = true;
        livenessState_.status = LivenessStatus::SpoofDetected;
        LOGE("Spoof Attack: Static Media Detected (variance=%.4f < %.4f)",
             variance, STATIC_VARIANCE_THRESHOLD);
        return;
    }

    // ── 7. Update sequential challenge status ──
    if (livenessState_.blinkDetected && livenessState_.smileDetected
        && livenessState_.headTurnDetected) {
        livenessState_.status = LivenessStatus::LivenessPassed;
    } else if (!livenessState_.blinkDetected) {
        livenessState_.status = LivenessStatus::BlinkRequired;
    } else if (!livenessState_.smileDetected) {
        livenessState_.status = LivenessStatus::SmileRequired;
    } else {
        // blink and smile done, waiting for head turn
        livenessState_.status = LivenessStatus::Processing;
    }
}

std::vector<float> FacePipeline::runRecognition(const uint8_t* alignedFace, int faceW, int faceH) {
    std::vector<float> emb;

    std::vector<uint8_t> resized(RECOGNIZER_INPUT_SIZE * RECOGNIZER_INPUT_SIZE * 3);
    bilinearResize(alignedFace, faceW, faceH, resized.data(), RECOGNIZER_INPUT_SIZE, RECOGNIZER_INPUT_SIZE);

    auto* inputTensor = TfLiteInterpreterGetInputTensor(recognizer_->interpreter, 0);
    if (TfLiteTensorType(inputTensor) == kTfLiteFloat32) {
        auto* input = (float*)TfLiteTensorData(TfLiteInterpreterGetInputTensor(recognizer_->interpreter, 0));
        normalizeToFloat(resized.data(), resized.size(), input, 127.5f, 127.5f);
    } else if (TfLiteTensorType(inputTensor) == kTfLiteUInt8) {
        auto* input = (uint8_t*)TfLiteTensorData(TfLiteInterpreterGetInputTensor(recognizer_->interpreter, 0));
        std::memcpy(input, resized.data(), resized.size());
    } else {
        auto* input = (float*)TfLiteTensorData(TfLiteInterpreterGetInputTensor(recognizer_->interpreter, 0));
        if (input) normalizeToFloat(resized.data(), resized.size(), input, 127.5f, 127.5f);
    }

    if (TfLiteInterpreterInvoke(recognizer_->interpreter) != kTfLiteOk) return emb;

    auto* outTensor = TfLiteInterpreterGetOutputTensor(recognizer_->interpreter, 0);
    int outputSize = 1;
    for (int i = 0; i < TfLiteTensorNumDims(outTensor); i++) {
        outputSize *= TfLiteTensorDim(outTensor, i);
    }

    emb.resize(outputSize);

    if (TfLiteTensorType(outTensor) == kTfLiteFloat32) {
        auto* output = (float*)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(recognizer_->interpreter, 0));
        std::memcpy(emb.data(), output, outputSize * sizeof(float));
    } else if (TfLiteTensorType(outTensor) == kTfLiteUInt8) {
        auto* output = (uint8_t*)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(recognizer_->interpreter, 0));
        float scale = TfLiteTensorQuantizationParams(outTensor).scale;
        int zeroPoint = TfLiteTensorQuantizationParams(outTensor).zero_point;
        for (int i = 0; i < outputSize; i++) {
            emb[i] = (static_cast<float>(output[i]) - zeroPoint) * scale;
        }
    }

    l2Normalize(emb.data(), outputSize);
    return emb;
}

PipelineResult FacePipeline::processFrame(const uint8_t* rgbData, int width, int height) {
    PipelineResult result;

    if (!initialized_) {
        result.error = "Pipeline not initialized";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto t0 = std::chrono::high_resolution_clock::now();

    FaceDetection det = runDetection(rgbData, width, height);
    if (det.confidence < 0.5f) {
        result.faceDetected = false;
        // Track consecutive no-face frames for non-face object rejection
        livenessState_.noFaceFrames++;
        if (livenessState_.noFaceFrames >= NO_FACE_TIMEOUT_FRAMES) {
            livenessState_.status = LivenessStatus::NoFaceDetected;
        }
        result.liveness = livenessState_;
        auto t1 = std::chrono::high_resolution_clock::now();
        result.inferenceTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        return result;
    }

    // Face found — reset the no-face counter
    livenessState_.noFaceFrames = 0;

    result.faceDetected = true;
    result.face = det;

    float pad = 0.2f;
    int cropX = std::max(0, static_cast<int>(det.x - det.w * pad));
    int cropY = std::max(0, static_cast<int>(det.y - det.h * pad));
    int cropW = std::min(width - cropX, static_cast<int>(det.w * (1.0f + 2.0f * pad)));
    int cropH = std::min(height - cropY, static_cast<int>(det.h * (1.0f + 2.0f * pad)));
    if (cropW <= 0 || cropH <= 0) {
        result.faceDetected = false;
        auto t1 = std::chrono::high_resolution_clock::now();
        result.inferenceTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        return result;
    }

    std::vector<uint8_t> croppedFace(cropW * cropH * 3);
    cropRgb(rgbData, width, height, cropX, cropY, cropW, cropH, croppedFace.data());

    LandmarkSet landmarks = runLandmarks(croppedFace.data(), cropW, cropH);

    updateLiveness(landmarks, width, height);
    result.liveness = livenessState_;

    // Calculate angle from eye landmarks for spatial alignment
    float lx = 0, ly = 0, rx = 0, ry = 0;
    for(int i = 0; i < 6; i++) {
        lx += landmarks.leftEye[i].x; ly += landmarks.leftEye[i].y;
        rx += landmarks.rightEye[i].x; ry += landmarks.rightEye[i].y;
    }
    lx /= 6.0f; ly /= 6.0f; rx /= 6.0f; ry /= 6.0f;
    float dx = rx - lx;
    float dy = ry - ly;
    float angleDeg = std::atan2(dy, dx) * 180.0f / 3.14159265f;

    // Apply affine rotation to align the face horizontally
    std::vector<uint8_t> alignedFace(cropW * cropH * 3);
    rotateAndResizeRgb(croppedFace.data(), cropW, cropH,
                       cropW * 0.5f, cropH * 0.5f, angleDeg,
                       alignedFace.data(), cropW, cropH, 1.0f);

    result.embedding = runRecognition(alignedFace.data(), cropW, cropH);

    // Vector Verification - Native Logging
    if (!targetEmbeddings_.empty() && !result.embedding.empty()) {
        float maxScore = -1.0f;
        for (const auto& targetEmb : targetEmbeddings_) {
            float score = 0.0f;
            int dim = std::min(result.embedding.size(), targetEmb.size());
            for (int i = 0; i < dim; i++) {
                score += result.embedding[i] * targetEmb[i]; // Embeddings are L2 normalized, dot product == cosine similarity
            }
            if (score > maxScore) maxScore = score;
        }
        LOGI("Verification Attempt - Vector Distance Score: %.2f", maxScore);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return result;
}

}

extern "C" {

static std::unique_ptr<nhai::FacePipeline> g_pipeline;

JNIEXPORT jboolean JNICALL
Java_com_anonymous_DatalakeApp_facepipeline_FacePipelineModule_nativeInitialize(
    JNIEnv* env, jobject thiz,
    jstring detectorPath, jstring landmarkPath, jstring recognizerPath,
    jint width, jint height) {

    const char* det = env->GetStringUTFChars(detectorPath, nullptr);
    const char* lm = env->GetStringUTFChars(landmarkPath, nullptr);
    const char* rec = env->GetStringUTFChars(recognizerPath, nullptr);

    g_pipeline = std::make_unique<nhai::FacePipeline>();
    bool ok = g_pipeline->initialize(det, lm, rec, width, height);

    env->ReleaseStringUTFChars(detectorPath, det);
    env->ReleaseStringUTFChars(landmarkPath, lm);
    env->ReleaseStringUTFChars(recognizerPath, rec);

    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobject JNICALL
Java_com_anonymous_DatalakeApp_facepipeline_FacePipelineModule_nativeProcessFrame(
    JNIEnv* env, jobject thiz,
    jbyteArray rgbData, jint width, jint height) {

    if (!g_pipeline) return nullptr;

    jbyte* data = env->GetByteArrayElements(rgbData, nullptr);
    int len = env->GetArrayLength(rgbData);

    auto result = g_pipeline->processFrame(
        reinterpret_cast<const uint8_t*>(data), width, height);

    env->ReleaseByteArrayElements(rgbData, data, JNI_ABORT);

    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableNativeMap");
    jmethodID mapInit = env->GetMethodID(mapClass, "<init>", "()V");
    jmethodID putBoolean = env->GetMethodID(mapClass, "putBoolean", "(Ljava/lang/String;Z)V");
    jmethodID putDouble = env->GetMethodID(mapClass, "putDouble", "(Ljava/lang/String;D)V");
    jmethodID putString = env->GetMethodID(mapClass, "putString", "(Ljava/lang/String;Ljava/lang/String;)V");
    jmethodID putMap = env->GetMethodID(mapClass, "putMap", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableMap;)V");

    jclass arrayClass = env->FindClass("com/facebook/react/bridge/WritableNativeArray");
    jmethodID arrayInit = env->GetMethodID(arrayClass, "<init>", "()V");
    jmethodID pushDouble = env->GetMethodID(arrayClass, "pushDouble", "(D)V");
    jmethodID putArray = env->GetMethodID(mapClass, "putArray", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableArray;)V");

    jobject rootMap = env->NewObject(mapClass, mapInit);

    env->CallVoidMethod(rootMap, putBoolean, env->NewStringUTF("faceDetected"), result.faceDetected);
    env->CallVoidMethod(rootMap, putDouble, env->NewStringUTF("inferenceTimeMs"), (double)result.inferenceTimeMs);

    if (result.faceDetected) {
        jobject faceBoundsMap = env->NewObject(mapClass, mapInit);
        env->CallVoidMethod(faceBoundsMap, putDouble, env->NewStringUTF("x"), (double)result.face.x);
        env->CallVoidMethod(faceBoundsMap, putDouble, env->NewStringUTF("y"), (double)result.face.y);
        env->CallVoidMethod(faceBoundsMap, putDouble, env->NewStringUTF("width"), (double)result.face.w);
        env->CallVoidMethod(faceBoundsMap, putDouble, env->NewStringUTF("height"), (double)result.face.h);
        env->CallVoidMethod(rootMap, putMap, env->NewStringUTF("faceBounds"), faceBoundsMap);
    }

    jobject livenessMap = env->NewObject(mapClass, mapInit);
    env->CallVoidMethod(livenessMap, putBoolean, env->NewStringUTF("blinkDetected"), result.liveness.blinkDetected);
    env->CallVoidMethod(livenessMap, putBoolean, env->NewStringUTF("smileDetected"), result.liveness.smileDetected);
    env->CallVoidMethod(livenessMap, putBoolean, env->NewStringUTF("headTurnDetected"), result.liveness.headTurnDetected);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("earLeft"), (double)result.liveness.earLeft);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("earRight"), (double)result.liveness.earRight);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("mar"), (double)result.liveness.mar);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("yawAngle"), (double)result.liveness.yawAngle);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("pitchAngle"), (double)result.liveness.pitchAngle);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("totalFrames"), (double)result.liveness.totalFrames);
    
    // Anti-spoofing fields MUST be added BEFORE putMap, otherwise ObjectAlreadyConsumedException is thrown
    env->CallVoidMethod(livenessMap, putBoolean, env->NewStringUTF("spoofDetected"), result.liveness.spoofDetected);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("livenessStatus"), (double)static_cast<int>(result.liveness.status));
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("coordVariance"), (double)result.liveness.coordVariance);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("neutralEar"), (double)result.liveness.neutralEar);
    env->CallVoidMethod(livenessMap, putDouble, env->NewStringUTF("neutralMar"), (double)result.liveness.neutralMar);

    env->CallVoidMethod(rootMap, putMap, env->NewStringUTF("liveness"), livenessMap);

    if (!result.embedding.empty()) {
        jobject embArray = env->NewObject(arrayClass, arrayInit);
        for (float v : result.embedding) {
            env->CallVoidMethod(embArray, pushDouble, (double)v);
        }
        env->CallVoidMethod(rootMap, putArray, env->NewStringUTF("embedding"), embArray);
    }

    if (!result.error.empty()) {
        env->CallVoidMethod(rootMap, putString, env->NewStringUTF("error"), env->NewStringUTF(result.error.c_str()));
    }

    return rootMap;
}

JNIEXPORT void JNICALL
Java_com_anonymous_DatalakeApp_facepipeline_FacePipelineModule_nativeResetLiveness(
    JNIEnv* env, jobject thiz) {
    if (g_pipeline) g_pipeline->resetLiveness();
}

JNIEXPORT jint JNICALL
Java_com_anonymous_DatalakeApp_facepipeline_FacePipelineModule_nativeGetEmbeddingDim(
    JNIEnv* env, jobject thiz) {
    return g_pipeline ? g_pipeline->getEmbeddingDim() : 128;
}

JNIEXPORT void JNICALL
Java_com_anonymous_DatalakeApp_facepipeline_FacePipelineModule_nativeSetTargetEmbeddings(
    JNIEnv* env, jobject thiz, jobjectArray embeddingsArray) {
    if (!g_pipeline) return;

    int numEmbeddings = env->GetArrayLength(embeddingsArray);
    std::vector<std::vector<float>> targetEmbeddings;
    targetEmbeddings.reserve(numEmbeddings);

    for (int i = 0; i < numEmbeddings; i++) {
        jfloatArray floatArray = (jfloatArray) env->GetObjectArrayElement(embeddingsArray, i);
        if (floatArray != nullptr) {
            int len = env->GetArrayLength(floatArray);
            jfloat* elements = env->GetFloatArrayElements(floatArray, nullptr);
            std::vector<float> emb(elements, elements + len);
            targetEmbeddings.push_back(emb);
            env->ReleaseFloatArrayElements(floatArray, elements, JNI_ABORT);
            env->DeleteLocalRef(floatArray);
        }
    }

    g_pipeline->setTargetEmbeddings(targetEmbeddings);
}

}
