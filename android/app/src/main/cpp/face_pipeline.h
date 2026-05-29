#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <mutex>
#include <chrono>
#include "math_utils.h"

namespace nhai {

// Liveness status enum — streamed to JS as integers
enum class LivenessStatus : int {
    Processing     = 0,
    BlinkRequired  = 1,
    SmileRequired  = 2,
    LivenessPassed = 3,
    SpoofDetected  = 4,
    NoFaceDetected = 5
};

struct FaceDetection {
    float x, y, w, h;
    float confidence;
    Point2f keypoints[6];
};

struct LivenessState {
    bool blinkDetected = false;
    bool smileDetected = false;
    bool headTurnDetected = false;
    float earLeft = 0.0f;
    float earRight = 0.0f;
    float mar = 0.0f;
    float yawAngle = 0.0f;
    float pitchAngle = 0.0f;
    int blinkFrameCount = 0;
    int smileFrameCount = 0;
    int headTurnFrameCount = 0;
    int totalFrames = 0;

    // Anti-spoofing state
    LivenessStatus status = LivenessStatus::Processing;
    bool spoofDetected = false;
    float coordVariance = 0.0f;     // variance of key landmarks over rolling window
    float neutralEar = 0.0f;        // calibrated neutral-state EAR baseline
    float neutralMar = 0.0f;        // calibrated neutral-state MAR baseline
    int neutralCalibFrames = 0;     // frames used for neutral calibration
    int noFaceFrames = 0;           // consecutive frames with no face
};

struct PipelineResult {
    bool faceDetected = false;
    FaceDetection face;
    LivenessState liveness;
    std::vector<float> embedding;
    float inferenceTimeMs = 0.0f;
    std::string error;
};

struct LandmarkSet {
    Point2f leftEye[6];
    Point2f rightEye[6];
    Point2f mouthOuter[6];
    Point2f noseTip;
    Point2f chin;
    Point2f leftEyeCorner;
    Point2f rightEyeCorner;
    Point2f mouthLeft;
    Point2f mouthRight;
    int count = 0;
};

class FacePipeline {
public:
    FacePipeline();
    ~FacePipeline();

    bool initialize(const std::string& detectorPath,
                    const std::string& landmarkPath,
                    const std::string& recognizerPath,
                    int inputWidth, int inputHeight);

    PipelineResult processFrame(const uint8_t* rgbData, int width, int height);
    void resetLiveness();
    
    // Set target embeddings for real-time verification matching in native layer
    void setTargetEmbeddings(const std::vector<std::vector<float>>& embeddings);

    int getEmbeddingDim() const { return embeddingDim_; }

private:
    struct TFLiteModel;

    FaceDetection runDetection(const uint8_t* rgbData, int width, int height);
    LandmarkSet runLandmarks(const uint8_t* croppedFace, int cropW, int cropH);
    void updateLiveness(const LandmarkSet& landmarks, int frameW, int frameH);
    std::vector<float> runRecognition(const uint8_t* alignedFace, int faceW, int faceH);

    std::unique_ptr<TFLiteModel> detector_;
    std::unique_ptr<TFLiteModel> landmarker_;
    std::unique_ptr<TFLiteModel> recognizer_;
    
    std::vector<std::vector<float>> targetEmbeddings_;

    LivenessState livenessState_;
    std::mutex mutex_;
    bool initialized_ = false;
    int embeddingDim_ = 128;

    // ── Detection thresholds (tuned for ~3fps photo-capture) ──
    static constexpr float EAR_THRESHOLD = 0.22f;
    static constexpr int EAR_CONSEC_FRAMES = 1;              // may only get 1 frame with eyes closed at 3fps
    static constexpr float MAR_THRESHOLD = 0.6f;
    static constexpr int MAR_CONSEC_FRAMES = 1;              // same: 1 frame is enough at 3fps
    static constexpr float YAW_THRESHOLD = 12.0f;            // slightly more lenient
    static constexpr int YAW_CONSEC_FRAMES = 1;

    // ── Anti-spoofing thresholds ──
    static constexpr int VARIANCE_WINDOW_FRAMES = 20;        // ~6s at 3fps (was 60 at 30fps)
    static constexpr float STATIC_VARIANCE_THRESHOLD = 0.005f;// very low to avoid false positives at 3fps
    static constexpr int NEUTRAL_CALIBRATION_FRAMES = 5;     // ~1.5s at 3fps for baseline
    static constexpr float SMILE_DEVIATION_RATIO = 0.12f;    // 12% deviation from neutral (slightly easier)
    static constexpr int NO_FACE_TIMEOUT_FRAMES = 3;
    static constexpr int BLINK_MAX_DURATION_MS = 1500;       // allow blink across multiple 300ms captures

    static constexpr int DETECTOR_INPUT_SIZE = 128;
    static constexpr int LANDMARK_INPUT_SIZE = 192;
    static constexpr int RECOGNIZER_INPUT_SIZE = 112;

    // ── Existing frame counters ──
    int earLowFrames_ = 0;
    int marHighFrames_ = 0;
    int yawHighFrames_ = 0;

    // ── Temporal blink tracking ──
    std::chrono::steady_clock::time_point blinkStartTime_;
    bool blinkInProgress_ = false;

    // ── Rolling landmark history for variance (6 key points × 2 coords = 12 floats) ──
    std::vector<std::array<float, 12>> landmarkHistory_;

    // ── Neutral baseline accumulators ──
    float earAccumulator_ = 0.0f;
    float marAccumulator_ = 0.0f;
};

}
