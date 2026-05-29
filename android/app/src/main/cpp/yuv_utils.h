#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace nhai {

inline void nv21ToRgb(const uint8_t* yuv, int width, int height, uint8_t* rgb) {
    const uint8_t* yPlane = yuv;
    const uint8_t* uvPlane = yuv + width * height;
    int frameSize = width * height;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int yIdx = j * width + i;
            int uvIdx = (j >> 1) * width + (i & ~1);

            int Y = static_cast<int>(yPlane[yIdx]) - 16;
            int V = static_cast<int>(uvPlane[uvIdx]) - 128;
            int U = static_cast<int>(uvPlane[uvIdx + 1]) - 128;

            Y = std::max(0, Y);
            int R = (298 * Y + 409 * V + 128) >> 8;
            int G = (298 * Y - 100 * U - 208 * V + 128) >> 8;
            int B = (298 * Y + 516 * U + 128) >> 8;

            int outIdx = (j * width + i) * 3;
            rgb[outIdx]     = static_cast<uint8_t>(std::clamp(R, 0, 255));
            rgb[outIdx + 1] = static_cast<uint8_t>(std::clamp(G, 0, 255));
            rgb[outIdx + 2] = static_cast<uint8_t>(std::clamp(B, 0, 255));
        }
    }
}

inline void yuvToRgb(const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane,
                     int yRowStride, int uvRowStride, int uvPixelStride,
                     int width, int height, uint8_t* rgb) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int Y = static_cast<int>(yPlane[j * yRowStride + i]) - 16;
            int uvRow = (j >> 1) * uvRowStride;
            int uvCol = (i >> 1) * uvPixelStride;
            int U = static_cast<int>(uPlane[uvRow + uvCol]) - 128;
            int V = static_cast<int>(vPlane[uvRow + uvCol]) - 128;

            Y = std::max(0, Y);
            int R = (298 * Y + 409 * V + 128) >> 8;
            int G = (298 * Y - 100 * U - 208 * V + 128) >> 8;
            int B = (298 * Y + 516 * U + 128) >> 8;

            int outIdx = (j * width + i) * 3;
            rgb[outIdx]     = static_cast<uint8_t>(std::clamp(R, 0, 255));
            rgb[outIdx + 1] = static_cast<uint8_t>(std::clamp(G, 0, 255));
            rgb[outIdx + 2] = static_cast<uint8_t>(std::clamp(B, 0, 255));
        }
    }
}

inline void cropRgb(const uint8_t* src, int srcW, int srcH,
                    int x, int y, int cropW, int cropH, uint8_t* dst) {
    x = std::max(0, std::min(x, srcW - cropW));
    y = std::max(0, std::min(y, srcH - cropH));
    cropW = std::min(cropW, srcW - x);
    cropH = std::min(cropH, srcH - y);
    for (int j = 0; j < cropH; j++) {
        std::memcpy(dst + j * cropW * 3, src + ((y + j) * srcW + x) * 3, cropW * 3);
    }
}

inline void bilinearResize(const uint8_t* src, int srcW, int srcH,
                           uint8_t* dst, int dstW, int dstH) {
    float xRatio = static_cast<float>(srcW) / dstW;
    float yRatio = static_cast<float>(srcH) / dstH;

    for (int j = 0; j < dstH; j++) {
        float srcY = j * yRatio;
        int y0 = static_cast<int>(srcY);
        int y1 = std::min(y0 + 1, srcH - 1);
        float fy = srcY - y0;

        for (int i = 0; i < dstW; i++) {
            float srcX = i * xRatio;
            int x0 = static_cast<int>(srcX);
            int x1 = std::min(x0 + 1, srcW - 1);
            float fx = srcX - x0;

            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * srcW + x0) * 3 + c];
                float v01 = src[(y0 * srcW + x1) * 3 + c];
                float v10 = src[(y1 * srcW + x0) * 3 + c];
                float v11 = src[(y1 * srcW + x1) * 3 + c];
                float val = v00 * (1-fx) * (1-fy) + v01 * fx * (1-fy) +
                            v10 * (1-fx) * fy + v11 * fx * fy;
                dst[(j * dstW + i) * 3 + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

inline void rotateAndResizeRgb(const uint8_t* src, int srcW, int srcH,
                               float cx, float cy, float angleDeg,
                               uint8_t* dst, int dstW, int dstH, float cropScale = 1.0f) {
    float angleRad = angleDeg * 3.14159265f / 180.0f;
    float cosA = std::cos(angleRad);
    float sinA = std::sin(angleRad);

    float halfDstW = dstW * 0.5f;
    float halfDstH = dstH * 0.5f;

    for (int j = 0; j < dstH; j++) {
        float dy = (j - halfDstH) * cropScale;
        for (int i = 0; i < dstW; i++) {
            float dx = (i - halfDstW) * cropScale;

            // Rotate back from destination to source coordinates
            float srcX = cx + (dx * cosA - dy * sinA);
            float srcY = cy + (dx * sinA + dy * cosA);

            int x0 = static_cast<int>(std::floor(srcX));
            int y0 = static_cast<int>(std::floor(srcY));
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float fx = srcX - x0;
            float fy = srcY - y0;

            for (int c = 0; c < 3; c++) {
                auto getPixel = [&](int px, int py) -> float {
                    if (px < 0 || px >= srcW || py < 0 || py >= srcH) return 0.0f;
                    return src[(py * srcW + px) * 3 + c];
                };

                float v00 = getPixel(x0, y0);
                float v01 = getPixel(x1, y0);
                float v10 = getPixel(x0, y1);
                float v11 = getPixel(x1, y1);

                float val = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                            v10 * (1 - fx) * fy + v11 * fx * fy;
                dst[(j * dstW + i) * 3 + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

inline void normalizeToFloat(const uint8_t* src, int count, float* dst,
                             float mean = 127.5f, float std = 127.5f) {
    float invStd = 1.0f / std;
    for (int i = 0; i < count; i++) {
        dst[i] = (static_cast<float>(src[i]) - mean) * invStd;
    }
}

inline void normalizeToUint8Quantized(const uint8_t* src, int count, uint8_t* dst) {
    std::memcpy(dst, src, count);
}

inline void flipHorizontalRgb(uint8_t* img, int width, int height) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width / 2; i++) {
            int leftIdx = (j * width + i) * 3;
            int rightIdx = (j * width + (width - 1 - i)) * 3;
            for (int c = 0; c < 3; c++) {
                std::swap(img[leftIdx + c], img[rightIdx + c]);
            }
        }
    }
}

}
