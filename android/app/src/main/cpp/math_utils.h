#pragma once
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>

namespace nhai {

struct Point2f { float x, y; };
struct Point3f { float x, y, z; };

inline float euclideanDist2D(const Point2f& a, const Point2f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline float euclideanDist3D(const Point3f& a, const Point3f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline float eyeAspectRatio(const Point2f& p1, const Point2f& p2,
                            const Point2f& p3, const Point2f& p4,
                            const Point2f& p5, const Point2f& p6) {
    float vertical1 = euclideanDist2D(p2, p6);
    float vertical2 = euclideanDist2D(p3, p5);
    float horizontal = euclideanDist2D(p1, p4);
    if (horizontal < 1e-6f) return 0.0f;
    return (vertical1 + vertical2) / (2.0f * horizontal);
}

inline float mouthAspectRatio(const Point2f& left, const Point2f& right,
                              const Point2f& topInner, const Point2f& bottomInner,
                              const Point2f& topOuter, const Point2f& bottomOuter) {
    float vertical1 = euclideanDist2D(topInner, bottomInner);
    float vertical2 = euclideanDist2D(topOuter, bottomOuter);
    float horizontal = euclideanDist2D(left, right);
    if (horizontal < 1e-6f) return 0.0f;
    return (vertical1 + vertical2) / (2.0f * horizontal);
}

inline float cosineSimilarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA < 1e-12f || normB < 1e-12f) return 0.0f;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

inline void l2Normalize(float* vec, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += vec[i] * vec[i];
    norm = std::sqrt(norm);
    if (norm < 1e-12f) return;
    float invNorm = 1.0f / norm;
    for (int i = 0; i < dim; i++) vec[i] *= invNorm;
}

struct Mat3x3 {
    float data[9] = {};
    float& operator()(int r, int c) { return data[r * 3 + c]; }
    float operator()(int r, int c) const { return data[r * 3 + c]; }
};

struct Vec3 {
    float x, y, z;
    float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

inline Mat3x3 rodrigues(const Vec3& rvec) {
    float theta = std::sqrt(rvec.x * rvec.x + rvec.y * rvec.y + rvec.z * rvec.z);
    Mat3x3 R;
    if (theta < 1e-6f) {
        R(0,0) = 1; R(0,1) = 0; R(0,2) = 0;
        R(1,0) = 0; R(1,1) = 1; R(1,2) = 0;
        R(2,0) = 0; R(2,1) = 0; R(2,2) = 1;
        return R;
    }
    float c = std::cos(theta);
    float s = std::sin(theta);
    float c1 = 1.0f - c;
    float itheta = 1.0f / theta;
    float rx = rvec.x * itheta, ry = rvec.y * itheta, rz = rvec.z * itheta;

    R(0,0) = c + rx * rx * c1;
    R(0,1) = rx * ry * c1 - rz * s;
    R(0,2) = rx * rz * c1 + ry * s;
    R(1,0) = ry * rx * c1 + rz * s;
    R(1,1) = c + ry * ry * c1;
    R(1,2) = ry * rz * c1 - rx * s;
    R(2,0) = rz * rx * c1 - ry * s;
    R(2,1) = rz * ry * c1 + rx * s;
    R(2,2) = c + rz * rz * c1;
    return R;
}

inline Vec3 rotationMatrixToEuler(const Mat3x3& R) {
    Vec3 euler;
    float sy = std::sqrt(R(0,0) * R(0,0) + R(1,0) * R(1,0));
    if (sy > 1e-6f) {
        euler.x = std::atan2(R(2,1), R(2,2));
        euler.y = std::atan2(-R(2,0), sy);
        euler.z = std::atan2(R(1,0), R(0,0));
    } else {
        euler.x = std::atan2(-R(1,2), R(1,1));
        euler.y = std::atan2(-R(2,0), sy);
        euler.z = 0;
    }
    const float rad2deg = 180.0f / 3.14159265358979f;
    euler.x *= rad2deg;
    euler.y *= rad2deg;
    euler.z *= rad2deg;
    return euler;
}

struct PnPResult {
    Vec3 rvec;
    Vec3 tvec;
    float yaw, pitch, roll;
};

inline PnPResult solvePnPIterative(
    const std::vector<Point3f>& objectPoints,
    const std::vector<Point2f>& imagePoints,
    float fx, float fy, float cx, float cy,
    int maxIter = 25)
{
    PnPResult result = {};
    int n = std::min(objectPoints.size(), imagePoints.size());
    if (n < 4) return result;

    float meanOx = 0, meanOy = 0, meanOz = 0;
    for (auto& p : objectPoints) { meanOx += p.x; meanOy += p.y; meanOz += p.z; }
    meanOx /= n; meanOy /= n; meanOz /= n;

    float meanIx = 0, meanIy = 0;
    for (auto& p : imagePoints) { meanIx += p.x; meanIy += p.y; }
    meanIx /= n; meanIy /= n;

    result.tvec = {(meanIx - cx) / fx, (meanIy - cy) / fy, 3.0f};
    result.rvec = {0, 0, 0};

    for (int iter = 0; iter < maxIter; iter++) {
        Mat3x3 R = rodrigues(result.rvec);
        float JtJ[6][6] = {};
        float Jte[6] = {};

        for (int i = 0; i < n; i++) {
            float X = R(0,0)*objectPoints[i].x + R(0,1)*objectPoints[i].y + R(0,2)*objectPoints[i].z + result.tvec.x;
            float Y = R(1,0)*objectPoints[i].x + R(1,1)*objectPoints[i].y + R(1,2)*objectPoints[i].z + result.tvec.y;
            float Z = R(2,0)*objectPoints[i].x + R(2,1)*objectPoints[i].y + R(2,2)*objectPoints[i].z + result.tvec.z;

            if (std::abs(Z) < 1e-6f) Z = 1e-6f;
            float invZ = 1.0f / Z;
            float px = fx * X * invZ + cx;
            float py = fy * Y * invZ + cy;

            float ex = imagePoints[i].x - px;
            float ey = imagePoints[i].y - py;

            float dXdr[3], dYdr[3], dZdr[3];
            dXdr[0] = 0;
            dXdr[1] = Z;
            dXdr[2] = -Y;
            dYdr[0] = -Z;
            dYdr[1] = 0;
            dYdr[2] = X;
            dZdr[0] = Y;
            dZdr[1] = -X;
            dZdr[2] = 0;

            float J[2][6];
            for (int j = 0; j < 3; j++) {
                J[0][j] = fx * (dXdr[j] * Z - X * dZdr[j]) * invZ * invZ;
                J[1][j] = fy * (dYdr[j] * Z - Y * dZdr[j]) * invZ * invZ;
            }
            J[0][3] = fx * invZ;
            J[0][4] = 0;
            J[0][5] = -fx * X * invZ * invZ;
            J[1][3] = 0;
            J[1][4] = fy * invZ;
            J[1][5] = -fy * Y * invZ * invZ;

            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 6; c++) {
                    JtJ[r][c] += J[0][r] * J[0][c] + J[1][r] * J[1][c];
                }
                Jte[r] += J[0][r] * ex + J[1][r] * ey;
            }
        }

        for (int i = 0; i < 6; i++) JtJ[i][i] += 1e-4f;

        float aug[6][7];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) aug[i][j] = JtJ[i][j];
            aug[i][6] = Jte[i];
        }
        for (int i = 0; i < 6; i++) {
            int maxRow = i;
            for (int k = i+1; k < 6; k++)
                if (std::abs(aug[k][i]) > std::abs(aug[maxRow][i])) maxRow = k;
            for (int j = 0; j < 7; j++) std::swap(aug[i][j], aug[maxRow][j]);
            if (std::abs(aug[i][i]) < 1e-10f) continue;
            float pivInv = 1.0f / aug[i][i];
            for (int j = i; j < 7; j++) aug[i][j] *= pivInv;
            for (int k = 0; k < 6; k++) {
                if (k == i) continue;
                float factor = aug[k][i];
                for (int j = i; j < 7; j++) aug[k][j] -= factor * aug[i][j];
            }
        }

        float delta[6];
        for (int i = 0; i < 6; i++) delta[i] = aug[i][6];

        result.rvec.x += delta[0];
        result.rvec.y += delta[1];
        result.rvec.z += delta[2];
        result.tvec.x += delta[3];
        result.tvec.y += delta[4];
        result.tvec.z += delta[5];

        float normDelta = 0;
        for (int i = 0; i < 6; i++) normDelta += delta[i] * delta[i];
        if (normDelta < 1e-10f) break;
    }

    Mat3x3 Rfinal = rodrigues(result.rvec);
    Vec3 euler = rotationMatrixToEuler(Rfinal);
    result.pitch = euler.x;
    result.yaw = euler.y;
    result.roll = euler.z;
    return result;
}

// ── Variance computation for static media detection ──

inline float computeVariance(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    float mean = std::accumulate(values.begin(), values.end(), 0.0f)
                 / static_cast<float>(values.size());
    float variance = 0.0f;
    for (float v : values) {
        float diff = v - mean;
        variance += diff * diff;
    }
    return variance / static_cast<float>(values.size());
}

// Compute combined variance of all coordinate channels across a landmark
// history buffer.  Returns a high value when there is insufficient data
// so the caller does not trigger a false-positive spoof flag on startup.
inline float computeLandmarkVariance(
    const std::vector<std::array<float, 12>>& history) {
    if (history.empty()) return 999.0f;
    float totalVariance = 0.0f;
    for (int ch = 0; ch < 12; ch++) {
        std::vector<float> channel;
        channel.reserve(history.size());
        for (const auto& frame : history) {
            channel.push_back(frame[static_cast<size_t>(ch)]);
        }
        totalVariance += computeVariance(channel);
    }
    return totalVariance;
}

}
