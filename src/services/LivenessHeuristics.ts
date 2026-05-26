import { Frame } from 'react-native-vision-camera';

class LivenessHeuristicsImpl {
  private frameCount = 0;
  private livenessPassed = false;

  /**
   * Resets the liveness state.
   */
  public reset() {
    this.frameCount = 0;
    this.livenessPassed = false;
  }

  /**
   * Checks for liveness signs in the frame. Must be called from a worklet.
   * For this prototype, we simulate a liveness check that passes after a few frames
   * to represent "detecting a blink or smile".
   */
  public checkLiveness(frame: Frame): boolean {
    'worklet';

    if (this.livenessPassed) {
      return true;
    }

    this.frameCount++;

    // In a real implementation:
    // 1. Run BlazeFace or FaceMesh
    // 2. Extract eye landmarks and calculate EAR (Eye Aspect Ratio)
    // 3. If EAR < threshold for 1-2 frames then EAR > threshold, blink detected!

    // Mock: pass liveness after 30 frames (approx 1 second of looking at the camera)
    if (this.frameCount > 30) {
      this.livenessPassed = true;
      return true;
    }

    return false;
  }
}

export const LivenessHeuristics = new LivenessHeuristicsImpl();
