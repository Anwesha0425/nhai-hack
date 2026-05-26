import React, { useEffect, useState, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, useFrameOutput } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { FaceModelService } from '../services/FaceModelService';

interface Props {
  onAuthenticationResult: (success: boolean, message: string) => void;
  mode: 'register' | 'authenticate';
}

export const CameraScreen: React.FC<Props> = ({ onAuthenticationResult, mode }) => {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();
  const [isActive, setIsActive] = useState(true);
  const [statusText, setStatusText] = useState('Initializing...');
  const isProcessing = useRef(false);

  useEffect(() => {
    (async () => {
      if (!hasPermission) {
        await requestPermission();
      }
      
      // Initialize FaceModelService
      try {
        await FaceModelService.initialize();
        setStatusText(mode === 'register' ? 'Look at the camera to register' : 'Look at the camera to authenticate');
      } catch (e) {
        setStatusText('Failed to load models.');
        console.error(e);
      }
    })();
  }, [mode, hasPermission, requestPermission]);

  const handleResult = Worklets.createRunOnJS((success: boolean, message: string) => {
    setStatusText(message);
    if (success) {
      setIsActive(false);
      setTimeout(() => {
        onAuthenticationResult(true, message);
      }, 1000);
    } else {
      isProcessing.current = false;
    }
  });

  const frameOutput = useFrameOutput({
    onFrame: (frame) => {
      'worklet';
      if (isProcessing.current || !isActive) {
        frame.dispose();
        return;
      }

      try {
        isProcessing.current = true;
        // Mock processing
        const result = FaceModelService.processFrame(frame, mode);
        
        if (result.status === 'success') {
          handleResult(true, result.message);
        } else if (result.status === 'failed') {
          handleResult(false, result.message);
        } else {
          isProcessing.current = false;
        }
      } catch (e) {
        console.error(e);
        isProcessing.current = false;
      } finally {
        frame.dispose();
      }
    }
  });

  if (!hasPermission) return <View style={styles.container}><Text style={{color: 'white'}}>No Camera Permission</Text></View>;
  if (device == null) return <View style={styles.container}><Text style={{color: 'white'}}>No Front Camera</Text></View>;

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        outputs={[frameOutput]}
      />
      
      <View style={styles.overlay}>
        <View style={styles.guideBox} />
      </View>

      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>{statusText}</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  guideBox: {
    width: 250,
    height: 300,
    borderWidth: 2,
    borderColor: '#00FF00',
    borderRadius: 20,
    backgroundColor: 'transparent',
  },
  statusContainer: {
    position: 'absolute',
    bottom: 50,
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 15,
    borderRadius: 10,
  },
  statusText: {
    color: '#FFF',
    fontSize: 16,
    textAlign: 'center',
  },
});
