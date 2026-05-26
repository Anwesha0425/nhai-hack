import { StatusBar } from 'expo-status-bar';
import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, SafeAreaView, Dimensions } from 'react-native';
import { CameraScreen } from './src/components/CameraScreen';
import { SyncService } from './src/services/SyncService';

type Screen = 'home' | 'register' | 'authenticate';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('home');

  // Initialize sync service
  React.useEffect(() => {
    // This is just to ensure NetInfo listeners are attached
    console.log("App started. SyncService initialized.");
  }, []);

  const handleAuthenticationResult = (success: boolean, message: string) => {
    if (currentScreen === 'authenticate') {
      SyncService.logAttempt(success ? 'success' : 'failed');
    }
    alert(message);
    setCurrentScreen('home');
  };

  if (currentScreen === 'register' || currentScreen === 'authenticate') {
    return (
      <View style={styles.container}>
        <CameraScreen 
          mode={currentScreen} 
          onAuthenticationResult={handleAuthenticationResult} 
        />
        <TouchableOpacity 
          style={styles.backButton} 
          onPress={() => setCurrentScreen('home')}
        >
          <Text style={styles.backButtonText}>Cancel</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.title}>Datalake</Text>
        <Text style={styles.subtitle}>Secure Offline Authentication</Text>
      </View>

      <View style={styles.cardContainer}>
        <TouchableOpacity 
          style={[styles.button, styles.primaryButton]} 
          onPress={() => setCurrentScreen('authenticate')}
        >
          <Text style={styles.buttonText}>Authenticate</Text>
          <Text style={styles.buttonSubText}>Verify face offline</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.button, styles.secondaryButton]} 
          onPress={() => setCurrentScreen('register')}
        >
          <Text style={styles.secondaryButtonText}>Register New Face</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const { width } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A', // Dark aesthetic
  },
  header: {
    alignItems: 'center',
    marginTop: 80,
    marginBottom: 60,
  },
  title: {
    fontSize: 42,
    fontWeight: '800',
    color: '#38BDF8',
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 16,
    color: '#94A3B8',
    marginTop: 8,
    fontWeight: '500',
  },
  cardContainer: {
    flex: 1,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  button: {
    width: width - 48,
    paddingVertical: 20,
    paddingHorizontal: 24,
    borderRadius: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 8,
  },
  primaryButton: {
    backgroundColor: '#0EA5E9',
    borderWidth: 1,
    borderColor: '#38BDF8',
  },
  secondaryButton: {
    backgroundColor: 'rgba(30, 41, 59, 0.7)',
    borderWidth: 1,
    borderColor: '#334155',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 22,
    fontWeight: '700',
  },
  buttonSubText: {
    color: '#E0F2FE',
    fontSize: 14,
    marginTop: 4,
  },
  secondaryButtonText: {
    color: '#CBD5E1',
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
  },
  backButton: {
    position: 'absolute',
    top: 50,
    left: 20,
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 10,
  },
  backButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
});
