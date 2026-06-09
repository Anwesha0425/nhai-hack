/**
 * AdminPinGate.tsx
 *
 * A full-screen PIN pad that guards admin actions (Register, Users, Clear Data).
 * First-time use: prompts to SET a 4-digit PIN.
 * Subsequent use: prompts to ENTER the PIN.
 *
 * Props:
 *   visible       — controls modal visibility
 *   onSuccess     — called when PIN is verified / set successfully
 *   onCancel      — called when user taps Cancel
 *   mode          — 'verify' (default) | 'setup'
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Vibration,
  Platform,
} from 'react-native';
import {
  isPinConfigured,
  setAdminPin,
  verifyAdminPin,
} from '../services/AdminPinService';

// ── Types ─────────────────────────────────────────────────────────────────────

type PinGateMode = 'verify' | 'setup';

interface AdminPinGateProps {
  visible: boolean;
  onSuccess: () => void;
  onCancel: () => void;
  /** Override mode. If not provided, auto-detects (setup vs verify). */
  mode?: PinGateMode;
  /** Accent colour to match parent theme */
  accentColor?: string;
  /** Background colour */
  bgColor?: string;
  /** Text colour */
  textColor?: string;
}

// ── PIN Dot display ───────────────────────────────────────────────────────────

function PinDots({ length, filled, shake }: { length: number; filled: number; shake: Animated.Value }) {
  return (
    <Animated.View style={[pd.row, { transform: [{ translateX: shake }] }]}>
      {Array.from({ length }).map((_, i) => (
        <View
          key={i}
          style={[
            pd.dot,
            i < filled ? pd.dotFilled : pd.dotEmpty,
          ]}
        />
      ))}
    </Animated.View>
  );
}

const pd = StyleSheet.create({
  row: { flexDirection: 'row', gap: 14, marginBottom: 32 },
  dot: { width: 16, height: 16, borderRadius: 8, borderWidth: 2, borderColor: '#38BDF8' },
  dotFilled: { backgroundColor: '#38BDF8' },
  dotEmpty: { backgroundColor: 'transparent' },
});

// ── Numpad ────────────────────────────────────────────────────────────────────

const KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '0', '⌫'];

function NumPad({
  onKey,
  accentColor,
}: {
  onKey: (k: string) => void;
  accentColor: string;
}) {
  return (
    <View style={np.grid}>
      {KEYS.map((k, i) => (
        <TouchableOpacity
          key={i}
          style={[np.key, k === '' && { opacity: 0 }]}
          onPress={() => k && onKey(k)}
          activeOpacity={0.6}
          disabled={k === ''}
        >
          {k === '⌫' ? (
            <Text style={np.keyBackspace}>⌫</Text>
          ) : (
            <Text style={[np.keyText, { color: accentColor }]}>{k}</Text>
          )}
        </TouchableOpacity>
      ))}
    </View>
  );
}

const np = StyleSheet.create({
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    width: 240,
    justifyContent: 'center',
    gap: 12,
  },
  key: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: 'rgba(56,189,248,0.08)',
    borderWidth: 1,
    borderColor: 'rgba(56,189,248,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  keyText: {
    fontSize: 24,
    fontWeight: '600',
  },
  keyBackspace: {
    fontSize: 20,
    color: '#94A3B8',
  },
});

// ── Main Component ────────────────────────────────────────────────────────────

const PIN_LENGTH = 4;

export function AdminPinGate({
  visible,
  onSuccess,
  onCancel,
  mode: modeProp,
  accentColor = '#38BDF8',
  bgColor = '#080E1C',
  textColor = '#F1F5F9',
}: AdminPinGateProps) {
  const [mode, setMode] = useState<PinGateMode>(modeProp ?? 'verify');
  const [pin, setPin] = useState('');
  const [confirmPin, setConfirmPin] = useState('');
  const [isConfirming, setIsConfirming] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const shakeAnim = useRef(new Animated.Value(0)).current;

  // Auto-detect mode when modal opens
  useEffect(() => {
    if (visible) {
      setPin('');
      setConfirmPin('');
      setIsConfirming(false);
      setError('');

      if (!modeProp) {
        isPinConfigured().then(configured => {
          setMode(configured ? 'verify' : 'setup');
        });
      }
    }
  }, [visible, modeProp]);

  function shake() {
    shakeAnim.setValue(0);
    Animated.sequence([
      Animated.timing(shakeAnim, { toValue: 10, duration: 50, useNativeDriver: true }),
      Animated.timing(shakeAnim, { toValue: -10, duration: 50, useNativeDriver: true }),
      Animated.timing(shakeAnim, { toValue: 8, duration: 50, useNativeDriver: true }),
      Animated.timing(shakeAnim, { toValue: -8, duration: 50, useNativeDriver: true }),
      Animated.timing(shakeAnim, { toValue: 0, duration: 50, useNativeDriver: true }),
    ]).start();
    Vibration.vibrate(Platform.OS === 'android' ? [0, 50, 50, 50] : 50);
  }

  async function handleKey(key: string) {
    if (loading) return;
    setError('');

    if (key === '⌫') {
      if (isConfirming) {
        setConfirmPin(p => p.slice(0, -1));
      } else {
        setPin(p => p.slice(0, -1));
      }
      return;
    }

    const current = isConfirming ? confirmPin : pin;
    if (current.length >= PIN_LENGTH) return;

    const newVal = current + key;

    if (isConfirming) {
      setConfirmPin(newVal);
    } else {
      setPin(newVal);
    }

    // Auto-submit when PIN_LENGTH reached
    if (newVal.length === PIN_LENGTH) {
      setTimeout(() => handleComplete(newVal, isConfirming), 80);
    }
  }

  async function handleComplete(value: string, isConfirmStep: boolean) {
    setLoading(true);
    try {
      if (mode === 'setup') {
        if (!isConfirmStep) {
          // Move to confirm step
          setIsConfirming(true);
          setPin(value);
          setLoading(false);
          return;
        }
        // Confirm step — check they match
        if (value !== pin) {
          shake();
          setError('PINs do not match. Try again.');
          setConfirmPin('');
          setIsConfirming(false);
          setPin('');
          setLoading(false);
          return;
        }
        await setAdminPin(value);
        onSuccess();
      } else {
        // Verify mode
        const ok = await verifyAdminPin(value);
        if (ok) {
          onSuccess();
        } else {
          shake();
          setError('Incorrect PIN. Try again.');
          setPin('');
          setLoading(false);
        }
      }
    } catch (e: any) {
      setError(e?.message ?? 'Error. Try again.');
      setPin('');
      setLoading(false);
    }
  }

  const title = mode === 'setup'
    ? (isConfirming ? 'Confirm PIN' : 'Set Admin PIN')
    : 'Admin Verification';

  const subtitle = mode === 'setup'
    ? (isConfirming ? 'Re-enter your 4-digit PIN to confirm' : 'Choose a 4-digit PIN to protect admin actions')
    : 'Enter your 4-digit admin PIN to continue';

  const displayPin = isConfirming ? confirmPin : pin;

  return (
    <Modal
      visible={visible}
      animationType="fade"
      transparent
      statusBarTranslucent
      onRequestClose={onCancel}
    >
      <View style={[st.overlay, { backgroundColor: 'rgba(0,0,0,0.85)' }]}>
        <View style={[st.card, { backgroundColor: bgColor, borderColor: 'rgba(56,189,248,0.2)' }]}>
          {/* Shield icon */}
          <View style={[st.iconWrap, { backgroundColor: 'rgba(56,189,248,0.1)', borderColor: 'rgba(56,189,248,0.2)' }]}>
            <Text style={st.icon}>🛡️</Text>
          </View>

          <Text style={[st.title, { color: textColor }]}>{title}</Text>
          <Text style={[st.subtitle, { color: '#94A3B8' }]}>{subtitle}</Text>

          {/* PIN dots */}
          <PinDots length={PIN_LENGTH} filled={displayPin.length} shake={shakeAnim} />

          {/* Error message */}
          {error ? (
            <Text style={st.errorText}>{error}</Text>
          ) : null}

          {/* Numpad */}
          <NumPad onKey={handleKey} accentColor={accentColor} />

          {/* Cancel */}
          <TouchableOpacity style={st.cancelBtn} onPress={onCancel}>
            <Text style={[st.cancelText, { color: '#94A3B8' }]}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const st = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  card: {
    width: '100%',
    maxWidth: 360,
    borderRadius: 24,
    borderWidth: 1,
    padding: 28,
    alignItems: 'center',
  },
  iconWrap: {
    width: 64,
    height: 64,
    borderRadius: 32,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  icon: { fontSize: 30 },
  title: {
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 13,
    textAlign: 'center',
    marginBottom: 28,
    lineHeight: 18,
  },
  errorText: {
    color: '#EF4444',
    fontSize: 13,
    marginBottom: 12,
    textAlign: 'center',
  },
  cancelBtn: {
    marginTop: 20,
    paddingVertical: 10,
    paddingHorizontal: 24,
  },
  cancelText: {
    fontSize: 15,
    fontWeight: '500',
  },
});
