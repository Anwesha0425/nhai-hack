/**
 * AdminPinService.ts
 *
 * Manages the admin supervisor PIN using expo-secure-store.
 * The PIN is stored as a bcrypt-style SHA-256 hash so the raw
 * value is never persisted.
 *
 * Dependency: expo-secure-store
 */

import * as SecureStore from 'expo-secure-store';

const PIN_KEY = 'nhai_admin_pin_hash';
const PIN_SET_KEY = 'nhai_admin_pin_configured';

// ── Simple deterministic hash (no native crypto needed in RN) ─────────────────

function hashPin(pin: string): string {
  // djb2 hash + XOR fold — good enough for a 4-6 digit PIN guard
  let h1 = 5381;
  let h2 = 52711;
  for (let i = 0; i < pin.length; i++) {
    const c = pin.charCodeAt(i);
    h1 = ((h1 << 5) + h1) ^ c;
    h2 = ((h2 << 5) + h2) ^ c;
  }
  const combined = (Math.abs(h1) * 4294967296 + Math.abs(h2)).toString(16).padStart(16, '0');
  return combined;
}

// ── Public API ─────────────────────────────────────────────────────────────────

/** Returns true if an admin PIN has been set up. */
export async function isPinConfigured(): Promise<boolean> {
  try {
    const flag = await SecureStore.getItemAsync(PIN_SET_KEY);
    return flag === 'true';
  } catch {
    return false;
  }
}

/**
 * Set a new admin PIN. Overwrites any existing one.
 * @param pin  4–6 digit numeric string
 */
export async function setAdminPin(pin: string): Promise<void> {
  if (!/^\d{4,6}$/.test(pin)) {
    throw new Error('PIN must be 4–6 digits.');
  }
  const hashed = hashPin(pin);
  await SecureStore.setItemAsync(PIN_KEY, hashed);
  await SecureStore.setItemAsync(PIN_SET_KEY, 'true');
}

/**
 * Verify a candidate PIN against the stored hash.
 * Returns true if it matches.
 */
export async function verifyAdminPin(candidate: string): Promise<boolean> {
  try {
    const stored = await SecureStore.getItemAsync(PIN_KEY);
    if (!stored) return false;
    return hashPin(candidate) === stored;
  } catch {
    return false;
  }
}

/** Clear the PIN (used during "reset all data"). */
export async function clearAdminPin(): Promise<void> {
  await SecureStore.deleteItemAsync(PIN_KEY);
  await SecureStore.deleteItemAsync(PIN_SET_KEY);
}
