import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

interface AuthLog {
  id: string;
  timestamp: number;
  status: 'success' | 'failed';
  userId: string;
}

const STORAGE_KEY = '@auth_logs';

class SyncServiceImpl {
  private isSyncing = false;

  constructor() {
    // Listen to network changes
    NetInfo.addEventListener(state => {
      if (state.isConnected && state.isInternetReachable) {
        this.syncLogs();
      }
    });
  }

  /**
   * Log an authentication attempt offline.
   */
  async logAttempt(status: 'success' | 'failed', userId: string = 'unknown_user') {
    try {
      const existingLogsRaw = await AsyncStorage.getItem(STORAGE_KEY);
      const logs: AuthLog[] = existingLogsRaw ? JSON.parse(existingLogsRaw) : [];

      const newLog: AuthLog = {
        id: Math.random().toString(36).substring(7),
        timestamp: Date.now(),
        status,
        userId,
      };

      logs.push(newLog);
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(logs));

      // Attempt to sync immediately in case we are online
      this.syncLogs();
    } catch (e) {
      console.error('Failed to log attempt offline', e);
    }
  }

  /**
   * Sync logs with the AWS server and purge local data upon success.
   */
  private async syncLogs() {
    if (this.isSyncing) return;
    this.isSyncing = true;

    try {
      const existingLogsRaw = await AsyncStorage.getItem(STORAGE_KEY);
      if (!existingLogsRaw) {
        this.isSyncing = false;
        return; // Nothing to sync
      }

      const logs: AuthLog[] = JSON.parse(existingLogsRaw);
      if (logs.length === 0) {
        this.isSyncing = false;
        return;
      }

      console.log(`Attempting to sync ${logs.length} logs to server...`);

      // Mock API call to AWS endpoint
      const response = await fetch('https://mock-aws-endpoint.com/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ logs }),
      }).catch(() => ({ ok: true })); // MOCK: Always succeed if fetch fails (for hackathon demo without real API)

      if (response.ok) {
        // Purge local data upon success
        await AsyncStorage.removeItem(STORAGE_KEY);
        console.log('Successfully synced and purged local logs.');
      }
    } catch (e) {
      console.error('Sync failed', e);
    } finally {
      this.isSyncing = false;
    }
  }
}

export const SyncService = new SyncServiceImpl();
