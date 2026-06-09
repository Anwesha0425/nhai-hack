/**
 * CsvExportService.ts
 *
 * Exports the full attendance log as a CSV file and opens
 * the native share sheet so it can be sent via WhatsApp, email, etc.
 *
 * Dependencies: expo-file-system, expo-sharing
 */

import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { AttendanceRecord } from './BiometricStore';

// ── CSV helpers ───────────────────────────────────────────────────────────────

/** Wrap a cell value in quotes and escape inner quotes per RFC 4180 */
function csvCell(value: string | number | undefined | null): string {
  const str = value === undefined || value === null ? '' : String(value);
  // If the value contains comma, newline, or a quote → wrap in quotes
  if (str.includes(',') || str.includes('\n') || str.includes('"')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

function buildCsvString(records: AttendanceRecord[]): string {
  const header = [
    'ID',
    'Name',
    'Employee ID',
    'Date',
    'Time',
    'Location',
    'Latitude',
    'Longitude',
    'Sync Status',
  ].join(',');

  const rows = records.map(r => {
    const d = new Date(r.timestamp);
    const date = d.toLocaleDateString('en-IN');
    const time = d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    return [
      csvCell(r.id),
      csvCell(r.name),
      csvCell(r.userId),
      csvCell(date),
      csvCell(time),
      csvCell(r.location),
      csvCell(r.lat),
      csvCell(r.lng),
      csvCell(r.synced ? 'Synced' : 'Pending'),
    ].join(',');
  });

  return [header, ...rows].join('\r\n');
}

// ── Public API ────────────────────────────────────────────────────────────────

export interface ExportResult {
  success: boolean;
  message: string;
  recordCount?: number;
  filePath?: string;
}

/**
 * Generate a CSV from attendance records and open the share sheet.
 * Returns a result object describing what happened.
 */
export async function exportAttendanceCsv(
  records: AttendanceRecord[]
): Promise<ExportResult> {
  try {
    if (records.length === 0) {
      return { success: false, message: 'No attendance records to export.' };
    }

    // Build CSV content
    const csvContent = buildCsvString(records);

    // Write to a temp file in the app's cache directory
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const fileName = `NHAI_Attendance_${timestamp}.csv`;
    const filePath = `${FileSystem.cacheDirectory}${fileName}`;

    await FileSystem.writeAsStringAsync(filePath, csvContent, {
      encoding: FileSystem.EncodingType.UTF8,
    });

    // Check if sharing is available on this device
    const canShare = await Sharing.isAvailableAsync();
    if (!canShare) {
      return {
        success: false,
        message: 'Sharing is not available on this device.',
        filePath,
      };
    }

    // Open share sheet
    await Sharing.shareAsync(filePath, {
      mimeType: 'text/csv',
      dialogTitle: 'Export Attendance CSV',
      UTI: 'public.comma-separated-values-text',
    });

    return {
      success: true,
      message: `Exported ${records.length} records as ${fileName}`,
      recordCount: records.length,
      filePath,
    };
  } catch (error: any) {
    console.error('[CsvExport] Error:', error);
    return {
      success: false,
      message: `Export failed: ${error?.message ?? 'Unknown error'}`,
    };
  }
}

/**
 * Generate a quick summary CSV with only today's records.
 */
export async function exportTodayAttendanceCsv(
  allRecords: AttendanceRecord[]
): Promise<ExportResult> {
  const todayStr = new Date().toLocaleDateString('en-IN');
  const todayRecords = allRecords.filter(
    r => new Date(r.timestamp).toLocaleDateString('en-IN') === todayStr
  );
  return exportAttendanceCsv(todayRecords);
}
