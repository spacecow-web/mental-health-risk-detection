export interface ParsedCsv {
  headers: string[];
  rows: Record<string, string>[];
}

/**
 * RFC-4180 compliant CSV parser handling quoted fields, embedded commas/newlines,
 * escaped quotes (""), and UTF-8 BOM without external dependencies.
 */
export function parseCsvContent(rawContent: string): ParsedCsv {
  const content = rawContent.replace(/^\uFEFF/, '');
  const records: string[][] = [];
  let currentField = '';
  let currentRow: string[] = [];
  let inQuotes = false;

  for (let i = 0; i < content.length; i += 1) {
    const ch = content[i];
    const next = content[i + 1];

    if (inQuotes) {
      if (ch === '"' && next === '"') {
        currentField += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        currentField += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      currentRow.push(currentField);
      currentField = '';
    } else if (ch === '\r' && next === '\n') {
      currentRow.push(currentField);
      records.push(currentRow);
      currentRow = [];
      currentField = '';
      i += 1;
    } else if (ch === '\n' || ch === '\r') {
      currentRow.push(currentField);
      records.push(currentRow);
      currentRow = [];
      currentField = '';
    } else {
      currentField += ch;
    }
  }

  if (currentField.length > 0 || currentRow.length > 0) {
    currentRow.push(currentField);
    records.push(currentRow);
  }

  const nonEmptyRecords = records.filter((r) => r.some((cell) => cell.trim() !== ''));
  if (nonEmptyRecords.length < 2) {
    throw new Error('CSV file must include a header row and at least one data row.');
  }

  const headers = nonEmptyRecords[0].map((h, idx) => {
    const clean = h.trim();
    return clean || `column_${idx + 1}`;
  });

  const rows: Record<string, string>[] = [];
  for (let r = 1; r < nonEmptyRecords.length; r += 1) {
    const rowCells = nonEmptyRecords[r];
    const rowObj: Record<string, string> = {};
    for (let c = 0; c < headers.length; c += 1) {
      rowObj[headers[c]] = (rowCells[c] ?? '').trim();
    }
    rows.push(rowObj);
  }

  return { headers, rows };
}
