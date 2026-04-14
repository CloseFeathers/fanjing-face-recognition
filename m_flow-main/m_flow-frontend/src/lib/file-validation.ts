/**
 * File Validation Utilities
 * 
 * Shared validation logic for file uploads across the application.
 */

// ============================================================================
// Constants
// ============================================================================

export const ACCEPTED_FILE_TYPES = [
  ".txt", ".md", ".pdf", 
  ".docx", ".doc", 
  ".json", ".csv", ".xlsx",
  ".html", ".py", ".js", ".ts"
];

export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export const FILE_TYPE_LABELS: Record<string, string> = {
  ".txt": "Plain Text",
  ".md": "Markdown",
  ".pdf": "PDF",
  ".docx": "Word Document",
  ".doc": "Word Document (Legacy)",
  ".json": "JSON",
  ".csv": "CSV",
  ".xlsx": "Excel",
  ".html": "HTML",
  ".py": "Python",
  ".js": "JavaScript",
  ".ts": "TypeScript",
};

// ============================================================================
// Types
// ============================================================================

export interface FileValidationResult {
  valid: File[];
  rejected: FileRejection[];
}

export interface FileRejection {
  file: File;
  reason: string;
  code: "size" | "type" | "empty" | "other";
}

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * Check if a file has a valid type based on extension
 */
export function isValidFileType(file: File): boolean {
  const ext = getFileExtension(file.name);
  return ACCEPTED_FILE_TYPES.includes(ext);
}

/**
 * Extract file extension (lowercase with dot)
 */
export function getFileExtension(filename: string): string {
  const parts = filename.split(".");
  if (parts.length < 2) return "";
  return "." + parts.pop()!.toLowerCase();
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Validate a single file
 */
export function validateFile(file: File): FileRejection | null {
  if (file.size === 0) {
    return { file, reason: "file is empty", code: "empty" };
  }
  
  if (file.size > MAX_FILE_SIZE) {
    return { 
      file, 
      reason: `exceeds ${formatFileSize(MAX_FILE_SIZE)} limit`, 
      code: "size" 
    };
  }
  
  if (!isValidFileType(file)) {
    const ext = getFileExtension(file.name) || "unknown";
    return { 
      file, 
      reason: `unsupported file type (${ext})`, 
      code: "type" 
    };
  }
  
  return null;
}

/**
 * Validate multiple files
 */
export function validateFiles(files: File[]): FileValidationResult {
  const valid: File[] = [];
  const rejected: FileRejection[] = [];

  for (const file of files) {
    const rejection = validateFile(file);
    if (rejection) {
      rejected.push(rejection);
    } else {
      valid.push(file);
    }
  }

  return { valid, rejected };
}

/**
 * Get human-readable list of accepted file types
 */
export function getAcceptedTypesString(): string {
  return ACCEPTED_FILE_TYPES.join(", ");
}

/**
 * Validate dataset name
 */
export function validateDatasetName(name: string): { valid: boolean; error?: string } {
  if (!name.trim()) {
    return { valid: true }; // Empty is valid, will use default
  }
  
  if (name.length > 100) {
    return { valid: false, error: "Dataset name must be 100 characters or less" };
  }
  
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    return { 
      valid: false, 
      error: "Dataset name can only contain letters, numbers, underscores, and hyphens" 
    };
  }
  
  return { valid: true };
}
