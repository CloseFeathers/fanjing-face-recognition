"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  FileJson,
  FileCode,
  FileSpreadsheet,
  FileImage,
  FileArchive,
  File,
  FileType,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Sparkles,
} from "lucide-react";

// ============================================================================
// File Type Icons
// ============================================================================

const FILE_ICON_MAP: Record<string, { icon: React.ElementType; color: string }> = {
  // Documents
  pdf: { icon: FileText, color: "text-red-400" },
  doc: { icon: FileText, color: "text-blue-400" },
  docx: { icon: FileText, color: "text-blue-400" },
  txt: { icon: FileText, color: "text-[var(--text-muted)]" },
  md: { icon: FileText, color: "text-[var(--text-secondary)]" },
  rtf: { icon: FileText, color: "text-[var(--text-muted)]" },
  
  // Data
  json: { icon: FileJson, color: "text-yellow-400" },
  csv: { icon: FileSpreadsheet, color: "text-green-400" },
  xlsx: { icon: FileSpreadsheet, color: "text-green-500" },
  xls: { icon: FileSpreadsheet, color: "text-green-500" },
  
  // Code
  js: { icon: FileCode, color: "text-yellow-300" },
  ts: { icon: FileCode, color: "text-blue-400" },
  jsx: { icon: FileCode, color: "text-cyan-400" },
  tsx: { icon: FileCode, color: "text-blue-400" },
  py: { icon: FileCode, color: "text-yellow-400" },
  html: { icon: FileCode, color: "text-orange-400" },
  css: { icon: FileCode, color: "text-blue-300" },
  
  // Images
  png: { icon: FileImage, color: "text-purple-400" },
  jpg: { icon: FileImage, color: "text-purple-400" },
  jpeg: { icon: FileImage, color: "text-purple-400" },
  gif: { icon: FileImage, color: "text-purple-400" },
  svg: { icon: FileImage, color: "text-orange-400" },
  webp: { icon: FileImage, color: "text-purple-400" },
  
  // Archives
  zip: { icon: FileArchive, color: "text-amber-400" },
  rar: { icon: FileArchive, color: "text-amber-400" },
  tar: { icon: FileArchive, color: "text-amber-400" },
  gz: { icon: FileArchive, color: "text-amber-400" },
  
  // Default
  default: { icon: File, color: "text-[var(--text-muted)]" },
};

export interface FileTypeIconProps {
  filename: string;
  size?: number;
  className?: string;
}

export function FileTypeIcon({ filename, size = 16, className }: FileTypeIconProps) {
  const extension = filename.split(".").pop()?.toLowerCase() || "";
  const config = FILE_ICON_MAP[extension] || FILE_ICON_MAP.default;
  const Icon = config.icon;

  return (
    <Icon 
      size={size} 
      className={cn(config.color, className)} 
    />
  );
}

// ============================================================================
// Animated Progress Bar
// ============================================================================

export interface AnimatedProgressProps {
  progress?: number;
  indeterminate?: boolean;
  size?: "sm" | "md" | "lg";
  variant?: "default" | "success" | "error" | "warning";
  showLabel?: boolean;
  className?: string;
}

export function AnimatedProgress({
  progress = 0,
  indeterminate = false,
  size = "md",
  variant = "default",
  showLabel = false,
  className,
}: AnimatedProgressProps) {
  const sizeClasses = {
    sm: "h-1",
    md: "h-2",
    lg: "h-3",
  };

  const variantClasses = {
    default: "bg-blue-500",
    success: "bg-green-500",
    error: "bg-red-500",
    warning: "bg-yellow-500",
  };

  return (
    <div className={cn("relative", className)}>
      {showLabel && (
        <div className="flex justify-between mb-1">
          <span className="text-[10px] text-[var(--text-muted)]">Progress</span>
          <span className="text-[10px] text-[var(--text-secondary)]">
            {indeterminate ? "Processing..." : `${Math.round(progress)}%`}
          </span>
        </div>
      )}
      <div className={cn(
        "w-full bg-[var(--bg-elevated)] rounded-full overflow-hidden",
        sizeClasses[size]
      )}>
        {indeterminate ? (
          <motion.div
            className={cn("h-full rounded-full", variantClasses[variant])}
            initial={{ x: "-100%", width: "30%" }}
            animate={{ x: "400%" }}
            transition={{
              repeat: Infinity,
              duration: 1.5,
              ease: "easeInOut",
            }}
          />
        ) : (
          <motion.div
            className={cn("h-full rounded-full", variantClasses[variant])}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          />
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Success Animation
// ============================================================================

export interface SuccessAnimationProps {
  show: boolean;
  message?: string;
  onComplete?: () => void;
  className?: string;
}

export function SuccessAnimation({ 
  show, 
  message = "Success!",
  onComplete,
  className 
}: SuccessAnimationProps) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          onAnimationComplete={() => {
            if (onComplete) {
              setTimeout(onComplete, 1500);
            }
          }}
          className={cn(
            "flex flex-col items-center justify-center gap-3 p-6",
            className
          )}
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ 
              type: "spring", 
              stiffness: 200, 
              damping: 15,
              delay: 0.1 
            }}
            className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.3 }}
            >
              <CheckCircle2 size={32} className="text-green-400" />
            </motion.div>
          </motion.div>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="text-sm font-medium text-[var(--text-primary)]"
          >
            {message}
          </motion.p>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex gap-1"
          >
            {[...Array(3)].map((_, i) => (
              <motion.div
                key={i}
                initial={{ scale: 0 }}
                animate={{ scale: [0, 1, 0] }}
                transition={{
                  delay: 0.6 + i * 0.1,
                  duration: 0.5,
                  repeat: 1,
                }}
              >
                <Sparkles size={14} className="text-green-400" />
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ============================================================================
// Error Animation
// ============================================================================

export interface ErrorAnimationProps {
  show: boolean;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorAnimation({ 
  show, 
  message = "Something went wrong",
  onRetry,
  className 
}: ErrorAnimationProps) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className={cn(
            "flex flex-col items-center justify-center gap-3 p-6",
            className
          )}
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 15 }}
            className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center"
          >
            <motion.div
              initial={{ rotate: 0 }}
              animate={{ rotate: [0, -10, 10, -10, 10, 0] }}
              transition={{ delay: 0.3, duration: 0.5 }}
            >
              <XCircle size={32} className="text-red-400" />
            </motion.div>
          </motion.div>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-sm font-medium text-[var(--text-primary)]"
          >
            {message}
          </motion.p>
          {onRetry && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              onClick={onRetry}
              className="px-4 py-2 text-xs font-medium text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg hover:bg-red-500/20 transition-colors"
            >
              Try Again
            </motion.button>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ============================================================================
// Warning Banner
// ============================================================================

export interface WarningBannerProps {
  message: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function WarningBanner({ message, action, className }: WarningBannerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn(
        "flex items-center gap-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg",
        className
      )}
    >
      <AlertTriangle size={16} className="text-yellow-400 flex-shrink-0" />
      <p className="flex-1 text-xs text-yellow-200">{message}</p>
      {action && (
        <button
          onClick={action.onClick}
          className="px-3 py-1 text-[10px] font-medium text-yellow-400 bg-yellow-500/10 rounded hover:bg-yellow-500/20 transition-colors"
        >
          {action.label}
        </button>
      )}
    </motion.div>
  );
}

// ============================================================================
// Drop Zone with Enhanced Visual Feedback
// ============================================================================

export interface DropZoneProps {
  isDragging: boolean;
  isLoading?: boolean;
  accept?: string;
  maxSize?: string;
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onClick?: () => void;
  children?: React.ReactNode;
  className?: string;
}

export function DropZone({
  isDragging,
  isLoading = false,
  accept,
  maxSize,
  onDrop,
  onDragOver,
  onDragLeave,
  onClick,
  children,
  className,
}: DropZoneProps) {
  return (
    <motion.div
      animate={{
        scale: isDragging ? 1.02 : 1,
        borderColor: isDragging 
          ? "var(--text-primary)" 
          : "var(--border-subtle)",
      }}
      transition={{ duration: 0.15 }}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={onClick}
      className={cn(
        "relative flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-lg transition-colors cursor-pointer",
        isDragging 
          ? "bg-[var(--text-primary)]/5 border-[var(--text-primary)]" 
          : "bg-[var(--bg-surface)] border-[var(--border-subtle)] hover:border-[var(--text-muted)]",
        isLoading && "pointer-events-none opacity-50",
        className
      )}
      role="button"
      tabIndex={0}
      aria-label="Drop files here or click to browse"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick?.();
        }
      }}
    >
      {/* Animated border glow when dragging */}
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-blue-500/10"
            style={{
              backgroundSize: "200% 100%",
              animation: "shimmer 2s ease-in-out infinite",
            }}
          />
        )}
      </AnimatePresence>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center gap-2">
        {children || (
          <>
            <motion.div
              animate={{ y: isDragging ? -4 : 0 }}
              transition={{ duration: 0.15 }}
            >
              <FileType 
                size={24} 
                className={cn(
                  "transition-colors",
                  isDragging ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]"
                )} 
              />
            </motion.div>
            <p className="text-xs text-[var(--text-secondary)]">
              {isDragging ? "Drop files here" : "Drop files here or click to browse"}
            </p>
            {(accept || maxSize) && (
              <p className="text-[10px] text-[var(--text-muted)]">
                {accept && `Accepted: ${accept}`}
                {accept && maxSize && " · "}
                {maxSize && `Max: ${maxSize}`}
              </p>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}

// ============================================================================
// Shimmer Animation (CSS)
// ============================================================================

const shimmerKeyframes = `
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
`;

if (typeof document !== "undefined") {
  const style = document.createElement("style");
  style.textContent = shimmerKeyframes;
  document.head.appendChild(style);
}

// ============================================================================
// Loading Dots
// ============================================================================

export function LoadingDots({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1", className)}>
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="w-1.5 h-1.5 bg-current rounded-full"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{
            duration: 1,
            repeat: Infinity,
            delay: i * 0.2,
          }}
        />
      ))}
    </span>
  );
}

// ============================================================================
// Pulse Ring
// ============================================================================

export function PulseRing({ 
  active = true,
  size = 8,
  color = "bg-blue-500",
  className 
}: { 
  active?: boolean;
  size?: number;
  color?: string;
  className?: string;
}) {
  return (
    <span className={cn("relative inline-flex", className)}>
      <span 
        className={cn("rounded-full", color)}
        style={{ width: size, height: size }}
      />
      {active && (
        <motion.span
          className={cn("absolute inset-0 rounded-full opacity-75", color)}
          animate={{ scale: [1, 2], opacity: [0.75, 0] }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeOut",
          }}
        />
      )}
    </span>
  );
}
