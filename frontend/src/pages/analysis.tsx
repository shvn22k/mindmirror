import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronDown, ChevronUp, Download, AlertCircle } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import {
  type Session,
  type LoadLevel,
  saveSession,
  formatDuration,
} from '@/lib/sessions';
import { predictVideo } from '@/lib/inference-api';

// ─── Types ────────────────────────────────────────────────────────────────────
type Phase = 'permission' | 'session' | 'processing' | 'result' | 'error';

// ─── Constants ────────────────────────────────────────────────────────────────

const LOAD_COLORS: Record<LoadLevel, string> = {
  Low: '#8F9D8F',
  Medium: '#9A8CB1',
  High: '#b18c8c',
};

const LOAD_BG: Record<LoadLevel, string> = {
  Low: 'rgba(143,157,143,0.14)',
  Medium: 'rgba(154,140,177,0.14)',
  High: 'rgba(177,140,140,0.14)',
};

const LABEL_MAP: Record<'LOW' | 'MEDIUM' | 'HIGH', LoadLevel> = {
  LOW: 'Low',
  MEDIUM: 'Medium',
  HIGH: 'High',
};

function normalizeClassificationLabel(raw: string): keyof typeof LABEL_MAP {
  const u = raw.trim().toUpperCase();
  if (u === 'LOW' || u === 'MEDIUM' || u === 'HIGH') return u;
  return 'MEDIUM';
}

const HEADLINES: Record<LoadLevel, string> = {
  Low: 'Your mind seemed fairly at ease during this session.',
  Medium: 'It looks like your mind was working a bit harder during this session.',
  High: 'You may have been experiencing a higher mental load during this session.',
};

const SIGNALS: Record<LoadLevel, Array<{ label: string; value: string }>> = {
  Low: [
    { label: 'Eye activity', value: 'Steady and relaxed' },
    { label: 'Focus shifts', value: 'Minimal — you stayed present' },
    { label: 'Micro-expressions', value: 'Calm throughout' },
  ],
  Medium: [
    { label: 'Eye activity', value: 'Slight increase noticed' },
    { label: 'Focus shifts', value: 'Frequent shifts detected' },
    { label: 'Micro-expressions', value: 'Mild tension at times' },
  ],
  High: [
    { label: 'Eye activity', value: 'Notably elevated' },
    { label: 'Focus shifts', value: 'Consistent drift observed' },
    { label: 'Micro-expressions', value: 'Elevated tension patterns' },
  ],
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function pad(n: number) {
  return String(n).padStart(2, '0');
}

function formatTimer(s: number) {
  return `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
}

// ─── useSession hook — webcam + MediaRecorder ─────────────────────────────────
function useSession() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [cameraGranted, setCameraGranted] = useState<boolean | null>(null);

  const startCamera = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } },
        audio: false,
      });
      streamRef.current = stream;
      // Attach to video element if it's already mounted; if not, attachStream()
      // will be called by SessionScreen's useEffect once it mounts.
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraGranted(true);
      return true;
    } catch {
      setCameraGranted(false);
      return false;
    }
  }, []);

  // Called by SessionScreen after it mounts to guarantee the stream is attached
  const attachStream = useCallback(() => {
    if (videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, []);

  const startRecording = useCallback(() => {
    if (!streamRef.current) return;
    chunksRef.current = [];

    // Pick a supported MIME type
    const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']
      .find(t => MediaRecorder.isTypeSupported(t)) ?? '';

    const recorder = new MediaRecorder(streamRef.current, mimeType ? { mimeType } : undefined);
    recorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.start(1000); // collect chunks every second
    recorderRef.current = recorder;
  }, []);

  const stopRecording = useCallback((): Promise<Blob> => {
    return new Promise(resolve => {
      const recorder = recorderRef.current;
      if (!recorder || recorder.state === 'inactive') {
        resolve(new Blob(chunksRef.current, { type: 'video/webm' }));
        return;
      }
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || 'video/webm';
        resolve(new Blob(chunksRef.current, { type: mimeType }));
      };
      recorder.stop();
    });
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  return { videoRef, cameraGranted, startCamera, attachStream, startRecording, stopRecording, stopCamera };
}

// ─── Breathing circle ─────────────────────────────────────────────────────────
function BreathingCircle() {
  return (
    <div className="relative flex items-center justify-center" style={{ width: 80, height: 80 }}>
      <motion.div
        className="absolute rounded-full"
        style={{ width: 80, height: 80, background: 'rgba(154,140,177,0.12)' }}
        animate={{ scale: [1, 1.35, 1], opacity: [0.6, 0.15, 0.6] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute rounded-full"
        style={{ width: 52, height: 52, background: 'rgba(154,140,177,0.18)' }}
        animate={{ scale: [1, 1.18, 1], opacity: [0.8, 0.3, 0.8] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut', delay: 0.3 }}
      />
      <div className="rounded-full" style={{ width: 28, height: 28, background: '#9A8CB1', opacity: 0.85 }} />
    </div>
  );
}

// ─── Pulsing dot ──────────────────────────────────────────────────────────────
function PulseDot({ color = '#9A8CB1', size = 10 }: { color?: string; size?: number }) {
  return (
    <motion.div
      className="rounded-full flex-shrink-0"
      style={{ width: size, height: size, background: color }}
      animate={{ opacity: [0.35, 1, 0.35], scale: [0.88, 1.1, 0.88] }}
      transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
    />
  );
}

// ─── Permission Screen ────────────────────────────────────────────────────────
function PermissionScreen({ onStart, loading }: { onStart: () => void; loading: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.55, ease: 'easeOut' }}
      className="flex flex-col items-center text-center gap-7 max-w-sm mx-auto py-20"
    >
      <BreathingCircle />

      <div>
        <h1
          className="text-2xl font-light mb-3"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif', letterSpacing: '-0.01em' }}
        >
          Ready when you are
        </h1>
        <p className="text-sm leading-relaxed" style={{ color: '#6C757D', lineHeight: '1.85' }}>
          MindMirror will quietly observe patterns while you work. Your session video is sent for
          analysis only — it is never stored.
        </p>
      </div>

      <div
        className="w-full p-5 rounded-2xl flex flex-col gap-3"
        style={{ background: 'rgba(154,140,177,0.08)' }}
      >
        {[
          'Camera is used only for pattern observation',
          'Video is analysed and immediately discarded',
          'This is not a medical assessment',
        ].map(item => (
          <div key={item} className="flex items-start gap-2.5">
            <div className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0" style={{ background: '#8F9D8F' }} />
            <span className="text-xs text-left" style={{ color: '#6C757D', lineHeight: '1.7' }}>
              {item}
            </span>
          </div>
        ))}
      </div>

      <button
        onClick={onStart}
        disabled={loading}
        className="w-full py-4 rounded-full text-white text-sm font-medium transition-colors duration-200 disabled:opacity-60"
        style={{ background: '#9A8CB1' }}
        onMouseEnter={e => !loading && (e.currentTarget.style.background = '#7d6e9a')}
        onMouseLeave={e => (e.currentTarget.style.background = '#9A8CB1')}
      >
        {loading ? 'Starting camera…' : 'Start session'}
      </button>

      <Link
        to="/"
        className="text-sm transition-colors duration-200"
        style={{ color: '#6C757D' }}
        onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = '#4A405A'}
        onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = '#6C757D'}
      >
        Not now
      </Link>
    </motion.div>
  );
}

// ─── Active Session Screen ────────────────────────────────────────────────────
function SessionScreen({
  elapsed,
  onEnd,
  videoRef,
  attachStream,
}: {
  elapsed: number;
  onEnd: () => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  attachStream: () => void;
}) {
  // Attach the stream once this component mounts — the video element now exists
  useEffect(() => {
    attachStream();
  }, [attachStream]);
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.55, ease: 'easeOut' }}
      className="flex flex-col items-center gap-8 max-w-2xl mx-auto py-10"
    >
      {/* Webcam preview */}
      <div className="relative w-full max-w-md">
        <div
          className="relative w-full rounded-3xl overflow-hidden"
          style={{ background: '#1a1820', aspectRatio: '4/3' }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
          />
          {/* Subtle scan line */}
          <motion.div
            className="absolute left-0 right-0 h-px pointer-events-none"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(154,140,177,0.5), transparent)' }}
            animate={{ top: ['8%', '92%', '8%'] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          />
          {/* Corner brackets */}
          {[
            { top: 14, left: 14, bTop: true, bLeft: true },
            { top: 14, right: 14, bTop: true, bRight: true },
            { bottom: 14, right: 14, bBottom: true, bRight: true },
            { bottom: 14, left: 14, bBottom: true, bLeft: true },
          ].map((pos, i) => (
            <div
              key={i}
              className="absolute w-5 h-5 pointer-events-none"
              style={{
                top: pos.top, left: pos.left, right: pos.right, bottom: pos.bottom,
                borderTop: pos.bTop ? '1.5px solid rgba(154,140,177,0.6)' : 'none',
                borderBottom: pos.bBottom ? '1.5px solid rgba(154,140,177,0.6)' : 'none',
                borderLeft: pos.bLeft ? '1.5px solid rgba(154,140,177,0.6)' : 'none',
                borderRight: pos.bRight ? '1.5px solid rgba(154,140,177,0.6)' : 'none',
              }}
            />
          ))}
          {/* Recording badge */}
          <div
            className="absolute bottom-3 left-3 flex items-center gap-2 px-3 py-1.5 rounded-full"
            style={{ background: 'rgba(26,24,32,0.7)', backdropFilter: 'blur(8px)' }}
          >
            <PulseDot color="#b18c8c" size={7} />
            <span className="text-xs font-medium" style={{ color: 'rgba(255,255,255,0.8)', letterSpacing: '0.04em' }}>
              Recording
            </span>
          </div>
        </div>
      </div>

      {/* Info panel */}
      <div className="flex flex-col items-center text-center gap-5 max-w-sm">
        <div>
          <h1
            className="text-2xl font-light mb-2"
            style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif', letterSpacing: '-0.01em' }}
          >
            Your session has started
          </h1>
          <p className="text-sm" style={{ color: '#6C757D', lineHeight: '1.85' }}>
            You can go about your work. MindMirror will quietly observe patterns and reflect back
            when you're done.
          </p>
        </div>

        {/* Timer */}
        <div
          className="flex flex-col items-center gap-1 px-10 py-4 rounded-2xl"
          style={{ background: 'rgba(154,140,177,0.09)' }}
        >
          <span className="text-xs uppercase tracking-widest" style={{ color: '#9A8CB1' }}>
            Session running
          </span>
          <span
            className="text-3xl font-light tabular-nums"
            style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif', letterSpacing: '0.04em' }}
          >
            {formatTimer(elapsed)}
          </span>
        </div>

        {/* Monitoring indicator */}
        <div className="flex items-center gap-2.5">
          <PulseDot color="#8F9D8F" size={8} />
          <span className="text-xs" style={{ color: '#6C757D' }}>
            Monitoring gently in the background
          </span>
        </div>

        <p className="text-sm font-light italic" style={{ color: '#9A8CB1' }}>
          "Take your time. There's nothing you need to do."
        </p>

        <div className="flex flex-col gap-2.5 w-full">
          <button
            onClick={onEnd}
            className="w-full py-4 rounded-full text-white text-sm font-medium transition-colors duration-200"
            style={{ background: '#9A8CB1' }}
            onMouseEnter={e => (e.currentTarget.style.background = '#7d6e9a')}
            onMouseLeave={e => (e.currentTarget.style.background = '#9A8CB1')}
          >
            End session
          </button>
          <p className="text-xs" style={{ color: '#6C757D' }}>
            No data is stored. This session is private.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Processing Screen ────────────────────────────────────────────────────────
function ProcessingScreen() {
  const steps = ['Processing your session…', 'Reflecting on your patterns…', 'Almost ready…'];
  const [step, setStep] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setStep(1), 1400);
    const t2 = setTimeout(() => setStep(2), 3000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col items-center text-center gap-8 max-w-sm mx-auto py-24"
    >
      <BreathingCircle />
      <div>
        <AnimatePresence mode="wait">
          <motion.h2
            key={step}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.4 }}
            className="text-xl font-light mb-2"
            style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
          >
            {steps[step]}
          </motion.h2>
        </AnimatePresence>
        <p className="text-sm" style={{ color: '#6C757D' }}>
          Just a moment while we reflect on your patterns
        </p>
      </div>
      <div className="flex gap-2">
        {[0, 0.4, 0.8].map((delay, i) => (
          <motion.div
            key={i}
            className="rounded-full"
            style={{ width: 7, height: 7, background: '#9A8CB1' }}
            animate={{ opacity: [0.2, 0.9, 0.2] }}
            transition={{ duration: 1.6, delay, repeat: Infinity, ease: 'easeInOut' }}
          />
        ))}
      </div>
    </motion.div>
  );
}

// ─── Result Screen ────────────────────────────────────────────────────────────
function ResultScreen({ session }: { session: Session }) {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();
  const color = LOAD_COLORS[session.level];
  const bg = LOAD_BG[session.level];

  const handleDownload = () => {
    const lines = [
      'MindMirror Session Report',
      '─────────────────────────',
      `Date:           ${new Date(session.date).toLocaleString()}`,
      `Duration:       ${formatDuration(session.durationSeconds)}`,
      `Cognitive Load: ${session.level}`,
      `Score:          ${session.score.toFixed(2)}`,
      '',
      session.headline,
      '',
      'Observed signals:',
      ...session.signals.map(s => `  • ${s.label}: ${s.value}`),
      '',
      'Note: These insights are not a diagnosis.',
      'MindMirror is not a medical device.',
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mindmirror-${session.id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="max-w-md mx-auto py-12 flex flex-col gap-6"
    >
      <div className="text-center">
        <p className="text-xs font-medium uppercase tracking-widest mb-4" style={{ color: '#9A8CB1' }}>
          Session complete
        </p>
        <h1
          className="text-2xl font-light leading-snug"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif', letterSpacing: '-0.01em' }}
        >
          Here's what we noticed
        </h1>
      </div>

      {/* Main insight card */}
      <div
        className="rounded-3xl p-8"
        style={{
          background: 'rgba(255,255,255,0.72)',
          boxShadow: '0 6px 32px rgba(74,64,90,0.09)',
          backdropFilter: 'blur(12px)',
        }}
      >
        <p
          className="text-lg font-light leading-relaxed mb-6"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif', lineHeight: '1.7' }}
        >
          {session.headline}
        </p>

        {/* Stats row */}
        <div className="flex gap-3 mb-6">
          <div className="flex-1 flex flex-col gap-1 p-4 rounded-2xl" style={{ background: bg }}>
            <span className="text-xs" style={{ color: '#6C757D' }}>Cognitive load</span>
            <span className="text-sm font-medium" style={{ color }}>{session.level}</span>
          </div>
          <div className="flex-1 flex flex-col gap-1 p-4 rounded-2xl" style={{ background: 'rgba(74,64,90,0.05)' }}>
            <span className="text-xs" style={{ color: '#6C757D' }}>Duration</span>
            <span className="text-sm font-medium" style={{ color: '#4A405A' }}>
              {formatDuration(session.durationSeconds)}
            </span>
          </div>
          <div className="flex-1 flex flex-col gap-1 p-4 rounded-2xl" style={{ background: 'rgba(74,64,90,0.05)' }}>
            <span className="text-xs" style={{ color: '#6C757D' }}>Score</span>
            <span className="text-sm font-medium" style={{ color: '#4A405A' }}>
              {session.score.toFixed(1)}
            </span>
          </div>
        </div>

        {/* Load bar */}
        <div className="mb-6">
          <div className="w-full h-1.5 rounded-full" style={{ background: 'rgba(74,64,90,0.09)' }}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(session.score, 100)}%` }}
              transition={{ duration: 1.2, delay: 0.3, ease: 'easeOut' }}
              className="h-full rounded-full"
              style={{ background: color }}
            />
          </div>
        </div>

        {/* Expandable signals */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-sm transition-colors duration-200"
          style={{ color: '#9A8CB1' }}
        >
          {expanded ? <ChevronUp size={14} strokeWidth={1.5} /> : <ChevronDown size={14} strokeWidth={1.5} />}
          {expanded ? 'Hide details' : 'See more details'}
        </button>

        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.35, ease: 'easeOut' }}
              className="overflow-hidden"
            >
              <div className="pt-5">
                {session.signals.map((sig, i) => (
                  <div
                    key={sig.label}
                    className="flex items-center justify-between py-3"
                    style={{
                      borderBottom: i < session.signals.length - 1
                        ? '1px solid rgba(198,210,219,0.4)'
                        : 'none',
                    }}
                  >
                    <span className="text-sm" style={{ color: '#6C757D' }}>{sig.label}</span>
                    <span className="text-sm font-medium" style={{ color: '#4A405A' }}>{sig.value}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <p className="text-xs text-center" style={{ color: '#6C757D', lineHeight: '1.7' }}>
        These insights are not a diagnosis — just gentle reflections to help you stay aware.
      </p>

      <div className="flex flex-col gap-3">
        <button
          onClick={handleDownload}
          className="w-full flex items-center justify-center gap-2 py-4 rounded-full text-sm font-medium transition-colors duration-200"
          style={{ background: '#9A8CB1', color: '#fff' }}
          onMouseEnter={e => (e.currentTarget.style.background = '#7d6e9a')}
          onMouseLeave={e => (e.currentTarget.style.background = '#9A8CB1')}
        >
          <Download size={14} strokeWidth={1.5} />
          Download session report
        </button>
        <button
          onClick={() => navigate('/dashboard')}
          className="w-full py-4 rounded-full text-sm font-medium transition-all duration-200"
          style={{ border: '1.5px solid #9A8CB1', color: '#9A8CB1', background: 'transparent' }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLElement).style.background = '#9A8CB1';
            (e.currentTarget as HTMLElement).style.color = '#fff';
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLElement).style.background = 'transparent';
            (e.currentTarget as HTMLElement).style.color = '#9A8CB1';
          }}
        >
          View dashboard
        </button>
      </div>
    </motion.div>
  );
}

// ─── Error Screen ─────────────────────────────────────────────────────────────
function ErrorScreen({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="flex flex-col items-center text-center gap-6 max-w-sm mx-auto py-20"
    >
      <div
        className="w-14 h-14 rounded-3xl flex items-center justify-center"
        style={{ background: 'rgba(177,140,140,0.14)' }}
      >
        <AlertCircle size={24} style={{ color: '#b18c8c' }} strokeWidth={1.5} />
      </div>
      <div>
        <h2
          className="text-xl font-light mb-2"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
        >
          Something went wrong
        </h2>
        <p className="text-sm leading-relaxed" style={{ color: '#6C757D', lineHeight: '1.8' }}>
          {message}
        </p>
      </div>
      <div className="flex flex-col gap-3 w-full">
        <button
          onClick={onRetry}
          className="w-full py-3.5 rounded-full text-white text-sm font-medium transition-colors duration-200"
          style={{ background: '#9A8CB1' }}
          onMouseEnter={e => (e.currentTarget.style.background = '#7d6e9a')}
          onMouseLeave={e => (e.currentTarget.style.background = '#9A8CB1')}
        >
          Try again
        </button>
        <Link
          to="/"
          className="w-full py-3.5 rounded-full text-sm font-medium text-center transition-colors duration-200"
          style={{ color: '#6C757D' }}
          onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = '#4A405A'}
          onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = '#6C757D'}
        >
          Go home
        </Link>
      </div>
    </motion.div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function AnalysisPage() {
  const [phase, setPhase] = useState<Phase>('permission');
  const [elapsed, setElapsed] = useState(0);
  const [session, setSession] = useState<Session | null>(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [startingCamera, setStartingCamera] = useState(false);

  const { videoRef, startCamera, attachStream, startRecording, stopRecording, stopCamera } = useSession();

  // ── Timer ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (phase !== 'session') return;
    const interval = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(interval);
  }, [phase]);

  // ── Start session ──────────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    setStartingCamera(true);
    const ok = await startCamera();
    setStartingCamera(false);

    if (!ok) {
      setErrorMessage("We couldn't access your camera. Please allow camera access and try again.");
      setPhase('error');
      return;
    }

    setElapsed(0);
    startRecording();
    setPhase('session');
  }, [startCamera, startRecording]);

  // ── End session ────────────────────────────────────────────────────────────
  const handleEnd = useCallback(async () => {
    const duration = elapsed < 3 ? 3 : elapsed;
    setPhase('processing');

    let videoBlob: Blob;
    try {
      videoBlob = await stopRecording();
    } catch {
      stopCamera();
      setErrorMessage('Something went wrong while saving your session. Please try again.');
      setPhase('error');
      return;
    }
    stopCamera();

    try {
      const ext = videoBlob.type.includes('mp4') ? 'mp4' : 'webm';
      const data = await predictVideo(videoBlob, `session.${ext}`);
      const band = normalizeClassificationLabel(data.classification_label);
      const level = LABEL_MAP[band];
      const prob =
        data.probabilities?.[band] ??
        data.probabilities?.[data.classification_label] ??
        0.75;

      const newSession: Session = {
        id: Date.now().toString(),
        date: new Date().toISOString(),
        durationSeconds: duration,
        level,
        score: data.regression_score,
        confidence: Math.round(prob * 100),
        headline: HEADLINES[level],
        signals: SIGNALS[level],
      };

      saveSession(newSession);
      setSession(newSession);
      setPhase('result');
    } catch (err) {
      console.error('Predict API error:', err);
      const msg = err instanceof Error ? err.message : '';
      setErrorMessage(
        msg
          ? `We could not analyse your session: ${msg}`
          : 'Something went wrong while analysing your session. Please try again.',
      );
      setPhase('error');
    }
  }, [elapsed, stopRecording, stopCamera]);

  // ── Reset ──────────────────────────────────────────────────────────────────
  const handleRetry = useCallback(() => {
    setSession(null);
    setErrorMessage('');
    setElapsed(0);
    setPhase('permission');
  }, []);

  return (
    <>
      <title>Live Analysis — MindMirror</title>
      <meta name="description" content="Start a calm cognitive load session with MindMirror." />

      <div
        className="min-h-screen pt-20 pb-16"
        style={{ background: 'linear-gradient(175deg, #e8e4f0 0%, #edf0f4 30%, #F8F4EA 80%)' }}
      >
        {/* Back link — hidden during active session and processing */}
        <AnimatePresence>
          {phase !== 'session' && phase !== 'processing' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="container mx-auto px-6 pt-4"
            >
              <Link
                to="/"
                className="inline-flex items-center gap-2 text-sm transition-colors duration-200"
                style={{ color: '#6C757D' }}
                onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
                onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
              >
                <ChevronDown size={14} style={{ transform: 'rotate(90deg)' }} strokeWidth={1.5} />
                Back to home
              </Link>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="container mx-auto px-6">
          <AnimatePresence mode="wait">
            {phase === 'permission' && (
              <PermissionScreen key="permission" onStart={handleStart} loading={startingCamera} />
            )}
            {phase === 'session' && (
              <SessionScreen key="session" elapsed={elapsed} onEnd={handleEnd} videoRef={videoRef} attachStream={attachStream} />
            )}
            {phase === 'processing' && (
              <ProcessingScreen key="processing" />
            )}
            {phase === 'result' && session && (
              <ResultScreen key="result" session={session} />
            )}
            {phase === 'error' && (
              <ErrorScreen key="error" message={errorMessage} onRetry={handleRetry} />
            )}
          </AnimatePresence>
        </div>
      </div>
    </>
  );
}
