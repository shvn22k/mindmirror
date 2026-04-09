import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronDown, ChevronUp, X, Download } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  type Session,
  type LoadLevel,
  getSessions,
  formatDuration,
  formatDate,
  formatTime,
} from '@/lib/sessions';

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

// ─── Load dot ─────────────────────────────────────────────────────────────────
function LoadDot({ level }: { level: LoadLevel }) {
  return (
    <div
      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
      style={{ background: LOAD_COLORS[level] }}
    />
  );
}

// ─── Session detail drawer ────────────────────────────────────────────────────
function SessionDetail({
  session,
  onClose,
}: {
  session: Session;
  onClose: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const color = LOAD_COLORS[session.level];
  const bg = LOAD_BG[session.level];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4"
      style={{ background: 'rgba(74,64,90,0.25)', backdropFilter: 'blur(6px)' }}
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 24 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="w-full max-w-md rounded-3xl p-8"
        style={{
          background: '#F8F4EA',
          boxShadow: '0 24px 64px rgba(74,64,90,0.18)',
          maxHeight: '90vh',
          overflowY: 'auto',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <p className="text-xs uppercase tracking-widest mb-1" style={{ color: '#9A8CB1' }}>
              Session detail
            </p>
            <h2
              className="text-xl font-light"
              style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
            >
              {formatDate(session.date)}
            </h2>
            <p className="text-xs mt-0.5" style={{ color: '#6C757D' }}>
              {formatTime(session.date)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl transition-colors duration-200"
            style={{ color: '#6C757D' }}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(74,64,90,0.07)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
          >
            <X size={18} strokeWidth={1.5} />
          </button>
        </div>

        {/* Headline */}
        <p
          className="text-base font-light leading-relaxed mb-6"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif', lineHeight: '1.7' }}
        >
          {session.headline}
        </p>

        {/* Stats */}
        <div className="flex gap-3 mb-6">
          <div
            className="flex-1 flex flex-col gap-1 p-4 rounded-2xl"
            style={{ background: bg }}
          >
            <span className="text-xs" style={{ color: '#6C757D' }}>Cognitive load</span>
            <span className="text-sm font-medium" style={{ color }}>
              {session.level}
            </span>
          </div>
          <div
            className="flex-1 flex flex-col gap-1 p-4 rounded-2xl"
            style={{ background: 'rgba(74,64,90,0.05)' }}
          >
            <span className="text-xs" style={{ color: '#6C757D' }}>Duration</span>
            <span className="text-sm font-medium" style={{ color: '#4A405A' }}>
              {formatDuration(session.durationSeconds)}
            </span>
          </div>
          <div
            className="flex-1 flex flex-col gap-1 p-4 rounded-2xl"
            style={{ background: 'rgba(74,64,90,0.05)' }}
          >
            <span className="text-xs" style={{ color: '#6C757D' }}>Score</span>
            <span className="text-sm font-medium" style={{ color: '#4A405A' }}>
              {typeof session.score === 'number' ? session.score.toFixed(1) : '—'}
            </span>
          </div>
        </div>

        {/* Load bar */}
        <div className="mb-6">
          <div className="w-full h-1.5 rounded-full" style={{ background: 'rgba(74,64,90,0.09)' }}>
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${session.score}%`, background: color }}
            />
          </div>
        </div>

        {/* Expandable signals */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-sm mb-1 transition-colors duration-200"
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
              transition={{ duration: 0.3, ease: 'easeOut' }}
              className="overflow-hidden"
            >
              <div className="pt-4 flex flex-col">
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

        <div className="mt-6 pt-5 flex flex-col gap-4" style={{ borderTop: '1px solid rgba(198,210,219,0.4)' }}>
          {/* Download button */}
          <button
            onClick={() => {
              const lines = [
                'MindMirror Session Report',
                '─────────────────────────',
                `Date:           ${new Date(session.date).toLocaleString()}`,
                `Duration:       ${formatDuration(session.durationSeconds)}`,
                `Cognitive Load: ${session.level}`,
                `Score:          ${typeof session.score === 'number' ? session.score.toFixed(2) : '—'}`,
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
            }}
            className="w-full flex items-center justify-center gap-2 py-3.5 rounded-full text-sm font-medium transition-colors duration-200"
            style={{ background: '#9A8CB1', color: '#fff' }}
            onMouseEnter={e => (e.currentTarget.style.background = '#7d6e9a')}
            onMouseLeave={e => (e.currentTarget.style.background = '#9A8CB1')}
          >
            <Download size={13} strokeWidth={1.5} />
            Download report
          </button>
          <p className="text-xs text-center" style={{ color: '#6C757D', lineHeight: '1.7' }}>
            These insights are not a diagnosis — just gentle reflections to help you stay aware.
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}

// ─── Session card ─────────────────────────────────────────────────────────────
function SessionCard({
  session,
  delay,
  onClick,
}: {
  session: Session;
  delay: number;
  onClick: () => void;
}) {
  const color = LOAD_COLORS[session.level];

  return (
    <motion.button
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: 'easeOut' }}
      whileHover={{ y: -2, boxShadow: '0 10px 32px rgba(74,64,90,0.11)' }}
      onClick={onClick}
      className="w-full text-left rounded-3xl p-6 transition-shadow duration-200"
      style={{
        background: '#C6D2DB',
        boxShadow: '0 2px 12px rgba(74,64,90,0.06)',
      }}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-2 min-w-0">
          <div className="flex items-center gap-2">
            <LoadDot level={session.level} />
            <span
              className="text-sm font-medium"
              style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
            >
              {formatDate(session.date)}
            </span>
            <span className="text-xs" style={{ color: '#6C757D' }}>
              · {formatTime(session.date)}
            </span>
          </div>
          <p className="text-sm leading-relaxed line-clamp-2" style={{ color: '#6C757D', lineHeight: '1.65' }}>
            {session.headline}
          </p>
        </div>

        <div className="flex flex-col items-end gap-1.5 flex-shrink-0">
          <span
            className="text-xs px-3 py-1 rounded-full font-medium"
            style={{ background: color, color: '#fff' }}
          >
            {session.level}
          </span>
          <span className="text-xs" style={{ color: '#6C757D' }}>
            {formatDuration(session.durationSeconds)}
          </span>
        </div>
      </div>
    </motion.button>
  );
}

// ─── Empty state ──────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="flex flex-col items-center text-center gap-6 py-20"
    >
      <div
        className="w-16 h-16 rounded-3xl flex items-center justify-center"
        style={{ background: 'rgba(154,140,177,0.12)' }}
      >
        <motion.div
          className="rounded-full"
          style={{ width: 24, height: 24, background: '#9A8CB1', opacity: 0.7 }}
          animate={{ opacity: [0.4, 0.85, 0.4], scale: [0.9, 1.08, 0.9] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>
      <div>
        <h2
          className="text-xl font-light mb-2"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
        >
          No sessions yet
        </h2>
        <p className="text-sm" style={{ color: '#6C757D', lineHeight: '1.8' }}>
          Start a session to begin understanding your patterns
        </p>
      </div>
      <Link
        to="/analysis"
        className="px-8 py-3.5 rounded-full text-white text-sm font-medium transition-colors duration-200"
        style={{ background: '#9A8CB1' }}
        onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = '#7d6e9a'}
        onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = '#9A8CB1'}
      >
        Start your first session
      </Link>
    </motion.div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<Session | null>(null);

  useEffect(() => {
    setSessions(getSessions());
  }, []);

  return (
    <>
      <title>Your Sessions — MindMirror</title>
      <meta name="description" content="A gentle history of your cognitive patterns." />

      <AnimatePresence>
        {selected && (
          <SessionDetail session={selected} onClose={() => setSelected(null)} />
        )}
      </AnimatePresence>

      <div
        className="min-h-screen pt-24 pb-20"
        style={{ background: '#F8F4EA' }}
      >
        <div className="container mx-auto px-6 max-w-2xl">

          {/* Page header */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: 'easeOut' }}
            className="mb-10"
          >
            <Link
              to="/"
              className="inline-flex items-center gap-2 text-sm mb-7 transition-colors duration-200"
              style={{ color: '#6C757D' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
              onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
            >
              <ChevronDown size={14} style={{ transform: 'rotate(90deg)' }} strokeWidth={1.5} />
              Home
            </Link>

            <div className="flex items-end justify-between gap-4">
              <div>
                <h1
                  className="font-light mb-2"
                  style={{
                    color: '#4A405A',
                    fontFamily: 'Lexend, sans-serif',
                    fontSize: 'clamp(1.8rem, 3.5vw, 2.4rem)',
                    letterSpacing: '-0.02em',
                  }}
                >
                  Your sessions
                </h1>
                <p className="text-sm" style={{ color: '#6C757D' }}>
                  A gentle history of your cognitive patterns
                </p>
              </div>

              {sessions.length > 0 && (
                <Link
                  to="/analysis"
                  className="flex-shrink-0 px-5 py-2.5 rounded-full text-white text-sm font-medium transition-colors duration-200"
                  style={{ background: '#9A8CB1' }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = '#7d6e9a'}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = '#9A8CB1'}
                >
                  New session
                </Link>
              )}
            </div>
          </motion.div>

          {/* Summary strip — only if sessions exist */}
          {sessions.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: 0.1, ease: 'easeOut' }}
              className="grid grid-cols-3 gap-3 mb-8"
            >
              {[
                { label: 'Sessions', value: sessions.length },
                {
                  label: 'Most common',
                  value: (() => {
                    const counts: Record<string, number> = {};
                    sessions.forEach(s => { counts[s.level] = (counts[s.level] || 0) + 1; });
                    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—';
                  })(),
                },
                {
                  label: 'Avg. duration',
                  value: formatDuration(
                    Math.round(sessions.reduce((a, s) => a + s.durationSeconds, 0) / sessions.length)
                  ),
                },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="flex flex-col gap-1 p-4 rounded-2xl"
                  style={{ background: 'rgba(198,210,219,0.35)' }}
                >
                  <span className="text-xs" style={{ color: '#6C757D' }}>{label}</span>
                  <span
                    className="text-base font-medium"
                    style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
                  >
                    {value}
                  </span>
                </div>
              ))}
            </motion.div>
          )}

          {/* Disclaimer */}
          {sessions.length > 0 && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.4 }}
              className="text-xs mb-6"
              style={{ color: '#6C757D', lineHeight: '1.7' }}
            >
              These insights are not a diagnosis, just reflections to help you stay aware.
            </motion.p>
          )}

          {/* Session list or empty state */}
          {sessions.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="flex flex-col gap-3">
              {sessions.map((session, i) => (
                <SessionCard
                  key={session.id}
                  session={session}
                  delay={i * 0.06}
                  onClick={() => setSelected(session)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
