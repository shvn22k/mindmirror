// ─── Session store (in-memory, persisted to localStorage) ────────────────────

export type LoadLevel = 'Low' | 'Medium' | 'High';

export interface SessionSignal {
  label: string;
  value: string;
}

export interface Session {
  id: string;
  date: string; // ISO string
  durationSeconds: number;
  level: LoadLevel;
  score: number;
  confidence: number;
  headline: string;
  signals: SessionSignal[];
}

const STORAGE_KEY = 'mindmirror_sessions';

export function getSessions(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session[]) : [];
  } catch {
    return [];
  }
}

export function saveSession(session: Session): void {
  const sessions = getSessions();
  sessions.unshift(session);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

export function getSession(id: string): Session | undefined {
  return getSessions().find(s => s.id === id);
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m === 0) return `${s}s`;
  return s === 0 ? `${m} min` : `${m} min ${s}s`;
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'long', day: 'numeric' });
}

export function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

// ─── Simulate a result based on session duration ──────────────────────────────
export function generateResult(durationSeconds: number): Omit<Session, 'id' | 'date' | 'durationSeconds'> {
  const levels: LoadLevel[] = ['Low', 'Medium', 'High'];
  const weights = [0.35, 0.45, 0.20];
  const rand = Math.random();
  let level: LoadLevel = 'Low';
  let cumulative = 0;
  for (let i = 0; i < levels.length; i++) {
    cumulative += weights[i];
    if (rand < cumulative) { level = levels[i]; break; }
  }

  const scoreMap: Record<LoadLevel, [number, number]> = {
    Low: [18, 38],
    Medium: [42, 64],
    High: [68, 88],
  };
  const [min, max] = scoreMap[level];
  const score = Math.round(min + Math.random() * (max - min));
  const confidence = Math.round(70 + Math.random() * 20);

  const headlines: Record<LoadLevel, string> = {
    Low: 'It looks like your mind was fairly calm during this session.',
    Medium: 'It looks like your mind was working a bit harder during this session.',
    High: 'You might have experienced some elevated mental effort during this session.',
  };

  const signalSets: Record<LoadLevel, SessionSignal[]> = {
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

  return { level, score, confidence, headline: headlines[level], signals: signalSets[level] };
}
