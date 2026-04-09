import { motion } from 'motion/react';
import { Eye, Sparkles, Layers, ShieldCheck, VideoOff, Info } from 'lucide-react';
import { Link } from 'react-router-dom';

/** Served from `public/images/hero.png` — stable URL; not caught by SPA rewrite that was blocking `/assets/*` on Vercel. */
const heroImageUrl = `${import.meta.env.BASE_URL}images/hero.png`.replace(/\/{2,}/g, '/');

// ─── Organic Blob ────────────────────────────────────────────────────────────
function Blob({
  color,
  style,
}: {
  color: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className="absolute pointer-events-none select-none"
      style={{
        width: 480,
        height: 480,
        borderRadius: '60% 40% 55% 45% / 45% 55% 40% 60%',
        background: color,
        filter: 'blur(72px)',
        ...style,
      }}
    />
  );
}

// ─── Section fade-in wrapper ─────────────────────────────────────────────────
function FadeIn({
  children,
  delay = 0,
  className = '',
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.55, delay, ease: 'easeOut' }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ─── How It Works Card ───────────────────────────────────────────────────────
function StepCard({
  icon: Icon,
  title,
  description,
  delay,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.55, delay, ease: 'easeOut' }}
      whileHover={{ y: -4, boxShadow: '0 12px 40px rgba(74, 64, 90, 0.12)' }}
      className="flex flex-col gap-5 p-8 rounded-3xl"
      style={{
        background: 'rgba(255,255,255,0.72)',
        boxShadow: '0 4px 24px rgba(74, 64, 90, 0.07)',
        backdropFilter: 'blur(8px)',
        transition: 'box-shadow 0.25s ease, transform 0.25s ease',
      }}
    >
      <div
        className="w-12 h-12 rounded-2xl flex items-center justify-center"
        style={{ background: 'rgba(154, 140, 177, 0.15)' }}
      >
        <Icon size={22} style={{ color: '#9A8CB1' }} strokeWidth={1.5} />
      </div>
      <div>
        <h3
          className="text-lg font-medium mb-2"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
        >
          {title}
        </h3>
        <p className="text-sm leading-relaxed" style={{ color: '#6C757D', lineHeight: '1.75' }}>
          {description}
        </p>
      </div>
    </motion.div>
  );
}

// ─── Trust Point ─────────────────────────────────────────────────────────────
function TrustPoint({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-3">
      <div
        className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0"
        style={{ background: 'rgba(143, 157, 143, 0.2)' }}
      >
        <div className="w-2 h-2 rounded-full" style={{ background: '#8F9D8F' }} />
      </div>
      <span className="text-base" style={{ color: '#4F4F4F', lineHeight: '1.7' }}>
        {text}
      </span>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────
export default function HomePage() {
  return (
    <>
      <title>MindMirror — A gentle way to understand your mental load</title>
      <meta
        name="description"
        content="MindMirror uses subtle behavioral signals to help you gently notice when your mind might be overwhelmed. Calm, private, and non-clinical."
      />

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section
        className="relative overflow-hidden flex items-center justify-center min-h-screen"
        style={{
          paddingTop: '120px',
          paddingBottom: '100px',
        }}
      >
        {/* Photo: src/assets/hero.png (bundled by Vite — commit this file so Vercel gets it) */}
        <img
          src={heroImageUrl}
          alt=""
          decoding="async"
          fetchPriority="high"
          draggable={false}
          className="absolute inset-0 z-0 h-full w-full min-h-[100svh] object-cover pointer-events-none select-none"
          style={{ objectPosition: 'center bottom' }}
        />
        {/* Single soft overlay for headline readability — does not replace the image */}
        <div
          className="absolute inset-0 z-[1] pointer-events-none"
          style={{
            background:
              'linear-gradient(175deg, rgba(232,228,240,0.55) 0%, rgba(237,240,244,0.45) 40%, rgba(248,244,234,0.72) 100%)',
          }}
        />

        <div className="relative z-10 container mx-auto px-6 flex flex-col items-center text-center">
          <motion.div
            initial={{ opacity: 0, y: 32 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className="max-w-2xl"
          >
            {/* Eyebrow */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium mb-8"
              style={{
                background: 'rgba(154, 140, 177, 0.14)',
                color: '#9A8CB1',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
              }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: '#9A8CB1' }}
              />
              Mind Mirror
            </motion.div>

            <h1
              className="font-light leading-tight mb-7"
              style={{
                color: '#4A405A',
                fontFamily: 'Lexend, sans-serif',
                fontSize: 'clamp(2.6rem, 5.5vw, 3.8rem)',
                letterSpacing: '-0.02em',
                lineHeight: '1.18',
              }}
            >
              A gentle way to understand
              <br />
              your mental load
            </h1>

            <p
              className="mx-auto mb-10"
              style={{
                color: '#4F4F4F',
                fontSize: '1.1rem',
                lineHeight: '1.8',
                maxWidth: '520px',
              }}
            >
              MindMirror observes subtle patterns like eye movement and focus to
              help you notice when your mind might be overwhelmed.
            </p>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Link
                  to="/analysis"
                  className="inline-block px-8 py-3.5 rounded-full text-white text-sm font-medium transition-colors duration-200"
                  style={{ background: '#9A8CB1', minWidth: '180px', textAlign: 'center' }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = '#7d6e9a'}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = '#9A8CB1'}
                >
                  Start Live Analysis
                </Link>
              </motion.div>

              <a
                href="#how-it-works"
                className="text-sm font-medium transition-colors duration-200 pb-0.5"
                style={{
                  color: '#6C757D',
                  borderBottom: '1px solid transparent',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.color = '#4A405A';
                  e.currentTarget.style.borderBottomColor = '#4A405A';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.color = '#6C757D';
                  e.currentTarget.style.borderBottomColor = 'transparent';
                }}
              >
                Learn how it works
              </a>
            </div>
          </motion.div>

          {/* Floating preview card — removed */}
        </div>
      </section>

      {/* ── How It Works ──────────────────────────────────────────────────── */}
      <section
        id="how-it-works"
        className="relative overflow-hidden py-28"
        style={{ background: '#F8F4EA' }}
      >
        <Blob
          color="rgba(143, 157, 143, 0.08)"
          style={{ top: '-60px', right: '-100px', width: 360, height: 360 }}
        />

        <div className="container mx-auto px-6">
          <FadeIn className="text-center mb-16">
            <p
              className="text-xs font-medium uppercase tracking-widest mb-4"
              style={{ color: '#9A8CB1' }}
            >
              The process
            </p>
            <h2
              className="font-light"
              style={{
                color: '#4A405A',
                fontFamily: 'Lexend, sans-serif',
                fontSize: 'clamp(1.9rem, 3.5vw, 2.6rem)',
                letterSpacing: '-0.015em',
              }}
            >
              How it works
            </h2>
          </FadeIn>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <StepCard
              icon={Eye}
              title="We observe"
              description="Your camera captures natural facial patterns — no special setup, no performance needed."
              delay={0}
            />
            <StepCard
              icon={Sparkles}
              title="We understand"
              description="We analyze focus, eye movement, and subtle behavioral signals in real time."
              delay={0.15}
            />
            <StepCard
              icon={Layers}
              title="We reflect"
              description="You receive a gentle estimate of your mental effort — a quiet mirror, not a verdict."
              delay={0.3}
            />
          </div>
        </div>
      </section>

      {/* ── Trust Section ─────────────────────────────────────────────────── */}
      <section
        className="relative overflow-hidden py-28"
        style={{
          background: 'linear-gradient(160deg, rgba(198,210,219,0.22) 0%, rgba(248,244,234,1) 60%)',
        }}
      >
        <div className="container mx-auto px-6">
          <div className="max-w-2xl mx-auto">
            <FadeIn className="text-center mb-14">
              <p
                className="text-xs font-medium uppercase tracking-widest mb-4"
                style={{ color: '#8F9D8F' }}
              >
                Our commitment
              </p>
              <h2
                className="font-light"
                style={{
                  color: '#4A405A',
                  fontFamily: 'Lexend, sans-serif',
                  fontSize: 'clamp(1.7rem, 3vw, 2.3rem)',
                  letterSpacing: '-0.015em',
                  lineHeight: '1.3',
                }}
              >
                Designed to be transparent,
                <br />
                not intrusive
              </h2>
            </FadeIn>

            <FadeIn delay={0.1}>
              <div
                className="rounded-3xl p-10 flex flex-col gap-6"
                style={{
                  background: 'rgba(255,255,255,0.6)',
                  boxShadow: '0 4px 32px rgba(74, 64, 90, 0.07)',
                  backdropFilter: 'blur(8px)',
                }}
              >
                <TrustPoint text="No raw video is stored" />
                <div style={{ height: '1px', background: 'rgba(198,210,219,0.5)' }} />
                <TrustPoint text="Built on interpretable behavioral signals" />
                <div style={{ height: '1px', background: 'rgba(198,210,219,0.5)' }} />
                <TrustPoint text="Not a diagnosis, just awareness" />
              </div>
            </FadeIn>

            {/* Trust icons row */}
            <FadeIn delay={0.2} className="flex justify-center gap-10 mt-10">
              {[
                { icon: VideoOff, label: 'No storage' },
                { icon: ShieldCheck, label: 'Private' },
                { icon: Info, label: 'Transparent' },
              ].map(({ icon: Icon, label }) => (
                <div key={label} className="flex flex-col items-center gap-2">
                  <div
                    className="w-10 h-10 rounded-2xl flex items-center justify-center"
                    style={{ background: 'rgba(143, 157, 143, 0.15)' }}
                  >
                    <Icon size={18} style={{ color: '#8F9D8F' }} strokeWidth={1.5} />
                  </div>
                  <span className="text-xs" style={{ color: '#6C757D' }}>
                    {label}
                  </span>
                </div>
              ))}
            </FadeIn>
          </div>
        </div>
      </section>

      {/* ── Result Preview ────────────────────────────────────────────────── */}
      <section
        className="relative overflow-hidden py-28"
        style={{ background: '#F8F4EA' }}
      >
        <Blob
          color="rgba(154, 140, 177, 0.09)"
          style={{ bottom: '-80px', left: '-60px', width: 400, height: 400 }}
        />

        <div className="container mx-auto px-6">
          <FadeIn className="text-center mb-14">
            <p
              className="text-xs font-medium uppercase tracking-widest mb-4"
              style={{ color: '#9A8CB1' }}
            >
              Sample output
            </p>
            <h2
              className="font-light"
              style={{
                color: '#4A405A',
                fontFamily: 'Lexend, sans-serif',
                fontSize: 'clamp(1.9rem, 3.5vw, 2.6rem)',
                letterSpacing: '-0.015em',
              }}
            >
              What you might see
            </h2>
          </FadeIn>

          <FadeIn delay={0.1} className="flex justify-center">
            <motion.div
              whileHover={{ y: -4, boxShadow: '0 16px 56px rgba(74, 64, 90, 0.13)' }}
              className="rounded-3xl p-10 max-w-md w-full"
              style={{
                background: '#C6D2DB',
                boxShadow: '0 6px 32px rgba(74, 64, 90, 0.09)',
                transition: 'box-shadow 0.25s ease, transform 0.25s ease',
              }}
            >
              {/* Card header */}
              <div className="flex items-center gap-2 mb-6">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ background: '#9A8CB1' }}
                />
                <span
                  className="text-xs font-medium uppercase tracking-widest"
                  style={{ color: '#4A405A', opacity: 0.6 }}
                >
                  MindMirror Reading
                </span>
              </div>

              <p
                className="text-lg font-light leading-relaxed mb-8"
                style={{
                  color: '#4A405A',
                  fontFamily: 'Lexend, sans-serif',
                  lineHeight: '1.65',
                }}
              >
                It looks like your mind might be working a bit harder right now.
              </p>

              <div className="flex items-center justify-between mb-8">
                <span
                  className="text-sm px-4 py-1.5 rounded-full font-medium"
                  style={{ background: '#9A8CB1', color: '#fff' }}
                >
                  Cognitive Load: Medium
                </span>
                <span className="text-sm" style={{ color: '#6C757D' }}>
                  74% confidence
                </span>
              </div>

              {/* Load bar */}
              <div
                className="w-full h-1.5 rounded-full mb-8"
                style={{ background: 'rgba(74, 64, 90, 0.12)' }}
              >
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: '58%' }}
                  viewport={{ once: true }}
                  transition={{ duration: 1.2, delay: 0.3, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ background: '#9A8CB1' }}
                />
              </div>

              <button
                className="w-full py-3 rounded-full text-sm font-medium transition-all duration-200"
                style={{
                  border: '1.5px solid #9A8CB1',
                  color: '#9A8CB1',
                  background: 'transparent',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = '#9A8CB1';
                  e.currentTarget.style.color = '#fff';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = '#9A8CB1';
                }}
              >
                View details
              </button>
            </motion.div>
          </FadeIn>
        </div>
      </section>

      {/* ── Final CTA ─────────────────────────────────────────────────────── */}
      <section
        className="py-28 text-center"
        style={{
          background: 'linear-gradient(175deg, rgba(198,210,219,0.25) 0%, #F8F4EA 100%)',
        }}
      >
        <FadeIn className="container mx-auto px-6 max-w-xl">
          <h2
            className="font-light mb-5"
            style={{
              color: '#4A405A',
              fontFamily: 'Lexend, sans-serif',
              fontSize: 'clamp(1.8rem, 3.5vw, 2.4rem)',
              letterSpacing: '-0.015em',
              lineHeight: '1.3',
            }}
          >
            Ready when you are.
            <br />
            No rush.
          </h2>
          <p
            className="mb-10 mx-auto"
            style={{ color: '#6C757D', fontSize: '1rem', lineHeight: '1.8', maxWidth: '380px' }}
          >
            Take a breath. Open MindMirror whenever you feel ready to check in with yourself.
          </p>
          <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            <Link
              to="/analysis"
              className="inline-block px-10 py-4 rounded-full text-white text-sm font-medium transition-colors duration-200"
              style={{ background: '#9A8CB1' }}
              onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = '#7d6e9a'}
              onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = '#9A8CB1'}
            >
              Start Live Analysis
            </Link>
          </motion.div>
        </FadeIn>
      </section>
    </>
  );
}
