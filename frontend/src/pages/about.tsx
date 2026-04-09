import { motion } from 'motion/react';
import { Link } from 'react-router-dom';
import { ArrowLeft, BookOpen, HeartPulse, Scale } from 'lucide-react';

export default function AboutPage() {
  return (
    <>
      <title>About — MindMirror</title>
      <meta
        name="description"
        content="What MindMirror is, how it works, and how we treat your privacy. Estimates only—not clinical or diagnostic."
      />

      <div
        className="min-h-screen"
        style={{
          paddingTop: '100px',
          paddingBottom: '80px',
          background: 'linear-gradient(180deg, #F8F4EA 0%, rgba(237,240,244,0.9) 100%)',
        }}
      >
        <div className="container mx-auto px-6 max-w-3xl">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
          >
            <Link
              to="/"
              className="inline-flex items-center gap-2 text-sm font-medium mb-10 transition-colors"
              style={{ color: '#6C757D' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
              onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
            >
              <ArrowLeft size={16} strokeWidth={1.75} />
              Back to home
            </Link>

            <p
              className="text-xs font-medium uppercase tracking-widest mb-3"
              style={{ color: '#9A8CB1' }}
            >
              MindMirror
            </p>
            <h1
              className="font-light mb-6"
              style={{
                color: '#4A405A',
                fontFamily: 'Lexend, sans-serif',
                fontSize: 'clamp(2rem, 4vw, 2.75rem)',
                letterSpacing: '-0.02em',
                lineHeight: '1.2',
              }}
            >
              About this project
            </h1>
            <p className="text-base leading-relaxed mb-10" style={{ color: '#4F4F4F', lineHeight: '1.8' }}>
              MindMirror estimates <strong style={{ color: '#4A405A', fontWeight: 500 }}>cognitive load</strong> from
              short face video: we derive landmarks and behavioral features (eyes, gaze, head, mouth), then run trained
              models for a continuous score and a coarse band (for example LOW / MEDIUM / HIGH). Outputs are{' '}
              <strong style={{ color: '#4A405A', fontWeight: 500 }}>estimates for reflection and prototyping</strong>,
              not a diagnosis, medical device, or workplace certification.
            </p>
          </motion.div>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.08 }}
            className="rounded-3xl p-8 mb-8"
            style={{
              background: 'rgba(255,255,255,0.72)',
              boxShadow: '0 4px 24px rgba(74, 64, 90, 0.07)',
            }}
          >
            <div className="flex items-start gap-4 mb-4">
              <div
                className="w-11 h-11 rounded-2xl flex items-center justify-center flex-shrink-0"
                style={{ background: 'rgba(154, 140, 177, 0.14)' }}
              >
                <BookOpen size={20} style={{ color: '#9A8CB1' }} strokeWidth={1.5} />
              </div>
              <div>
                <h2
                  className="text-lg font-medium mb-2"
                  style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
                >
                  Research basis
                </h2>
                <p className="text-sm leading-relaxed" style={{ color: '#6C757D', lineHeight: '1.75' }}>
                  Models are trained on behavioral features aligned with remote-work style video and workload-related
                  ratings in the literature (e.g. AVCAffe-style supervision). Always check dataset licenses and ethics
                  before any commercial or redistributive use.
                </p>
              </div>
            </div>
          </motion.section>

          <motion.section
            id="privacy"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.14 }}
            className="rounded-3xl p-8 mb-8 scroll-mt-28"
            style={{
              background: 'rgba(255,255,255,0.72)',
              boxShadow: '0 4px 24px rgba(74, 64, 90, 0.07)',
            }}
          >
            <div className="flex items-start gap-4 mb-4">
              <div
                className="w-11 h-11 rounded-2xl flex items-center justify-center flex-shrink-0"
                style={{ background: 'rgba(143, 157, 143, 0.18)' }}
              >
                <Scale size={20} style={{ color: '#8F9D8F' }} strokeWidth={1.5} />
              </div>
              <div>
                <h2
                  className="text-lg font-medium mb-2"
                  style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
                >
                  Privacy
                </h2>
                <p className="text-sm leading-relaxed mb-3" style={{ color: '#6C757D', lineHeight: '1.75' }}>
                  The product is designed so analysis can run without storing raw video for later replay. What your
                  deployment logs or retains depends on your hosting and configuration—review your inference server and
                  host policies. Use clear consent when recording anyone on camera.
                </p>
                <p className="text-sm leading-relaxed" style={{ color: '#6C757D', lineHeight: '1.75' }}>
                  For questions about this deployment, use the contact section below.
                </p>
              </div>
            </div>
          </motion.section>

          <motion.section
            id="contact"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.2 }}
            className="rounded-3xl p-8 mb-12 scroll-mt-28"
            style={{
              background: 'rgba(255,255,255,0.72)',
              boxShadow: '0 4px 24px rgba(74, 64, 90, 0.07)',
            }}
          >
            <div className="flex items-start gap-4">
              <div
                className="w-11 h-11 rounded-2xl flex items-center justify-center flex-shrink-0"
                style={{ background: 'rgba(198, 210, 219, 0.35)' }}
              >
                <HeartPulse size={20} style={{ color: '#4A405A' }} strokeWidth={1.5} />
              </div>
              <div>
                <h2
                  className="text-lg font-medium mb-2"
                  style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
                >
                  Contact
                </h2>
                <p className="text-sm leading-relaxed" style={{ color: '#6C757D', lineHeight: '1.75' }}>
                  For feedback or deployment issues, reach out through the channels you use for this repository or
                  project maintainer. This page is informational and does not provide clinical support.
                </p>
              </div>
            </div>
          </motion.section>

          <div className="text-center">
            <Link
              to="/analysis"
              className="inline-block px-8 py-3.5 rounded-full text-white text-sm font-medium transition-colors duration-200"
              style={{ background: '#9A8CB1' }}
              onMouseEnter={e => ((e.currentTarget as HTMLElement).style.background = '#7d6e9a')}
              onMouseLeave={e => ((e.currentTarget as HTMLElement).style.background = '#9A8CB1')}
            >
              Try live analysis
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}
