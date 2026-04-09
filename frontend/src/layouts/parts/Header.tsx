import { Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import {
  SignInButton,
  SignUpButton,
  UserButton,
  useClerk,
  useUser,
} from '@clerk/react';

const CLERK_ENABLED = !!import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

// ─── Auth buttons (only rendered when ClerkProvider is present) ───────────────
function AuthButtonsInner({ mobile = false }: { mobile?: boolean }) {
  const { isSignedIn, isLoaded } = useUser();
  const { signOut } = useClerk();

  if (!isLoaded) return null;

  if (isSignedIn) {
    const signOutBtn = (
      <button
        type="button"
        onClick={() => void signOut()}
        className="text-sm font-medium transition-colors duration-200 text-left"
        style={{ color: '#6C757D' }}
        onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
        onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
      >
        Sign out
      </button>
    );

    return (
      <div
        className={
          mobile
            ? 'flex flex-col items-stretch gap-3 mt-2'
            : 'flex items-center gap-4'
        }
      >
        <Link
          to="/dashboard"
          className="text-sm font-medium transition-colors duration-200"
          style={{ color: '#6C757D' }}
          onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
          onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
        >
          Dashboard
        </Link>
        <div className={mobile ? 'flex items-center gap-3' : 'flex items-center gap-3'}>
          <UserButton
            appearance={{
              elements: { avatarBox: 'w-8 h-8' },
            }}
          />
          {!mobile && signOutBtn}
        </div>
        {mobile && signOutBtn}
      </div>
    );
  }

  return (
    <div className={mobile ? 'flex flex-col gap-2 mt-2' : 'flex items-center gap-3'}>
      <SignInButton mode="modal">
        <button
          className="text-sm font-medium transition-colors duration-200"
          style={{ color: '#6C757D' }}
          onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
          onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
        >
          Sign in
        </button>
      </SignInButton>
      <SignUpButton mode="modal">
        <button
          className="text-sm font-medium px-5 py-2 rounded-full text-white transition-colors duration-200"
          style={{ background: '#9A8CB1' }}
          onMouseEnter={e => (e.currentTarget.style.background = '#7d6e9a')}
          onMouseLeave={e => (e.currentTarget.style.background = '#9A8CB1')}
        >
          Sign up
        </button>
      </SignUpButton>
    </div>
  );
}

// Guard: only render Clerk hooks when ClerkProvider is actually in the tree
function AuthButtons({ mobile = false }: { mobile?: boolean }) {
  if (!CLERK_ENABLED) return null;
  return <AuthButtonsInner mobile={mobile} />;
}

// ─── Header ───────────────────────────────────────────────────────────────────
export default function Header() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 transition-all duration-500"
      style={{
        background: scrolled ? 'rgba(248, 244, 234, 0.96)' : 'transparent',
        boxShadow: scrolled ? '0 1px 24px rgba(74, 64, 90, 0.07)' : 'none',
        backdropFilter: scrolled ? 'blur(12px)' : 'none',
      }}
    >
      <div className="container mx-auto px-6">
        <div className="flex h-18 items-center justify-between py-4">
          {/* Logo */}
          <Link
            to="/"
            className="text-xl font-semibold tracking-tight transition-opacity hover:opacity-70"
            style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif', letterSpacing: '-0.01em' }}
          >
            MindMirror
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-8">
            <a
              href="#how-it-works"
              className="text-sm font-medium transition-colors duration-200"
              style={{ color: '#6C757D' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
              onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
            >
              How it works
            </a>
            <Link
              to="/about"
              className="text-sm font-medium transition-colors duration-200"
              style={{ color: '#6C757D' }}
              onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
              onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
            >
              About
            </Link>
            <Link
              to="/analysis"
              className="text-sm font-medium px-5 py-2 rounded-full text-white transition-colors duration-200"
              style={{ background: '#9A8CB1' }}
              onMouseEnter={e => (e.currentTarget.style.background = '#7d6e9a')}
              onMouseLeave={e => (e.currentTarget.style.background = '#9A8CB1')}
            >
              Start
            </Link>
            <AuthButtons />
          </nav>

          {/* Mobile toggle */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-lg transition-colors"
            style={{ color: '#4A405A' }}
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Mobile menu */}
        {isMobileMenuOpen && (
          <div
            className="md:hidden py-4 pb-6"
            style={{ borderTop: '1px solid rgba(198, 210, 219, 0.5)' }}
          >
            <nav className="flex flex-col gap-4">
              <a
                href="#how-it-works"
                className="text-sm font-medium py-1"
                style={{ color: '#6C757D' }}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                How it works
              </a>
              <Link
                to="/about"
                className="text-sm font-medium py-1"
                style={{ color: '#6C757D' }}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                About
              </Link>
              <Link
                to="/analysis"
                className="text-sm font-medium px-5 py-2 rounded-full text-white text-center mt-1"
                style={{ background: '#9A8CB1' }}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Start
              </Link>
              <AuthButtons mobile />
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}
