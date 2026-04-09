import { Link, useLocation } from 'react-router-dom';
import { Show, SignInButton, SignUpButton, useAuth } from '@clerk/react';
import Spinner from '@/components/Spinner';

const CLERK_ENABLED = !!import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

function ClerkNotConfigured() {
  return (
    <div
      className="min-h-[60vh] flex flex-col items-center justify-center px-6 text-center"
      style={{ background: 'linear-gradient(175deg, #e8e4f0 0%, #edf0f4 40%, #F8F4EA 100%)' }}
    >
      <h1
        className="text-xl font-light mb-3"
        style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
      >
        Authentication is not configured
      </h1>
      <p className="text-sm max-w-md mb-6" style={{ color: '#6C757D', lineHeight: 1.75 }}>
        Add <code className="text-xs px-1.5 py-0.5 rounded bg-white/80">VITE_CLERK_PUBLISHABLE_KEY</code> to{' '}
        <code className="text-xs px-1.5 py-0.5 rounded bg-white/80">frontend/.env</code> (see{' '}
        <code className="text-xs px-1.5 py-0.5 rounded bg-white/80">env.example</code>), restart{' '}
        <code className="text-xs px-1.5 py-0.5 rounded bg-white/80">npm run dev</code>, then return here.
      </p>
      <Link
        to="/"
        className="text-sm font-medium"
        style={{ color: '#9A8CB1' }}
      >
        Back to home
      </Link>
    </div>
  );
}

function SignedOutWall({ title, description }: { title: string; description: string }) {
  const location = useLocation();
  const returnUrl = `${location.pathname}${location.search}`;

  return (
    <div
      className="min-h-[60vh] flex flex-col items-center justify-center px-6 text-center gap-6"
      style={{ background: 'linear-gradient(175deg, #e8e4f0 0%, #edf0f4 40%, #F8F4EA 100%)' }}
    >
      <div>
        <h1
          className="text-2xl font-light mb-2"
          style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
        >
          {title}
        </h1>
        <p className="text-sm max-w-md mx-auto" style={{ color: '#6C757D', lineHeight: 1.75 }}>
          {description}
        </p>
      </div>
      <div className="flex flex-col sm:flex-row items-center gap-3">
        <SignInButton mode="redirect" forceRedirectUrl={returnUrl}>
          <button
            type="button"
            className="px-8 py-3.5 rounded-full text-white text-sm font-medium transition-colors duration-200"
            style={{ background: '#9A8CB1' }}
          >
            Sign in
          </button>
        </SignInButton>
        <SignUpButton mode="redirect" forceRedirectUrl={returnUrl}>
          <button
            type="button"
            className="px-8 py-3.5 rounded-full text-sm font-medium transition-colors duration-200"
            style={{ border: '1.5px solid #9A8CB1', color: '#9A8CB1', background: 'transparent' }}
          >
            Create account
          </button>
        </SignUpButton>
      </div>
      <Link to="/" className="text-sm" style={{ color: '#6C757D' }}>
        Back to home
      </Link>
    </div>
  );
}

function RequireAuthClerk({
  children,
  title,
  description,
}: {
  children: React.ReactNode;
  title: string;
  description: string;
}) {
  const { isLoaded } = useAuth();

  if (!isLoaded) {
    return (
      <div className="flex justify-center py-24 min-h-[40vh] items-center">
        <Spinner />
      </div>
    );
  }

  return (
    <Show
      when="signed-in"
      fallback={<SignedOutWall title={title} description={description} />}
    >
      {children}
    </Show>
  );
}

/**
 * Wraps routes that require a signed-in Clerk user (Clerk MCP pattern: &lt;Show when="signed-in" /&gt;).
 */
export default function RequireAuth({
  children,
  title = 'Sign in to continue',
  description = 'MindMirror uses your account to keep sessions private to you.',
}: {
  children: React.ReactNode;
  title?: string;
  description?: string;
}) {
  if (!CLERK_ENABLED) {
    return <ClerkNotConfigured />;
  }

  return (
    <RequireAuthClerk title={title} description={description}>
      {children}
    </RequireAuthClerk>
  );
}
