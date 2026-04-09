import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer style={{ background: '#C6D2DB' }}>
      <div className="container mx-auto px-6 py-12">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
          {/* Logo */}
          <span
            className="text-lg font-semibold tracking-tight"
            style={{ color: '#4A405A', fontFamily: 'Lexend, sans-serif' }}
          >
            MindMirror
          </span>

          {/* Links */}
          <nav className="flex gap-8">
            {['Privacy', 'About', 'Contact'].map(label => (
              <Link
                key={label}
                to={`/${label.toLowerCase()}`}
                className="text-sm transition-colors duration-200"
                style={{ color: '#6C757D' }}
                onMouseEnter={e => (e.currentTarget.style.color = '#4A405A')}
                onMouseLeave={e => (e.currentTarget.style.color = '#6C757D')}
              >
                {label}
              </Link>
            ))}
          </nav>
        </div>

        {/* Divider */}
        <div
          className="mt-8 pt-6"
          style={{ borderTop: '1px solid rgba(74, 64, 90, 0.12)' }}
        >
          <p className="text-xs" style={{ color: '#6C757D' }}>
            © 2026 MindMirror. Not a medical device.
          </p>
        </div>
      </div>
    </footer>
  );
}
