import { RouteObject } from 'react-router-dom';
import { lazy } from 'react';
import HomePage from './pages/index';
import AboutPage from './pages/about';
import AnalysisPage from './pages/analysis';
import DashboardPage from './pages/dashboard';
import RequireAuth from './components/RequireAuth';
import SignInPage from './pages/sign-in';
import SignUpPage from './pages/sign-up';

const NotFoundPage = lazy(() => import('./pages/_404'));

export const routes: RouteObject[] = [
  { path: '/', element: <HomePage /> },
  { path: '/about', element: <AboutPage /> },
  { path: '/sign-in/*', element: <SignInPage /> },
  { path: '/sign-up/*', element: <SignUpPage /> },
  {
    path: '/analysis',
    element: (
      <RequireAuth
        title="Sign in to start a session"
        description="Live analysis sends video to your inference server. Sign in so only you can run sessions from this browser."
      >
        <AnalysisPage />
      </RequireAuth>
    ),
  },
  {
    path: '/dashboard',
    element: (
      <RequireAuth
        title="Sign in to view your sessions"
        description="Your saved sessions are tied to your account."
      >
        <DashboardPage />
      </RequireAuth>
    ),
  },
  { path: '*', element: <NotFoundPage /> },
];

export type Path = '/' | '/about' | '/analysis' | '/dashboard' | '/sign-in' | '/sign-up';
export type Params = Record<string, string | undefined>;
