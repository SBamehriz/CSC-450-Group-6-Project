import { useQuery } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { Link, NavLink } from 'react-router-dom';

import { api } from '../api/client';
import { Icon } from './ui';

const LINKS = [
  { to: '/', label: 'Overview', icon: 'home' },
  { to: '/data', label: 'Collections', icon: 'data' },
  { to: '/datasets', label: 'Datasets', icon: 'snapshot' },
] as const;

export default function Layout({ children }: { children: ReactNode }) {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000 });
  const databaseUp = !health.error && health.data?.database === 'ok';
  const state = health.isPending ? 'checking' : databaseUp ? 'ok' : 'bad';
  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <header className="topbar">
        <Link to="/" className="wordmark" aria-label="Forge home">
          Forge<span>Group 6</span>
        </Link>
        <nav className="nav" aria-label="Main navigation">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) => (isActive ? 'active' : undefined)}
            >
              <Icon name={link.icon} />
              {link.label}
            </NavLink>
          ))}
        </nav>
        <div className="topbar-end">
          <span className={`system-status system-status-${state}`} role="status">
            <span className="dot" aria-hidden="true" />
            {health.isPending ? 'Connecting…' : databaseUp ? 'Connected' : 'Server unavailable'}
          </span>
          {state === 'bad' ? (
            <button
              className="btn btn-sm"
              type="button"
              disabled={health.isFetching}
              onClick={() => void health.refetch()}
            >
              Reconnect
            </button>
          ) : null}
        </div>
      </header>
      <main className="page" id="main-content" tabIndex={-1}>
        {children}
      </main>
      <footer className="page-footer">Forge · CSC 450 Group 6</footer>
    </>
  );
}
