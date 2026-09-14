import { useQuery } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

import { api } from '../api/client';

const LINKS = [
  { to: '/', label: 'Overview' },
  { to: '/data', label: 'Data' },
];

export default function Layout({ children }: { children: ReactNode }) {
  const health = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  const databaseUp = !health.error && health.data?.database === 'ok';
  const state = health.isPending ? 'checking' : databaseUp ? 'ok' : 'bad';

  return (
    <>
      <header className="topbar">
        <nav className="nav" aria-label="Main navigation">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) => (isActive ? 'active' : undefined)}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
        <div className="topbar-end">
          <span className={`system-status system-status-${state}`} role="status">
            <span className="dot" aria-hidden="true" />
            {health.isPending ? 'Checking.' : databaseUp ? 'System OK' : 'System unavailable'}
          </span>
        </div>
      </header>
      <main className="page">{children}</main>
    </>
  );
}