import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';

import { api } from '../api/client';
import { apply, remember, remembered, useAppearance } from '../lib/appearance';
import { Icon, type IconName } from './ui';

type Place = { to: string; label: string; icon: IconName; owns?: string[] };

const LINKS: Place[] = [
  { to: '/', label: 'Home', icon: 'home' },
  { to: '/data', label: 'Data', icon: 'data' },
  { to: '/train', label: 'Train', icon: 'train', owns: ['/runs'] },
  { to: '/models', label: 'Models', icon: 'model' },
  { to: '/chat', label: 'Chat', icon: 'chat' },
  { to: '/compare', label: 'Compare', icon: 'compare' },
];
const SETTINGS: Place = { to: '/settings', label: 'Settings', icon: 'settings' };

const FOLDED_KEY = 'forge.sidebar-folded';
// keep in sync with theme.css
const NARROW = '(max-width: 960px)';

function followWidth(listener: () => void) {
  const query = window.matchMedia?.(NARROW);
  query?.addEventListener?.('change', listener);
  return () => query?.removeEventListener?.('change', listener);
}
const narrowNow = () => window.matchMedia?.(NARROW).matches ?? false;

function owns(place: Place, path: string) {
  if (place.to === '/') return path === '/';
  return [place.to, ...(place.owns ?? [])].some(
    (prefix) => path === prefix || path.startsWith(`${prefix}/`),
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000 });
  const down = !health.isPending && (!!health.error || health.data?.database !== 'ok');
  const [appearance] = useAppearance();
  // another tab may change it
  useEffect(() => apply(appearance), [appearance]);

  // going to another page closes the menu
  const [openOn, setOpenOn] = useState<string | null>(null);
  const open = openOn === pathname;
  const [foldedChoice, setFolded] = useState(() => remembered(FOLDED_KEY));
  const narrow = useSyncExternalStore(followWidth, narrowNow, () => false);
  const folded = foldedChoice && !narrow;
  const menu = useRef<HTMLButtonElement>(null);
  const nav = useRef<HTMLElement>(null);
  const main = useRef<HTMLElement>(null);

  // the page goes inert while the menu is open, escape closes it
  useEffect(() => {
    if (!open) return;
    const page = main.current;
    page?.setAttribute('inert', '');
    nav.current?.querySelector<HTMLElement>('a')?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setOpenOn(null);
      menu.current?.focus();
    };
    document.addEventListener('keydown', key);
    return () => {
      page?.removeAttribute('inert');
      document.removeEventListener('keydown', key);
    };
  }, [open]);

  const close = () => {
    setOpenOn(null);
    menu.current?.focus();
  };
  const fold = () => {
    remember(FOLDED_KEY, !foldedChoice);
    setFolded(!foldedChoice);
  };
  const item = (place: Place) => {
    const here = owns(place, pathname);
    return (
      <li key={place.to}>
        <Link
          to={place.to}
          className={`nav-item${here ? ' active' : ''}`}
          aria-current={here ? 'page' : undefined}
          title={folded ? place.label : undefined}
        >
          <Icon name={place.icon} />
          <span className="nav-label">{place.label}</span>
        </Link>
      </li>
    );
  };

  return (
    <div className={`shell${folded ? ' folded' : ''}${open ? ' open' : ''}`}>
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <header className="mobilebar">
        <button
          ref={menu}
          type="button"
          className="icon-button"
          aria-expanded={open}
          aria-controls="sidebar"
          aria-label={open ? 'Close the menu' : 'Open the menu'}
          onClick={() => setOpenOn(open ? null : pathname)}
        >
          <Icon name={open ? 'close' : 'menu'} />
        </button>
        <Link to="/" className="wordmark">
          Forge
        </Link>
      </header>
      <aside
        className="sidebar"
        id="sidebar"
        onClick={(event) => {
          if (!open || !(event.target as Element).closest('a')) return;
          main.current?.removeAttribute('inert');
          setOpenOn(null);
          main.current?.focus({ preventScroll: true });
        }}
      >
        <div className="sidebar-head">
          <Link to="/" className="wordmark" aria-label="Forge home">
            {folded ? 'F' : 'Forge'}
          </Link>
        </div>
        <nav className="sidebar-nav" aria-label="Main navigation" ref={nav}>
          <ul>{LINKS.map(item)}</ul>
        </nav>
        <div className="sidebar-foot">
          <ul>{item(SETTINGS)}</ul>
          <button
            type="button"
            className="nav-item fold-button"
            onClick={fold}
            title={folded ? 'Unfold the side panel' : undefined}
          >
            <Icon name="sidebar" />
            <span className="nav-label">
              {folded ? 'Unfold the side panel' : 'Fold the side panel'}
            </span>
          </button>
        </div>
      </aside>
      <div className="scrim" aria-hidden="true" onClick={close} />
      <main className="page" id="main-content" tabIndex={-1} ref={main}>
        {/* always rendered so screen readers catch the message */}
        <p className="system-status" role="status">
          {down ? (
            <>
              Server unavailable
              <button
                className="btn btn-sm"
                type="button"
                disabled={health.isFetching}
                onClick={() => void health.refetch()}
              >
                Reconnect
              </button>
            </>
          ) : null}
        </p>
        {children}
      </main>
    </div>
  );
}
