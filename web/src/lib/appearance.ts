// index.html reads the same key early so dark mode doesn't flash white
import { useSyncExternalStore } from 'react';

const APPEARANCE_KEY = 'forge.appearance';

type Appearance = 'light' | 'dark' | 'system';

export const APPEARANCES: { value: Appearance; label: string }[] = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'Match this computer' },
];

// for private windows that can't save it
let visit: Appearance | null = null;
const listeners = new Set<() => void>();

function read(): Appearance {
  if (visit) return visit;
  try {
    const kept = window.localStorage.getItem(APPEARANCE_KEY);
    return kept === 'dark' || kept === 'system' ? kept : 'light';
  } catch {
    return 'light';
  }
}

// no data-theme means the css media query decides
export function apply(choice: Appearance) {
  const root = document.documentElement;
  if (choice === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', choice);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  window.addEventListener('storage', listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener('storage', listener);
  };
}

function choose(choice: Appearance) {
  try {
    if (choice === 'light') window.localStorage.removeItem(APPEARANCE_KEY);
    else window.localStorage.setItem(APPEARANCE_KEY, choice);
    visit = null;
  } catch {
    visit = choice;
  }
  apply(choice);
  for (const listener of listeners) listener();
}

export function useAppearance(): [Appearance, (choice: Appearance) => void] {
  return [useSyncExternalStore(subscribe, read, () => 'light' as Appearance), choose];
}

export function remembered(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === 'yes';
  } catch {
    return false;
  }
}

export function remember(key: string, yes: boolean) {
  try {
    if (yes) window.localStorage.setItem(key, 'yes');
    else window.localStorage.removeItem(key);
  } catch {
    // fine, it just forgets
  }
}
