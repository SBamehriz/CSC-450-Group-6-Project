// every page title uses .h1 so they all match
import { expect, it } from 'vitest';

const pages = import.meta.glob<string>(['./pages/*.tsx', '!./pages/*.test.tsx'], {
  query: '?raw',
  import: 'default',
  eager: true,
});
const styles = import.meta.glob<string>('./**/*.css', {
  query: '?raw',
  import: 'default',
  eager: true,
});

it('gives every page title the shared class and no size of its own', () => {
  expect(Object.keys(pages).length).toBeGreaterThan(5);
  for (const [file, source] of Object.entries(pages)) {
    for (const tag of source.match(/<h1\b[^>]*>/g) ?? []) {
      expect(tag, file).toBe('<h1 className="h1">');
    }
  }
});

it('sizes the title in one place only', () => {
  for (const [file, css] of Object.entries(styles)) {
    const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, '');
    for (const [, selector, body] of withoutComments.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      const names = selector.split(',').map((one) => one.trim());
      const aboutTitle = names.filter((one) => /(^|[\s>+~])\.?h1\b/.test(one));
      if (!aboutTitle.length || !/font-(size|family)|(^|[\s;])font:/.test(body)) continue;
      expect(aboutTitle, `${file}: ${selector.trim()}`).toEqual(['.h1']);
    }
  }
});
