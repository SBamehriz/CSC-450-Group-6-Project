const FLAGS: Record<string, { label: string; why: string }> = {
  too_short: { label: 'Very short', why: 'There may be too little text here to be useful.' },
  low_alpha: {
    label: 'Mostly symbols',
    why: 'Check whether this is a table, code, or text that was read incorrectly.',
  },
  long_lines: {
    label: 'Very long lines',
    why: 'The source may have lost its line breaks during extraction.',
  },
  encoding_suspect: {
    label: 'Check the characters',
    why: 'The source triggered an encoding check. Read the extracted text to see whether it looks right.',
  },
};

export function flagText(name: string): { label: string; why: string } {
  return (
    FLAGS[name] ?? {
      label: name.replace(/_/g, ' '),
      why: 'Check the extracted text before using it.',
    }
  );
}
