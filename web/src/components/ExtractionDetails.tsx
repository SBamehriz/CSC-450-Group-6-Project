const LABELS: Record<string, string> = {
  source_format: 'Source format',
  engine: 'Extraction engine',
  ocr_applied: 'OCR used',
  page_count: 'Pages',
  paragraph_count: 'Paragraphs',
  heading_count: 'Headings',
  table_count: 'Tables',
  char_count: 'Extracted characters',
  word_count: 'Extracted words',
  line_count: 'Extracted lines',
  width: 'Image width',
  height: 'Image height',
  image_mode: 'Image mode',
  image_format: 'Image format',
};

export default function ExtractionDetails({ quality }: { quality: Record<string, unknown> }) {
  const extraction = quality.extraction;
  const entries =
    extraction && typeof extraction === 'object' && !Array.isArray(extraction)
      ? Object.entries(extraction).filter(([, value]) =>
          ['string', 'number', 'boolean'].includes(typeof value),
        )
      : [];
  return (
    <section aria-label="Extraction details">
      <h3 className="h2">Extraction details</h3>
      {entries.length ? (
        <dl className="extraction-grid">
          {entries.map(([key, value]) => (
            <div key={key}>
              <dt className="label">{LABELS[key] ?? key.replace(/_/g, ' ')}</dt>
              <dd>{typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="muted">No extraction details stored for this file.</p>
      )}
    </section>
  );
}
