import { Link } from 'react-router-dom';

import type { Bucket, OverviewStats } from '../api/client';
import { plural } from '../lib/format';
import { Card, CardHead } from './ui';

export default function DataSteps({ buckets, stats }: { buckets: Bucket[]; stats: OverviewStats }) {
  const source = buckets.find(
    (bucket) => bucket.stats.parsed_documents >= 2 && bucket.id !== 'large',
  );
  const review = source ?? buckets.find((bucket) => bucket.stats.documents > 0);
  return (
    <Card>
      <CardHead title="From files to a model" end={<span className="muted">Three steps</span>} />
      <ol className="workflow">
        <li>
          <span className="step-number">1</span>
          <div>
            <span className="step-state">
              {stats.parsed_documents
                ? `${plural(stats.parsed_documents, 'document')} parsed`
                : 'Start here'}
            </span>
            <h3>Add your text</h3>
            <p>Keep files from the same source in one collection.</p>
            <Link to="/data">
              {buckets.length ? 'See your collections →' : 'Add a collection →'}
            </Link>
          </div>
        </li>
        <li>
          <span className="step-number">2</span>
          <div>
            <span className="step-state">{review ? 'Ready to review' : 'Waiting for files'}</span>
            <h3>Check your files</h3>
            <p>Read the extracted text and check any quality flags.</p>
            {review ? (
              <Link to={`/data/${review.id}`}>Review {review.name} →</Link>
            ) : (
              <span className="muted">Upload a file to get started.</span>
            )}
          </div>
        </li>
        <li>
          <span className="step-number">3</span>
          <div>
            <span className="step-state">{source ? 'Ready to train' : 'Needs more text'}</span>
            <h3>Train a model</h3>
            <p>Train one on a collection, then fine-tune it and chat with it.</p>
            {source ? (
              <Link to={`/train?data=${encodeURIComponent(source.id)}`}>Train on this →</Link>
            ) : (
              <span className="muted">One collection needs 2 to 1,000 parsed documents.</span>
            )}
          </div>
        </li>
      </ol>
    </Card>
  );
}
