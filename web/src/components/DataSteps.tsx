import { Link } from 'react-router-dom';

import type { Bucket, OverviewStats } from '../api/client';
import { plural } from '../lib/format';
import { Card, CardHead } from './ui';

export default function DataSteps({ buckets, stats }: { buckets: Bucket[]; stats: OverviewStats }) {
  const source = buckets.find(
    (bucket) => bucket.stats.parsed_documents >= 2 && bucket.stats.parsed_documents <= 1000,
  );
  const review = source ?? buckets.find((bucket) => bucket.stats.documents > 0);
  return (
    <Card>
      <CardHead title="From files to a dataset" end={<span className="muted">Three steps</span>} />
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
            <span className="step-state">
              {stats.ready_snapshots
                ? `${plural(stats.ready_snapshots, 'snapshot')} saved`
                : source
                  ? 'Ready to build'
                  : 'Needs more text'}
            </span>
            <h3>Save a snapshot</h3>
            <p>Freeze one source into train and validation files.</p>
            {source ? (
              <Link to={`/datasets?bucket=${encodeURIComponent(source.id)}`}>
                Create a snapshot →
              </Link>
            ) : (
              <span className="muted">One collection needs 2 to 1,000 parsed documents.</span>
            )}
          </div>
        </li>
      </ol>
    </Card>
  );
}
