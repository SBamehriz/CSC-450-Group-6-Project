import { Navigate, Route, Routes } from 'react-router-dom';

import Layout from './components/Layout';
import BucketDetailPage from './pages/BucketDetailPage';
import BucketsPage from './pages/BucketsPage';
import OverviewPage from './pages/OverviewPage';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/data" element={<BucketsPage />} />
        <Route path="/data/:bucketId" element={<BucketDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
