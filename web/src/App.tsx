import { Navigate, Route, Routes } from 'react-router-dom';

import Layout from './components/Layout';
import BucketDetailPage from './pages/BucketDetailPage';
import BucketsPage from './pages/BucketsPage';
import DatasetsPage from './pages/DatasetsPage';
import ModelsPage from './pages/ModelsPage';
import OverviewPage from './pages/OverviewPage';
import PreparedDataPage from './pages/PreparedDataPage';
import RunDetailPage from './pages/RunDetailPage';
import TrainingHistoryPage from './pages/TrainingHistoryPage';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/data" element={<BucketsPage />} />
        <Route path="/data/:bucketId" element={<BucketDetailPage />} />
        <Route path="/datasets" element={<DatasetsPage />} />
        <Route path="/datasets/:datasetId/prepare" element={<PreparedDataPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/history" element={<TrainingHistoryPage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
