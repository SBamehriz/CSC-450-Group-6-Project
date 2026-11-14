import { Navigate, Route, Routes, useLocation } from 'react-router-dom';

import Layout from './components/Layout';
import BucketDetailPage from './pages/BucketDetailPage';
import BucketsPage from './pages/BucketsPage';
import ChatPage from './pages/ChatPage';
import ComparePage from './pages/ComparePage';
import ComputerPage from './pages/ComputerPage';
import ModelsPage from './pages/ModelsPage';
import OverviewPage from './pages/OverviewPage';
import RunDetailPage from './pages/RunDetailPage';
import TrainingHistoryPage from './pages/TrainingHistoryPage';

// old links still work, query string and all
function Moved({ to }: { to: string }) {
  const { search } = useLocation();
  return <Navigate to={`${to}${search}`} replace />;
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/data" element={<BucketsPage />} />
        <Route path="/data/:bucketId" element={<BucketDetailPage />} />
        <Route path="/datasets/*" element={<Moved to="/data" />} />
        <Route path="/vocabularies" element={<Moved to="/train" />} />
        <Route path="/training-sets" element={<Moved to="/train" />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/evaluations" element={<Moved to="/compare/measure" />} />
        <Route path="/train" element={<TrainingHistoryPage />} />
        <Route path="/history" element={<Moved to="/train" />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/compare/measure" element={<ComparePage />} />
        <Route path="/settings" element={<ComputerPage />} />
        <Route path="/gpu" element={<Moved to="/settings" />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
