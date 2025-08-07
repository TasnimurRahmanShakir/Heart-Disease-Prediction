import React, { useState } from 'react';
import SingleImagePrediction from './SingleImagePrediction';
import BatchPrediction from './BatchPrediction';
import './PredictionTabs.css';

type TabType = 'single' | 'batch';

const PredictionTabs: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('single');

  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
  };

  return (
    <div className="prediction-tabs">
      <div className="nav-tabs">
        <button 
          className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => handleTabChange('single')}
        >
          <i className="fas fa-image"></i>
          Single Image
        </button>
        <button 
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => handleTabChange('batch')}
        >
          <i className="fas fa-folder"></i>
          Batch Upload
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'single' && <SingleImagePrediction />}
        {activeTab === 'batch' && <BatchPrediction />}
      </div>
    </div>
  );
};

export default PredictionTabs; 