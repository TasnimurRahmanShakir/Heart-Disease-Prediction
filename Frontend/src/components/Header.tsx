import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Header.css';

interface ModelStatus {
  status: string;
  device: string;
  model_type: string;
  trained_model_loaded: boolean;
  model_file_size: string;
  message: string;
}

const Header: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const response = await axios.get(' https://heart-disease-prediction-8-xuyq.onrender.com/api/model/status');
        setModelStatus(response.data);
      } catch (error) {
        console.error('Failed to get model status:', error);
        setModelStatus({
          status: 'error',
          device: 'Unknown',
          model_type: 'Unknown',
          trained_model_loaded: false,
          model_file_size: 'N/A',
          message: 'Could not connect to backend'
        });
      } finally {
        setLoading(false);
      }
    };

    checkModelStatus();
  }, []);

  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <div className="heartbeat-icon">
            <i className="fas fa-heartbeat"></i>
          </div>
          <h1>MedScan</h1>
        </div>
        <p className="tagline">AI-Powered Heart Disease Detection</p>
        
        {/* Model Status Indicator */}
        <div className="model-status">
          {loading ? (
            <div className="status-indicator loading">
              <i className="fas fa-spinner fa-spin"></i>
              <span>Checking Model...</span>
            </div>
          ) : modelStatus ? (
            <div className={`status-indicator ${modelStatus.trained_model_loaded ? 'ready' : 'demo'}`}>
              <i className={`fas ${modelStatus.trained_model_loaded ? 'fa-check-circle' : 'fa-info-circle'}`}></i>
              <span>
                {modelStatus.trained_model_loaded ? modelStatus.trained_model_loaded : 'Demo Mode'}
              </span>
              <div className="status-tooltip">
                <strong>Model Status:</strong> {modelStatus.message}<br/>
                <strong>Device:</strong> {modelStatus.device}<br/>
                <strong>Type:</strong> {modelStatus.model_type}<br/>
                {modelStatus.trained_model_loaded && (
                  <><strong>Size:</strong> {modelStatus.model_file_size}</>
                )}
              </div>
            </div>
          ) : (
            <div className="status-indicator error">
              <i className="fas fa-exclamation-triangle"></i>
              <span>Model Error</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header; 