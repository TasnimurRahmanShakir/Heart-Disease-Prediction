import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'react-hot-toast';
import axios from 'axios';
import './BatchPrediction.css';

interface BatchResult {
  total_images: number;
  positive_cases: number;
  negative_cases: number;
  confusion_matrix_path?: string;
  roc_curve_path?: string;
  sample_predictions_path?: string;
  classification_report?: string;
  roc_auc?: number;
}

const BatchPrediction: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      if (!file.name.endsWith('.zip')) {
        toast.error('Please select a ZIP file');
        return;
      }
      setSelectedFile(file);
      setResult(null);
      toast.success('ZIP file selected successfully');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/zip': ['.zip'],
    },
    multiple: false,
  });

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('https://heart-disease-prediction-8-xuyq.onrender.com/api/predict/folder', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
      toast.success('Batch analysis completed successfully!');
    } catch (error) {
      console.error('Error:', error);
      toast.error('Error analyzing batch. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = (type: 'confusion_matrix' | 'roc_curve' | 'sample_predictions') => {
    if (result) {
      const url = `https://heart-disease-prediction-8-xuyq.onrender.com/api/download/${type}`;
      const link = document.createElement('a');
      link.href = url;
      link.download = `${type}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const confMatrixUrl = result?.confusion_matrix_path
    ? `https://heart-disease-prediction-8-xuyq.onrender.com/api/download/confusion_matrix`
    : '';
  const rocCurveUrl = result?.roc_curve_path
    ? `https://heart-disease-prediction-8-xuyq.onrender.com/api/download/roc_curve`
    : '';
  const samplePredictionsUrl = result?.sample_predictions_path
    ? `https://heart-disease-prediction-8-xuyq.onrender.com/api/download/sample_predictions`
    : '';

  return (
    <div className="batch-prediction">
      <div className="prediction-card">
        <h2>Batch Analysis</h2>
        <p>Upload a ZIP file containing multiple chest X-ray images</p>

        <div {...getRootProps()} className={`upload-area ${isDragActive ? 'dragover' : ''}`}>
          <input {...getInputProps()} />
          <div className="upload-content">
            <i className="fas fa-file-archive"></i>
            <h3>Drop your ZIP file here</h3>
            <p>or click to browse</p>
            {selectedFile && (
              <div className="selected-file">
                <i className="fas fa-check"></i>
                {selectedFile.name}
              </div>
            )}
          </div>
        </div>

        <button className="predict-btn" onClick={handlePredict} disabled={!selectedFile || isLoading}>
          {isLoading ? (
            <>
              <div className="spinner"></div>
              Analyzing Batch...
            </>
          ) : (
            <>
              <i className="fas fa-search"></i>
              Analyze Batch
            </>
          )}
        </button>

        {result && (
          <div className="batch-result-area">
            <div className="result-card">
              <h3>Batch Analysis Results</h3>
              <div className="batch-stats">
                <div className="stat">
                  <span className="stat-number">{result.total_images}</span>
                  <span className="stat-label">Total Images</span>
                </div>
                <div className="stat">
                  <span className="stat-number positive">{result.positive_cases}</span>
                  <span className="stat-label">Positive Cases</span>
                </div>
                <div className="stat">
                  <span className="stat-number negative">{result.negative_cases}</span>
                  <span className="stat-label">Negative Cases</span>
                </div>
              </div>


              {result.roc_auc && (
                <div className="roc-score">
                  <h4>ROC AUC Score: {result.roc_auc.toFixed(4)}</h4>
                </div>
              )}

              {result.classification_report && (
                <div className="classification-report">
                  <h4>Classification Report</h4>
                  <pre>{result.classification_report}</pre>
                </div>
              )}

              <div className="results-section">
                <div className="result-item">
                  <h3>Confusion Matrix</h3>
                  {confMatrixUrl && <img src={confMatrixUrl} alt="Confusion Matrix" />}
                  <button
                  className="download-btn"
                  onClick={() => handleDownload('confusion_matrix')}
                  disabled={!result.confusion_matrix_path}
                >
                  <i className="fas fa-chart-bar"></i>
                  Download Confusion Matrix
                </button>
                </div>
                <div className="result-item">
                  <h3>ROC Curve</h3>
                  {rocCurveUrl && <img src={rocCurveUrl} alt="ROC Curve" />}
                  <button
                  className="download-btn"
                  onClick={() => handleDownload('roc_curve')}
                  disabled={!result.roc_curve_path}
                >
                  <i className="fas fa-chart-line"></i>
                  Download ROC Curve
                </button>
                </div>
                <div className="result-item full-width">
                  <h3>Sample Predictions</h3>
                  {samplePredictionsUrl && <img src={samplePredictionsUrl} alt="Sample Predictions" />}
                  <button
                  className="download-btn"
                  onClick={() => handleDownload('sample_predictions')}
                  disabled={!result.sample_predictions_path}
                >
                  <i className="fas fa-images"></i>
                  Download Sample Predictions
                </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BatchPrediction;
