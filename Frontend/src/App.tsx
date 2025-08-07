import React from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import PredictionTabs from './components/PredictionTabs';
import './App.css';

function App() {
  return (
    <div className="App">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#27ae60',
              secondary: '#fff',
            },
          },
          error: {
            duration: 4000,
            iconTheme: {
              primary: '#e74c3c',
              secondary: '#fff',
            },
          },
        }}
      />
      <div className="container">
        <Header />
        <PredictionTabs />
      </div>
    </div>
  );
}

export default App;
