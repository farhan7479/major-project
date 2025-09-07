import React, { useState, useEffect } from 'react';
import EnergyChart from './components/EnergyChart';
import PredictionCard from './components/PredictionCard';
import LoadingSpinner from './components/LoadingSpinner';
import AlgorithmComparisonChart from './components/AlgorithmComparisonChart';
import ConfidenceIntervalChart from './components/ConfidenceIntervalChart';
import SeasonalPatternChart from './components/SeasonalPatternChart';
import api from './services/api';
import './App.css';

function App() {
  const [energyData, setEnergyData] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [modelType, setModelType] = useState('both');
  const [dataHours, setDataHours] = useState(168);
  const [error, setError] = useState(null);
  const [statistics, setStatistics] = useState({});
  const [correlations, setCorrelations] = useState({});

  useEffect(() => {
    loadSampleData();
  }, [dataHours]);

  const loadSampleData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await api.getSampleData(dataHours);
      setEnergyData(response.data);
      setStatistics(response.statistics || {});
      setCorrelations(response.correlations || {});
    } catch (error) {
      setError('Failed to load sample data. Make sure the backend is running.');
      console.error('Error loading sample data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const makePrediction = async () => {
    if (energyData.length < 90) {
      setError('Need at least 90 data points for prediction');
      return;
    }

    try {
      setIsPredicting(true);
      setError(null);
      
      // Use the last 90 data points for prediction
      const inputData = energyData.slice(-90);
      const response = await api.predictEnergyConsumption(inputData, modelType);
      setPredictions(response);
    } catch (error) {
      setError('Failed to make prediction. Check if the backend is running.');
      console.error('Error making prediction:', error);
    } finally {
      setIsPredicting(false);
    }
  };

  const refreshData = () => {
    loadSampleData();
    setPredictions({});
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Energy Consumption Forecasting
              </h1>
              <p className="text-gray-600 mt-1">
                AI-powered LSTM/GRU models for energy demand prediction
              </p>
            </div>
            <div className="flex space-x-4">
              <button
                onClick={refreshData}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Refresh Data
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {/* Controls */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Data Hours
              </label>
              <select
                value={dataHours}
                onChange={(e) => setDataHours(Number(e.target.value))}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value={24}>24 Hours (1 Day)</option>
                <option value={168}>168 Hours (1 Week)</option>
                <option value={720}>720 Hours (1 Month)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Type
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="ensemble">Ensemble (All Algorithms)</option>
                <option value="both">Traditional Models</option>
                <option value="lstm">Linear Regression</option>
                <option value="gru">Holt-Winters</option>
                <option value="arima">ARIMA</option>
                <option value="seasonal">Seasonal Decomposition</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={makePrediction}
                disabled={isPredicting || isLoading}
                className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isPredicting ? 'Predicting...' : 'Make Prediction'}
              </button>
            </div>
          </div>
        </div>

                {/* Predictions */}
        {(predictions.gru_prediction || predictions.lstm_prediction || predictions.ensemble_prediction) && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {predictions.gru_prediction && (
              <PredictionCard
                title="Holt-Winters Model"
                prediction={predictions.gru_prediction}
                isLoading={isPredicting}
                color="blue"
              />
            )}
            {predictions.lstm_prediction && (
              <PredictionCard
                title="Linear Regression Model"
                prediction={predictions.lstm_prediction}
                isLoading={isPredicting}
                color="green"
              />
            )}
            {predictions.ensemble_prediction && (
              <PredictionCard
                title="Ensemble Prediction"
                prediction={predictions.next_hour_forecast?.ensemble_prediction || predictions.ensemble_prediction.ensemble_prediction}
                isLoading={isPredicting}
                color="purple"
              />
            )}
          </div>
        )}

        {/* Algorithm Comparison */}
        {predictions.algorithm_comparison && (
          <div className="mb-8">
            <AlgorithmComparisonChart 
              predictions={predictions.algorithm_comparison}
              title="Algorithm Performance Comparison"
            />
          </div>
        )}

        {/* Confidence Interval Chart */}
        {predictions.next_hour_forecast && energyData.length > 0 && (
          <div className="mb-8">
            <ConfidenceIntervalChart 
              historicalData={energyData}
              prediction={predictions}
              title="Prediction with Confidence Intervals"
            />
          </div>
        )}

        {/* Main Energy Chart */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Energy Consumption Chart</h2>
          {isLoading ? (
            <LoadingSpinner message="Loading energy data..." />
          ) : energyData.length > 0 ? (
            <EnergyChart data={energyData} predictions={predictions} />
          ) : (
            <div className="text-center py-8 text-gray-500">
              No data available
            </div>
          )}
        </div>

        {/* Seasonal Patterns */}
        {energyData.length > 0 && Object.keys(statistics).length > 0 && (
          <div className="mb-8">
            <SeasonalPatternChart 
              data={energyData}
              statistics={statistics}
              title="Energy Consumption Patterns & Statistics"
            />
          </div>
        )}

        {/* Correlations and Statistics */}
        {Object.keys(correlations).length > 0 && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4">Data Analysis</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="p-4 bg-blue-50 rounded-lg">
                <h3 className="font-semibold text-blue-800">Temperature Correlation</h3>
                <p className="text-2xl font-mono mt-2">{(correlations.temperature_correlation || 0).toFixed(3)}</p>
                <p className="text-sm text-blue-600">
                  {Math.abs(correlations.temperature_correlation || 0) > 0.5 ? 'Strong' : 'Moderate'} correlation
                </p>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <h3 className="font-semibold text-green-800">Humidity Correlation</h3>
                <p className="text-2xl font-mono mt-2">{(correlations.humidity_correlation || 0).toFixed(3)}</p>
                <p className="text-sm text-green-600">
                  {Math.abs(correlations.humidity_correlation || 0) > 0.5 ? 'Strong' : 'Moderate'} correlation
                </p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <h3 className="font-semibold text-purple-800">Data Quality</h3>
                <p className="text-2xl font-mono mt-2">{energyData.length}</p>
                <p className="text-sm text-purple-600">Data points analyzed</p>
              </div>
            </div>
          </div>
        )}

        {/* Statistics */}
        {energyData.length > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4">Data Statistics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {energyData.length}
                </div>
                <div className="text-sm text-gray-600">Data Points</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {Math.max(...energyData.map(d => d.consumption)).toFixed(1)}
                </div>
                <div className="text-sm text-gray-600">Max Consumption (MW)</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {Math.min(...energyData.map(d => d.consumption)).toFixed(1)}
                </div>
                <div className="text-sm text-gray-600">Min Consumption (MW)</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {(energyData.reduce((sum, d) => sum + d.consumption, 0) / energyData.length).toFixed(1)}
                </div>
                <div className="text-sm text-gray-600">Avg Consumption (MW)</div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-600">
            <p>Energy Forecasting Dashboard - Built with React, FastAPI, and PyTorch</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
