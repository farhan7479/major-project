import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ConfidenceIntervalChart = ({ historicalData, prediction, title = "Prediction with Confidence Intervals" }) => {
  if (!historicalData || !prediction) return <div className="text-gray-500">No data available</div>;

  const last24Hours = historicalData.slice(-24);
  const labels = last24Hours.map((_, index) => `Hour ${index + 1}`);
  
  // Add prediction point
  labels.push('Next Hour');

  const historicalValues = last24Hours.map(point => point.consumption);
  const predictionValue = prediction.ensemble_prediction || prediction.next_hour_forecast?.ensemble_prediction || 0;
  
  // Calculate confidence intervals
  const confidenceInterval = prediction.next_hour_forecast?.confidence_interval || {
    lower: predictionValue - 10,
    upper: predictionValue + 10
  };

  const data = {
    labels,
    datasets: [
      {
        label: 'Historical Consumption',
        data: [...historicalValues, null],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
        pointRadius: 3,
      },
      {
        label: 'Prediction',
        data: [...Array(historicalValues.length).fill(null), predictionValue],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        pointRadius: 8,
        pointStyle: 'star',
      },
      {
        label: 'Confidence Upper',
        data: [...Array(historicalValues.length).fill(null), confidenceInterval.upper],
        borderColor: 'rgba(255, 99, 132, 0.3)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        borderDash: [5, 5],
        pointRadius: 0,
        fill: '+1',
      },
      {
        label: 'Confidence Lower',
        data: [...Array(historicalValues.length).fill(null), confidenceInterval.lower],
        borderColor: 'rgba(255, 99, 132, 0.3)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: title,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time Period'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Energy Consumption (kWh)'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <Line data={data} options={options} />
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold text-gray-800">Prediction Details</h4>
        <div className="grid grid-cols-2 gap-4 mt-2 text-sm">
          <div>
            <span className="text-gray-600">Predicted Value:</span>
            <span className="ml-2 font-mono">{predictionValue.toFixed(2)} kWh</span>
          </div>
          <div>
            <span className="text-gray-600">Confidence:</span>
            <span className="ml-2 font-mono">{((prediction.next_hour_forecast?.confidence || 0.85) * 100).toFixed(1)}%</span>
          </div>
          <div>
            <span className="text-gray-600">Lower Bound:</span>
            <span className="ml-2 font-mono">{confidenceInterval.lower.toFixed(2)} kWh</span>
          </div>
          <div>
            <span className="text-gray-600">Upper Bound:</span>
            <span className="ml-2 font-mono">{confidenceInterval.upper.toFixed(2)} kWh</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceIntervalChart;
