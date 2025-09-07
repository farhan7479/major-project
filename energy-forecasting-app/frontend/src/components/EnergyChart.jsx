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
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const EnergyChart = ({ data, predictions }) => {
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Energy Consumption Forecast',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Energy Consumption (MW)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time (Hours)'
        }
      }
    }
  };

  const chartData = {
    labels: data.map((_, index) => `Hour ${index + 1}`),
    datasets: [
      {
        label: 'Historical Data',
        data: data.map(point => point.consumption),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
      ...(predictions.gru_prediction ? [{
        label: 'GRU Prediction',
        data: [...Array(data.length - 1).fill(null), predictions.gru_prediction],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        pointStyle: 'rectRot',
        pointRadius: 8,
        showLine: false,
      }] : []),
      ...(predictions.lstm_prediction ? [{
        label: 'LSTM Prediction',
        data: [...Array(data.length - 1).fill(null), predictions.lstm_prediction],
        borderColor: 'rgb(255, 205, 86)',
        backgroundColor: 'rgba(255, 205, 86, 0.2)',
        pointStyle: 'triangle',
        pointRadius: 8,
        showLine: false,
      }] : []),
    ],
  };

  return (
    <div className="w-full h-96">
      <Line options={chartOptions} data={chartData} />
    </div>
  );
};

export default EnergyChart;
