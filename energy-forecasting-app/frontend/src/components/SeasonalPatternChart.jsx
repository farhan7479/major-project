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

const SeasonalPatternChart = ({ data, statistics, title = "Seasonal Energy Patterns" }) => {
  if (!data || !statistics) return <div className="text-gray-500">No seasonal data available</div>;

  // Group data by hour of day
  const hourlyAverages = Array(24).fill(0);
  const hourlyCounts = Array(24).fill(0);

  data.forEach(point => {
    if (point.hour !== undefined && point.consumption !== undefined) {
      hourlyAverages[point.hour] += point.consumption;
      hourlyCounts[point.hour]++;
    }
  });

  // Calculate averages
  for (let i = 0; i < 24; i++) {
    if (hourlyCounts[i] > 0) {
      hourlyAverages[i] /= hourlyCounts[i];
    }
  }

  // Group by day of week
  const weeklyAverages = Array(7).fill(0);
  const weeklyCounts = Array(7).fill(0);
  const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  data.forEach(point => {
    if (point.dayofweek !== undefined && point.consumption !== undefined) {
      weeklyAverages[point.dayofweek] += point.consumption;
      weeklyCounts[point.dayofweek]++;
    }
  });

  // Calculate weekly averages
  for (let i = 0; i < 7; i++) {
    if (weeklyCounts[i] > 0) {
      weeklyAverages[i] /= weeklyCounts[i];
    }
  }

  const hourlyData = {
    labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
    datasets: [
      {
        label: 'Average Hourly Consumption',
        data: hourlyAverages,
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.2)',
        tension: 0.3,
      },
    ],
  };

  const weeklyData = {
    labels: dayNames,
    datasets: [
      {
        label: 'Average Daily Consumption',
        data: weeklyAverages,
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        tension: 0.3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Energy Consumption Patterns',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Energy Consumption (kWh)'
        }
      },
    },
  };

  const seasonalStats = statistics.seasonal_averages || {};

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-md font-medium mb-2">Hourly Pattern</h4>
          <Line data={hourlyData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {display: false}}}} />
        </div>
        
        <div>
          <h4 className="text-md font-medium mb-2">Weekly Pattern</h4>
          <Line data={weeklyData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {display: false}}}} />
        </div>
      </div>

      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold text-gray-800 mb-3">Seasonal Statistics</h4>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          {Object.entries(seasonalStats).map(([season, avg]) => (
            <div key={season} className="text-center">
              <div className="text-gray-600 capitalize">{season}</div>
              <div className="font-mono text-lg">{avg.toFixed(1)}</div>
              <div className="text-xs text-gray-500">kWh avg</div>
            </div>
          ))}
        </div>
        
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-4 text-sm">
          <div className="text-center">
            <div className="text-gray-600">Peak Hours</div>
            <div className="font-mono">{(statistics.peak_hours || []).join(', ')}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-600">Avg Consumption</div>
            <div className="font-mono">{statistics.mean_consumption?.toFixed(1) || 'N/A'} kWh</div>
          </div>
          <div className="text-center">
            <div className="text-gray-600">Min/Max</div>
            <div className="font-mono">{statistics.min_consumption?.toFixed(1) || 'N/A'} / {statistics.max_consumption?.toFixed(1) || 'N/A'}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-600">Std Deviation</div>
            <div className="font-mono">{statistics.std_consumption?.toFixed(1) || 'N/A'}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SeasonalPatternChart;
