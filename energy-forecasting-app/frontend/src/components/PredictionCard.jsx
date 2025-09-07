import React from 'react';

const PredictionCard = ({ title, prediction, isLoading, color = 'blue' }) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
    red: 'from-red-500 to-red-600',
  };

  return (
    <div className={`bg-gradient-to-r ${colorClasses[color]} rounded-lg shadow-lg p-6 text-white`}>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <div className="text-3xl font-bold">
        {isLoading ? (
          <div className="animate-pulse">
            <div className="h-8 bg-white bg-opacity-30 rounded"></div>
          </div>
        ) : prediction !== null ? (
          `${prediction.toFixed(2)} MW`
        ) : (
          'N/A'
        )}
      </div>
      <p className="text-sm opacity-90 mt-2">Next Hour Forecast</p>
    </div>
  );
};

export default PredictionCard;
