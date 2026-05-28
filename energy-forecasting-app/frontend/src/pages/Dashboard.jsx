import { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import api from "../services/api";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const DISPLAY_METRICS = [
  { key: "mae", label: "MAE (kWh)", lowerIsBetter: true },
  { key: "rmse", label: "RMSE (kWh)", lowerIsBetter: true },
  { key: "mape", label: "MAPE (%)", lowerIsBetter: true },
  { key: "r2", label: "R²", lowerIsBetter: false },
  { key: "peak_f1", label: "Peak F1", lowerIsBetter: false },
];

const COLORS = {
  random_forest: "#2e8b57",
  xgboost: "#d2691e",
  transformer: "#117a65",
  arima: "#3a6ea5",
  cnn_lstm: "#6c3483",
  lstm: "#9b59b6",
  bilstm: "#8e44ad",
  naive_last: "#aaaaaa",
  naive_seasonal_24h: "#cccccc",
};

function modelDisplay(name) {
  return name
    .replace("naive_last", "Naive (last)")
    .replace("naive_seasonal_24h", "Naive (24h)")
    .replace("random_forest", "Random Forest")
    .replace("xgboost", "XGBoost")
    .replace("arima", "ARIMA")
    .replace("cnn_lstm", "CNN-LSTM")
    .replace("transformer", "Transformer")
    .replace(/^lstm$/, "LSTM")
    .replace(/^bilstm$/, "BiLSTM");
}

export default function Dashboard() {
  const [leaderboard, setLeaderboard] = useState([]);
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([api.getMetrics(), api.getModels()])
      .then(([lb, m]) => {
        setLeaderboard(lb);
        setModels(m);
      })
      .catch((e) => setError(`Failed to load metrics: ${e.message}. Is the backend on port 8000?`));
  }, []);

  if (error) {
    return (
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="p-4 bg-red-50 border border-red-300 text-red-700 rounded-lg">{error}</div>
      </div>
    );
  }

  const sortedByMAE = [...leaderboard].sort((a, b) => a.mae - b.mae);
  const winners = {};
  for (const { key, lowerIsBetter } of DISPLAY_METRICS) {
    if (!leaderboard.length) continue;
    const best = leaderboard.reduce((acc, r) =>
      acc == null
        ? r
        : lowerIsBetter
        ? r[key] < acc[key] ? r : acc
        : r[key] > acc[key] ? r : acc,
    null);
    winners[key] = best?.model;
  }

  const chartData = {
    labels: sortedByMAE.map((r) => modelDisplay(r.model)),
    datasets: [
      {
        label: "Test MAE (kWh)",
        data: sortedByMAE.map((r) => r.mae),
        backgroundColor: sortedByMAE.map((r) => COLORS[r.model] || "#888"),
      },
    ],
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-3xl font-bold text-gray-900">Model dashboard</h1>
      <p className="text-gray-600 mt-1 mb-6">
        Test-set performance for every trained model, macro-averaged across the 3 office buildings.
      </p>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Comparison (lower MAE is better)</h2>
        <Bar
          data={chartData}
          options={{
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, title: { display: true, text: "MAE (kWh)" } } },
          }}
        />
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6 overflow-x-auto">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Leaderboard</h2>
        <table className="w-full text-sm">
          <thead className="text-left text-gray-500 border-b border-gray-200">
            <tr>
              <th className="py-2 font-medium">Model</th>
              {DISPLAY_METRICS.map((m) => (
                <th key={m.key} className="py-2 font-medium text-right">
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedByMAE.map((r) => (
              <tr key={r.model} className="border-b border-gray-100">
                <td className="py-2 font-medium">{modelDisplay(r.model)}</td>
                {DISPLAY_METRICS.map((m) => {
                  const isWinner = winners[m.key] === r.model;
                  return (
                    <td
                      key={m.key}
                      className={`py-2 text-right font-mono ${isWinner ? "text-green-700 font-semibold" : ""}`}
                    >
                      {Number(r[m.key]).toFixed(m.key === "r2" ? 3 : 2)}
                      {isWinner ? " ★" : ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-gray-500 mt-3">★ marks the winner per metric.</p>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Available models</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((m) => (
            <div key={m.name} className="border border-gray-200 rounded-md p-4">
              <div
                className="inline-block w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: COLORS[m.name] || "#888" }}
              />
              <span className="font-semibold text-gray-900">{m.display_name}</span>
              <div className="text-xs uppercase tracking-wide text-gray-500 mt-1">{m.kind}</div>
              <div className="text-xs text-gray-500 font-mono mt-1">{m.name}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
