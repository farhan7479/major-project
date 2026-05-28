import { useEffect, useMemo, useState } from "react";
import api from "../services/api";

function fmtLocalInput(iso) {
  // FastAPI returns ISO with no timezone; HTML datetime-local needs YYYY-MM-DDTHH:MM
  return iso.slice(0, 16);
}

function shortBuilding(id) {
  return id.replace(/^Hog_office_/, "");
}

export default function Forecast() {
  const [models, setModels] = useState([]);
  const [buildings, setBuildings] = useState([]);
  const [testPeriod, setTestPeriod] = useState(null);
  const [error, setError] = useState(null);

  const [selectedModel, setSelectedModel] = useState("random_forest");
  const [selectedBuilding, setSelectedBuilding] = useState("");
  const [targetDatetime, setTargetDatetime] = useState("2017-11-15T14:00");
  const [mode, setMode] = useState("single"); // "single" or "compare"

  const [loading, setLoading] = useState(false);
  const [singleResult, setSingleResult] = useState(null);
  const [batchResult, setBatchResult] = useState(null);

  useEffect(() => {
    Promise.all([api.getModels(), api.getBuildings()])
      .then(([m, b]) => {
        setModels(m);
        setBuildings(b.buildings);
        setTestPeriod(b.testPeriod);
        if (b.buildings.length) setSelectedBuilding(b.buildings[0]);
      })
      .catch((e) => setError(`Failed to load API: ${e.message}. Is the backend on port 8000?`));
  }, []);

  const dateMin = useMemo(() => (testPeriod ? fmtLocalInput(testPeriod.start) : undefined), [testPeriod]);
  const dateMax = useMemo(() => (testPeriod ? fmtLocalInput(testPeriod.end) : undefined), [testPeriod]);

  async function runForecast() {
    setLoading(true);
    setError(null);
    setSingleResult(null);
    setBatchResult(null);
    try {
      const targetISO = `${targetDatetime}:00`;
      if (mode === "single") {
        const r = await api.forecast({
          model: selectedModel,
          buildingId: selectedBuilding,
          targetDatetime: targetISO,
        });
        setSingleResult(r);
      } else {
        const r = await api.batchForecast({
          models: models.map((m) => m.name),
          buildingId: selectedBuilding,
          targetDatetime: targetISO,
        });
        setBatchResult(r);
      }
    } catch (e) {
      setError(`Forecast failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-3xl font-bold text-gray-900">Forecast a building hour</h1>
      <p className="text-gray-600 mt-1 mb-6">
        Pick a building and an hour inside the test window, and get a prediction from any of the trained models.
      </p>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-300 text-red-700 rounded-lg">{error}</div>
      )}

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Building</label>
            <select
              value={selectedBuilding}
              onChange={(e) => setSelectedBuilding(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            >
              {buildings.map((b) => (
                <option key={b} value={b}>
                  {shortBuilding(b)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Target hour (test set)</label>
            <input
              type="datetime-local"
              value={targetDatetime}
              min={dateMin}
              max={dateMax}
              step={3600}
              onChange={(e) => setTargetDatetime(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            />
            {testPeriod && (
              <p className="text-xs text-gray-500 mt-1">
                Test period: {testPeriod.start.slice(0, 16)} → {testPeriod.end.slice(0, 16)}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-end gap-4 flex-wrap">
          <div className="flex-1 min-w-[240px]">
            <label className="block text-sm font-medium text-gray-700 mb-1">Mode</label>
            <div className="flex gap-4 text-sm">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="mode"
                  value="single"
                  checked={mode === "single"}
                  onChange={() => setMode("single")}
                />
                Single model
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="mode"
                  value="compare"
                  checked={mode === "compare"}
                  onChange={() => setMode("compare")}
                />
                Compare all models
              </label>
            </div>
          </div>
          {mode === "single" && (
            <div className="flex-1 min-w-[240px]">
              <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.display_name}
                  </option>
                ))}
              </select>
            </div>
          )}
          <button
            onClick={runForecast}
            disabled={loading || !selectedBuilding}
            className="px-5 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Predicting…" : "Forecast"}
          </button>
        </div>
      </div>

      {singleResult && (
        <SinglePredictionCard result={singleResult} />
      )}

      {batchResult && <BatchTable result={batchResult} models={models} />}
    </div>
  );
}

function SinglePredictionCard({ result }) {
  const diff = result.actual_kwh != null ? result.prediction_kwh - result.actual_kwh : null;
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-3">Prediction</h2>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Stat label="Predicted (kWh)" value={result.prediction_kwh.toFixed(2)} accent="blue" />
        {result.actual_kwh != null && (
          <Stat label="Actual (kWh)" value={result.actual_kwh.toFixed(2)} accent="green" />
        )}
        {diff != null && (
          <Stat
            label="Error (kWh)"
            value={(diff >= 0 ? "+" : "") + diff.toFixed(2)}
            accent={Math.abs(diff) < 5 ? "green" : Math.abs(diff) < 15 ? "amber" : "red"}
          />
        )}
      </div>
      <p className="text-sm text-gray-500 mt-4">
        Model: <span className="font-medium">{result.model}</span> · Building:{" "}
        <span className="font-medium">{shortBuilding(result.building_id)}</span> · Hour:{" "}
        <span className="font-medium">{result.target_datetime.slice(0, 16).replace("T", " ")}</span>
      </p>
    </div>
  );
}

function BatchTable({ result, models }) {
  const displayName = (name) => models.find((m) => m.name === name)?.display_name ?? name;
  const rows = Object.entries(result.predictions).map(([name, pred]) => ({
    name,
    pred,
    diff: result.actual_kwh != null ? pred - result.actual_kwh : null,
  }));
  rows.sort((a, b) => (a.diff == null ? 0 : Math.abs(a.diff) - Math.abs(b.diff)));

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-3">
        All models for {shortBuilding(result.building_id)} at{" "}
        {result.target_datetime.slice(0, 16).replace("T", " ")}
      </h2>
      {result.actual_kwh != null && (
        <p className="text-sm text-gray-600 mb-3">
          Actual consumption: <span className="font-semibold">{result.actual_kwh.toFixed(2)} kWh</span> · sorted by
          accuracy
        </p>
      )}
      <table className="w-full text-sm">
        <thead className="text-left text-gray-500 border-b border-gray-200">
          <tr>
            <th className="py-2 font-medium">Model</th>
            <th className="py-2 font-medium text-right">Predicted (kWh)</th>
            <th className="py-2 font-medium text-right">Error (kWh)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className="border-b border-gray-100">
              <td className="py-2">{displayName(r.name)}</td>
              <td className="py-2 text-right font-mono">{r.pred.toFixed(2)}</td>
              <td
                className={`py-2 text-right font-mono ${
                  r.diff == null
                    ? ""
                    : Math.abs(r.diff) < 5
                    ? "text-green-700"
                    : Math.abs(r.diff) < 15
                    ? "text-amber-700"
                    : "text-red-700"
                }`}
              >
                {r.diff == null ? "—" : (r.diff >= 0 ? "+" : "") + r.diff.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Stat({ label, value, accent }) {
  const colors = {
    blue: "bg-blue-50 text-blue-700",
    green: "bg-green-50 text-green-700",
    amber: "bg-amber-50 text-amber-700",
    red: "bg-red-50 text-red-700",
  };
  return (
    <div className={`p-4 rounded-md ${colors[accent] || "bg-gray-50 text-gray-800"}`}>
      <div className="text-xs uppercase tracking-wide opacity-75">{label}</div>
      <div className="text-2xl font-bold font-mono mt-1">{value}</div>
    </div>
  );
}
