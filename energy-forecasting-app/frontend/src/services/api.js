import axios from "axios";

const client = axios.create({
  baseURL: "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
});

export const api = {
  health: () => client.get("/health").then((r) => r.data),
  getModels: () => client.get("/models").then((r) => r.data.models),
  getBuildings: () =>
    client.get("/buildings").then((r) => ({
      buildings: r.data.buildings,
      testPeriod: r.data.test_period,
    })),
  getMetrics: () => client.get("/metrics").then((r) => r.data.leaderboard),
  forecast: ({ model, buildingId, targetDatetime, weatherOverride }) =>
    client
      .post("/forecast", {
        model,
        building_id: buildingId,
        target_datetime: targetDatetime,
        weather_override: weatherOverride ?? undefined,
      })
      .then((r) => r.data),
  batchForecast: ({ models, buildingId, targetDatetime, weatherOverride }) =>
    client
      .post("/batch-forecast", {
        models,
        building_id: buildingId,
        target_datetime: targetDatetime,
        weather_override: weatherOverride ?? undefined,
      })
      .then((r) => r.data),
};

export default api;
