import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import Forecast from "./pages/Forecast";
import Dashboard from "./pages/Dashboard";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-gray-900">Load Forecasting</h1>
              <p className="text-xs text-gray-500">AI-based building electricity demand prediction</p>
            </div>
            <nav className="flex gap-6 text-sm">
              <NavTab to="/forecast">Forecast</NavTab>
              <NavTab to="/dashboard">Dashboard</NavTab>
            </nav>
          </div>
        </header>

        <main>
          <Routes>
            <Route path="/" element={<Navigate to="/forecast" replace />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>

        <footer className="border-t border-gray-200 bg-white mt-8">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 text-center text-xs text-gray-500">
            DTU B.Tech major project — FastAPI + React + PyTorch
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}

function NavTab({ to, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `pb-1 border-b-2 transition-colors ${
          isActive
            ? "border-blue-600 text-blue-700 font-medium"
            : "border-transparent text-gray-600 hover:text-gray-900"
        }`
      }
    >
      {children}
    </NavLink>
  );
}
