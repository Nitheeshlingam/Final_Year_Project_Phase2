import React, { useState, useEffect } from "react";

// Mock API function for demo purposes
const getTrainingAccuracies = async () => {
  await new Promise(resolve => setTimeout(resolve, 1000));
  return {
    "Random Forest": { Nitrogen: 75.5, Phosphorus: 84.3, Potassium: 81.7, Overall: 80.5 },
    "SVM": { Nitrogen: 89.2, Phosphorus: 85.6, Potassium: 87.4, Overall: 87.4 },
    "XGBoost": { Nitrogen: 88.7, Phosphorus: 89.1, Potassium: 94.3, Overall: 90.7 },
    "EfficientNetB0": { Nitrogen: 94.1, Phosphorus: 91.8, Potassium: 93.2, Overall: 93.0 }
  };
};

export default function TrainingAccuracyTable() {
  const [accuracies, setAccuracies] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [hoveredRow, setHoveredRow] = useState(null);
  const [hoveredCol, setHoveredCol] = useState(null);

  useEffect(() => {
    async function fetchAccuracies() {
      try {
        const data = await getTrainingAccuracies();
        setAccuracies(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchAccuracies();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-900 via-green-800 to-teal-900 p-8 flex items-center justify-center">
        <div className="bg-gradient-to-br from-emerald-50 to-green-100 rounded-2xl border-2 border-emerald-300 p-8 shadow-2xl max-w-6xl w-full">
          <h3 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-emerald-700 to-green-600">
            Training Accuracies
          </h3>
          <div className="flex items-center justify-center py-12">
            <div className="relative">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-emerald-200"></div>
              <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-emerald-600 absolute top-0 left-0"></div>
            </div>
            <span className="ml-4 text-emerald-700 font-semibold text-lg">Loading accuracies...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-900 via-green-800 to-teal-900 p-8 flex items-center justify-center">
        <div className="bg-gradient-to-br from-emerald-50 to-green-100 rounded-2xl border-2 border-emerald-300 p-8 shadow-2xl max-w-6xl w-full">
          <h3 className="text-2xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-emerald-700 to-green-600">
            Training Accuracies
          </h3>
          <div className="p-6 bg-gradient-to-r from-red-50 to-orange-50 border-l-4 border-red-500 rounded-lg shadow-md">
            <p className="text-red-700 font-medium">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const models = Object.keys(accuracies);
  const nutrients = ["Nitrogen", "Phosphorus", "Potassium", "Overall"];

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-900 via-green-800 to-teal-900 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h2 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-200 to-green-300 mb-2">
            Model Performance Dashboard
          </h2>
          <p className="text-emerald-200 text-lg">Training accuracy metrics across nutrients</p>
        </div>

        <div className="overflow-hidden shadow-2xl rounded-2xl border-2 border-emerald-400 bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gradient-to-r from-emerald-600 via-green-600 to-emerald-700 text-white">
                  <th
                    scope="col"
                    className="px-8 py-5 text-left font-bold uppercase tracking-wider text-base border-r border-emerald-500"
                  >
                    Model
                  </th>
                  {nutrients.map((nutrient, idx) => (
                    <th
                      key={nutrient}
                      scope="col"
                      className={`px-8 py-5 text-center font-bold uppercase tracking-wider text-base transition-all duration-300 ${hoveredCol === idx ? 'bg-emerald-700 scale-105' : ''
                        } ${idx < nutrients.length - 1 ? 'border-r border-emerald-500' : ''}`}
                      onMouseEnter={() => setHoveredCol(idx)}
                      onMouseLeave={() => setHoveredCol(null)}
                    >
                      {nutrient}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map((model, rowIdx) => (
                  <tr
                    key={model}
                    className={`border-b-2 border-emerald-200 transition-all duration-300 ${hoveredRow === rowIdx
                        ? 'bg-gradient-to-r from-emerald-100 via-green-100 to-emerald-100 shadow-lg scale-[1.02] z-10'
                        : 'bg-white hover:bg-gradient-to-r hover:from-emerald-50 hover:to-green-50'
                      }`}
                    onMouseEnter={() => setHoveredRow(rowIdx)}
                    onMouseLeave={() => setHoveredRow(null)}
                  >
                    <th
                      scope="row"
                      className={`px-8 py-5 font-bold text-emerald-900 whitespace-nowrap text-left border-r-2 border-emerald-200 transition-all duration-300 ${hoveredRow === rowIdx ? 'text-emerald-700 text-lg' : 'text-base'
                        }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full transition-all duration-300 ${hoveredRow === rowIdx
                            ? 'bg-emerald-600 shadow-lg shadow-emerald-400'
                            : 'bg-emerald-400'
                          }`}></div>
                        {model}
                      </div>
                    </th>
                    {nutrients.map((nutrient, colIdx) => {
                      const accuracy = accuracies[model][nutrient];
                      const isOverall = nutrient === "Overall";
                      const isHighAccuracy = accuracy >= 90;
                      const isMediumAccuracy = accuracy >= 80 && accuracy < 90;

                      const colorClass = isHighAccuracy
                        ? 'text-emerald-700 font-bold'
                        : isMediumAccuracy
                          ? 'text-green-600 font-semibold'
                          : 'text-green-500 font-medium';

                      const bgClass = isHighAccuracy
                        ? 'bg-emerald-100'
                        : isMediumAccuracy
                          ? 'bg-green-50'
                          : '';

                      return (
                        <td
                          key={nutrient}
                          className={`px-8 py-5 text-center transition-all duration-300 ${hoveredRow === rowIdx || hoveredCol === colIdx
                              ? 'bg-gradient-to-br from-emerald-100 to-green-100 scale-110'
                              : bgClass
                            } ${colIdx < nutrients.length - 1 ? 'border-r border-emerald-100' : ''}`}
                        >
                          <div className={`inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${hoveredRow === rowIdx || hoveredCol === colIdx
                              ? 'shadow-md bg-white'
                              : ''
                            }`}>
                            <span className={`${colorClass} text-lg transition-all duration-300`}>
                              {Number.isFinite(accuracy) ? accuracy.toFixed(1) : '—'}%
                            </span>
                            {isHighAccuracy && (
                              <span className="text-emerald-600 text-xl">✓</span>
                            )}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-6 flex items-center justify-center gap-8 text-emerald-100">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-emerald-700"></div>
            <span className="font-medium">≥ 90% Accuracy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-green-600"></div>
            <span className="font-medium">80-90% Accuracy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-green-500"></div>
            <span className="font-medium">&lt; 80% Accuracy</span>
          </div>
        </div>
      </div>
    </div>
  );
}