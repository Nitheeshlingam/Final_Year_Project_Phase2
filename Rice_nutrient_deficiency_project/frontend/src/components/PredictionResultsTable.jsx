import React from "react";

export default function PredictionResultsTable({ results, loading, error }) {
  if (loading) {
    return (
      <div className="bg-white rounded-lg border-2 border-gray-200 p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Prediction Results</h3>
        <div className="flex flex-col items-center justify-center py-12">
          {/* Interactive Loading Animation */}
          <div className="relative w-24 h-24 mb-4">
            <div className="absolute inset-0 rounded-full border-4 border-gray-200"></div>
            <div 
              className="absolute inset-0 rounded-full border-4 border-t-transparent animate-spin"
              style={{ borderColor: '#3cb371', borderTopColor: 'transparent' }}
            ></div>
            <div className="absolute inset-2 rounded-full bg-green-50 flex items-center justify-center">
              <svg className="w-8 h-8 animate-pulse" style={{ color: '#3cb371' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <p className="text-base font-semibold mb-1" style={{ color: '#3cb371' }}>Analyzing Image</p>
          <p className="text-sm text-gray-500">Please wait while we process your image...</p>
          
          {/* Progress Dots */}
          <div className="flex gap-2 mt-4">
            <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: '#3cb371', animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: '#3cb371', animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: '#3cb371', animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg border-2 border-gray-200 p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Prediction Results</h3>
        <div className="p-4 bg-red-50 border-l-4 border-red-500 rounded">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-white rounded-lg border-2 border-gray-200 p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Prediction Results</h3>
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <div className="text-5xl mb-3">🌾</div>
          <p className="text-gray-500">Upload an image to see predictions</p>
        </div>
      </div>
    );
  }

  const { 
    ensemble_prediction, 
    ensemble_confidence, 
    model_agreement,
    individual_predictions, 
    individual_confidences, 
    individual_probabilities,
    image_info 
  } = results;

  return (
    <div className="bg-white rounded-lg border-2 border-gray-200 p-6 shadow-md">
      <h3 className="text-lg font-semibold mb-4 text-gray-800">Prediction Results</h3>
      
      {/* Ensemble Prediction */}
      {ensemble_prediction && (
        <div className="mb-4 p-4 rounded-lg border-2 shadow-sm" style={{ backgroundColor: '#f0f9f4', borderColor: '#3cb371' }}>
          <h4 className="text-sm font-semibold text-gray-600 mb-1">Ensemble Prediction</h4>
          <div className="text-2xl font-bold" style={{ color: '#3cb371' }}>
            {ensemble_prediction}
          </div>
          <div className="text-sm font-medium text-gray-600 mt-1">
            Confidence: {(ensemble_confidence * 100).toFixed(1)}%
          </div>
          {model_agreement && (
            <div className="text-xs text-gray-500 mt-1">
              Model Agreement: {(model_agreement.agreement_score * 100).toFixed(1)}%
            </div>
          )}
        </div>
      )}
      
      {image_info && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600">
            <strong>File:</strong> {image_info.filename} | 
            <strong> Size:</strong> {image_info.shape[1]} × {image_info.shape[0]}px
          </p>
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-sm">
          <thead style={{ backgroundColor: '#3cb371' }}>
            <tr>
              <th className="text-left py-3 px-3 font-semibold text-white">Model</th>
              <th className="text-center py-3 px-3 font-semibold text-white">Prediction</th>
              <th className="text-center py-3 px-3 font-semibold text-white">Confidence</th>
              <th className="text-center py-3 px-3 font-semibold text-white">Details</th>
            </tr>
          </thead>
          <tbody className="bg-white">
            {individual_predictions && Object.entries(individual_predictions).map(([modelName, prediction], idx) => (
              <tr key={modelName} className={`border-b border-gray-100 hover:bg-gray-50 ${idx % 2 === 0 ? 'bg-gray-50' : ''}`}>
                <td className="py-3 px-3 font-medium text-gray-800">{modelName}</td>
                <td className="text-center py-3 px-3">
                  <span className={`font-semibold ${
                    prediction === ensemble_prediction 
                      ? "px-2 py-1 rounded" 
                      : ""
                  }`} style={prediction === ensemble_prediction ? { backgroundColor: '#3cb371', color: 'white' } : { color: '#374151' }}>
                    {prediction || "N/A"}
                  </span>
                </td>
                <td className="text-center py-3 px-3">
                  {individual_confidences?.[modelName] !== undefined ? (
                    <span className={`font-medium ${
                      individual_confidences[modelName] >= 0.8 
                        ? "text-green-600" 
                        : individual_confidences[modelName] >= 0.6 
                        ? "text-blue-600" 
                        : "text-orange-600"
                    }`}>
                      {(individual_confidences[modelName] * 100).toFixed(1)}%
                    </span>
                  ) : (
                    <span className="text-gray-400">N/A</span>
                  )}
                </td>
                <td className="text-center py-3 px-3">
                  {individual_probabilities?.[modelName] ? (
                    <div className="text-xs text-gray-600">
                      {Object.entries(individual_probabilities[modelName]).map(([nutrient, prob]) => (
                        <div key={nutrient}>
                          {nutrient}: {(prob * 100).toFixed(1)}%
                        </div>
                      ))}
                    </div>
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 text-xs text-gray-600 space-y-1">
        <p><span className="inline-block w-3 h-3 rounded-full bg-green-500 mr-2"></span>High confidence (≥80%)</p>
        <p><span className="inline-block w-3 h-3 rounded-full bg-blue-500 mr-2"></span>Medium confidence (60-79%)</p>
        <p><span className="inline-block w-3 h-3 rounded-full bg-orange-500 mr-2"></span>Lower confidence (&lt;60%)</p>
      </div>
    </div>
  );
}