import React, { useState } from "react";
import { predictEnsemble } from "../lib/api";

export default function UploadCard({ onResultsChange, onLoadingChange }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function onSelectFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setError("");
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  }

  async function onSubmit(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    if (onLoadingChange) onLoadingChange(true);
    setError("");

    try {
      const data = await predictEnsemble(file);
      onResultsChange(data);
    } catch (err) {
      setError(err?.message || "Upload failed");
      onResultsChange(null);
    } finally {
      setLoading(false);
      if (onLoadingChange) onLoadingChange(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-emerald-500 to-green-600 px-6 py-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="text-2xl">📸</span>
            Upload Rice Leaf Image
          </h2>
          <p className="text-emerald-50 text-sm mt-1">Get instant AI-powered analysis</p>
        </div>

        <div className="p-6">
          <div className="space-y-5">
            {/* Upload Area */}
            <label
              htmlFor="dropzone-file"
              className={`relative flex flex-col items-center justify-center w-full h-56 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
                previewUrl 
                  ? 'border-emerald-300 bg-emerald-50/30' 
                  : 'border-gray-300 bg-gradient-to-br from-gray-50 to-emerald-50/20 hover:border-emerald-400 hover:bg-emerald-50/40'
              }`}
            >
              {previewUrl ? (
                <div className="relative w-full h-full p-4">
                  <img
                    src={previewUrl}
                    alt="preview"
                    className="w-full h-full object-contain rounded-lg"
                  />
                  <div className="absolute inset-0 bg-black/0 hover:bg-black/10 transition-all duration-300 rounded-lg flex items-center justify-center">
                    <span className="opacity-0 hover:opacity-100 text-white font-semibold bg-emerald-600 px-4 py-2 rounded-lg shadow-lg transition-opacity duration-300">
                      Change Image
                    </span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-6">
                  <div className="mb-4 p-4 bg-emerald-100 rounded-full">
                    <svg className="w-10 h-10 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <p className="mb-2 text-base font-semibold text-gray-700">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-gray-500">PNG, JPG, JPEG (MAX. 5MB)</p>
                </div>
              )}
              <input id="dropzone-file" type="file" accept="image/*" className="hidden" onChange={onSelectFile} />
            </label>

            {/* File Info and Button */}
            <div className="flex flex-col sm:flex-row items-center gap-3">
              {file && (
                <div className="flex-1 flex items-center gap-2 px-4 py-2 bg-emerald-50 border border-emerald-200 rounded-lg">
                  <svg className="w-5 h-5 text-emerald-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                  </svg>
                  <span className="text-sm text-gray-700 font-medium truncate">{file.name}</span>
                  <span className="text-xs text-gray-500 ml-auto">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              )}
              
              <button
                type="button"
                onClick={onSubmit}
                disabled={!file || loading}
                className={`px-8 py-3 rounded-xl font-semibold text-base transition-all duration-300 shadow-lg ${
                  !file || loading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-emerald-600 to-green-600 text-white hover:from-emerald-700 hover:to-green-700 hover:shadow-xl hover:scale-105 active:scale-95'
                }`}
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analyzing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                    </svg>
                    Analyze Image
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg flex items-start gap-3">
              <svg className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div>
                <p className="text-sm font-semibold text-red-800">Upload Error</p>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}