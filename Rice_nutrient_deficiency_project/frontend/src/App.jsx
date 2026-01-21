import React, { useEffect, useRef, useState } from "react";
import UploadCard from "./components/UploadCard";
import TrainingAccuracyTable from "./components/TrainingAccuracyTable";
import PredictionResultsTable from "./components/PredictionResultsTable";
import { healthCheck } from "./lib/api";
import HeroCarousel from "./components/HeroCarousel"; // Import the HeroCarousel

export default function App() {
  const [apiOk, setApiOk] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState("");

  // Section refs
  const homeRef = useRef(null);
  const analyzeRef = useRef(null);
  const aboutRef = useRef(null);
  const contactRef = useRef(null);

  const sections = { home: homeRef, analyze: analyzeRef, about: aboutRef, contact: contactRef };

  useEffect(() => {
    healthCheck()
      .then((r) => setApiOk(r?.status === "ok"))
      .catch(() => setApiOk(false));
  }, []);

  const handleResultsChange = (results) => {
    setPredictionResults(results);
    setPredictionLoading(false);
    setPredictionError("");
  };

  const handleLoadingChange = (isLoading) => {
    setPredictionLoading(isLoading);
  };

  const scrollToSection = (section) => {
    sections[section]?.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen m-0 p-0 bg-white">
      {/* Fixed Navbar */}
      <nav className="fixed top-0 w-full z-50 bg-gradient-to-r from-emerald-600 via-green-600 to-emerald-700 shadow-lg backdrop-blur-sm border-b border-white/10">
        <div className="w-full px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-xl bg-white/95 grid place-items-center text-2xl shadow-md transform transition-all duration-300 hover:scale-110 hover:rotate-6 hover:shadow-xl">
              🌱
            </div>
            <h1 className="text-white font-bold text-sm md:text-base lg:text-xl tracking-wide drop-shadow-md">
              AI-Enhanced Identification of Rice Leaf Nutrient Deficiencies
            </h1>
          </div>
          <div
            className={`px-4 py-2 text-xs rounded-full font-semibold transition-all duration-300 shadow-md ${
              apiOk
                ? "bg-white/20 text-white backdrop-blur-sm border border-white/30"
                : "bg-amber-100 text-amber-900 border border-amber-300"
            }`}
          >
            <span className={`inline-block mr-1 ${apiOk ? "animate-pulse" : ""}`}>
              {apiOk == null ? "⏳" : apiOk ? "✓" : "✗"}
            </span>
            {apiOk == null ? "Checking..." : apiOk ? "Online" : "Offline"}
          </div>
        </div>
        {/* Navbar Tabs */}
        <div className="flex gap-2 px-4 pb-3">
          {["home", "analyze", "about", "contact"].map((tab) => (
            <button
              key={tab}
              onClick={() => scrollToSection(tab)}
              className="px-4 py-2 rounded-lg font-medium text-sm text-white/90 hover:bg-white/10 hover:text-white transition-all duration-300"
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </nav>

      {/* Home / Carousel Section */}
      <section ref={homeRef} className="relative w-full h-screen overflow-hidden pt-28">
        <HeroCarousel />
      </section>

      {/* Analyze Section */}
      <section ref={analyzeRef} className="w-full py-16 bg-gray-50" id="analyze">
  <div className="max-w-6xl mx-auto px-4">
    <UploadCard onResultsChange={handleResultsChange} onLoadingChange={handleLoadingChange} />
    <div className="grid grid-cols-1  gap-6 mt-8">
      {/* Prediction table comes first */}
      <PredictionResultsTable
        results={predictionResults}
        loading={predictionLoading}
        error={predictionError}
      />
      {/* Training table comes second */}
      <TrainingAccuracyTable />
    </div>
  </div>
</section>


      {/* About Section */}
      <section ref={aboutRef} className="w-full py-16 bg-white">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-gray-800 mb-6">About Us</h2>
          <p className="text-gray-600 text-lg mb-4">
            We are dedicated to revolutionizing agriculture through artificial intelligence and machine learning.
            Our rice leaf analysis system helps farmers and agricultural experts identify nutrient deficiencies early.
          </p>
          <p className="text-gray-600 text-lg">
            Using state-of-the-art deep learning models, we've trained our system on thousands of rice leaf images
            to accurately detect various nutrient deficiencies.
          </p>
        </div>
      </section>

      {/* Contact Section */}
      <section ref={contactRef} className="w-full py-16 bg-gray-50">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-gray-800 mb-6">Contact Us</h2>
          <p className="text-gray-600 text-lg mb-4">
            Get in touch with our team for support, partnerships, or inquiries.
          </p>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <span className="text-2xl">📧</span>
              <span className="text-gray-700">contact@riceleafai.com</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-2xl">📱</span>
              <span className="text-gray-700">+1 (555) 123-4567</span>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="text-center text-sm text-gray-600 py-6 bg-gray-100">
        <div className="flex items-center justify-center gap-2">
          <span>Model Classes:</span>
          <span className="text-gray-700">Nitrogen</span>
          <span className="text-gray-700">Phosphorus</span>
          <span className="text-gray-700">Potassium</span>
        </div>
      </footer>
    </div>
  );
}
