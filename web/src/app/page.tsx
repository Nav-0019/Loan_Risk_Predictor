"use client";

import { useState } from "react";
import LoadingScreen from "@/components/LoadingScreen";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Methodology from "@/components/Methodology";
import DatasetEDA from "@/components/DatasetEDA";
import ModelsPerformance from "@/components/ModelsPerformance";
import LivePrediction from "@/components/LivePrediction";
import Footer from "@/components/Footer";

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);

  return (
    <main className="min-h-screen bg-black text-white selection:bg-primary-500/30 font-sans">
      {isLoading ? (
        <LoadingScreen onComplete={() => setIsLoading(false)} />
      ) : (
        <>
          <Navbar />
          <Hero />
          
          <div id="intro" className="py-24 relative">
            <div className="container mx-auto px-6 max-w-4xl text-center">
              <h2 className="text-3xl font-bold tracking-tight mb-6">The Research Gap</h2>
              <p className="text-lg text-gray-400 leading-relaxed mb-8">
                Traditional credit scoring models often fail to capture complex, non-linear relationships in borrower behavior. 
                This research bridges that gap by implementing advanced machine learning architectures—such as Gradient Boosting and Random Forests—to significantly enhance default prediction accuracy and reduce financial risk.
              </p>
            </div>
          </div>

          <DatasetEDA />
          <Methodology />
          <ModelsPerformance />
          <LivePrediction />
          <Footer />
        </>
      )}
    </main>
  );
}
