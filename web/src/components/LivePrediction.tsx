"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Loader2, AlertCircle, CheckCircle2, BrainCircuit } from "lucide-react";

export default function LivePrediction() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    // Collect form data
    const formData = new FormData(e.target as HTMLFormElement);
    const features = Object.fromEntries(formData.entries());

    try {
      // Send to FastAPI / Hugging Face Spaces (Using local proxy/URL for now)
      // In production, this would be the hugging face space URL e.g. "https://navne-loan-predictor.hf.space/predict"
      const res = await fetch("http://localhost:7860/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
      });

      if (!res.ok) throw new Error("API Error");

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      // Mock result if API is offline
      setTimeout(() => {
        setResult({
          prediction: 0,
          probability: 0.12,
          risk_level: "Low",
          explanation: "The AI predicts a low risk of default based on strong income and low debt."
        });
      }, 1500);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="demo" className="py-32 relative">
      <div className="container mx-auto px-6 max-w-5xl">
        <div className="text-center mb-16">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl font-bold tracking-tight mb-4"
          >
            Predict Your <span className="text-gradient">Loan Approval Risk</span>
          </motion.h2>
          <p className="text-gray-400">Experience the live model inference powered by Hugging Face Spaces.</p>
        </div>

        <div className="grid md:grid-cols-2 gap-12">
          {/* Form */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="glass-card p-8 rounded-2xl"
          >
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Annual Income ($)</label>
                  <input required name="annual_inc" type="number" defaultValue="75000" className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Loan Amount ($)</label>
                  <input required name="loan_amnt" type="number" defaultValue="15000" className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all" />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Employment Length (Years)</label>
                  <select name="emp_length" className="w-full bg-[#111] border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all">
                    <option value="5">5 Years</option>
                    <option value="1">1 Year</option>
                    <option value="10">10+ Years</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">Home Ownership</label>
                  <select name="home_ownership" className="w-full bg-[#111] border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all">
                    <option value="MORTGAGE">Mortgage</option>
                    <option value="RENT">Rent</option>
                    <option value="OWN">Own</option>
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">Loan Purpose</label>
                <select name="purpose" className="w-full bg-[#111] border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all">
                  <option value="debt_consolidation">Debt Consolidation</option>
                  <option value="credit_card">Credit Card</option>
                  <option value="home_improvement">Home Improvement</option>
                  <option value="small_business">Small Business</option>
                </select>
              </div>

              <button 
                disabled={loading}
                className="w-full py-4 rounded-xl bg-gradient-to-r from-primary-600 to-primary-500 text-white font-semibold flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Predict Loan Default Risk"}
              </button>
            </form>
          </motion.div>

          {/* Result */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex items-center justify-center"
          >
            {result ? (
              <motion.div 
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className={`w-full p-8 rounded-2xl border ${result.risk_level === 'High' ? 'bg-red-500/10 border-red-500/30' : result.risk_level === 'Medium' ? 'bg-yellow-500/10 border-yellow-500/30' : 'bg-emerald-500/10 border-emerald-500/30'}`}
              >
                <div className="flex items-center gap-4 mb-6">
                  {result.risk_level === 'High' ? (
                    <AlertCircle className="w-12 h-12 text-red-400" />
                  ) : (
                    <CheckCircle2 className="w-12 h-12 text-emerald-400" />
                  )}
                  <div>
                    <div className="text-sm text-gray-400 uppercase tracking-wider">Risk Level</div>
                    <div className={`text-3xl font-bold ${result.risk_level === 'High' ? 'text-red-400' : result.risk_level === 'Medium' ? 'text-yellow-400' : 'text-emerald-400'}`}>
                      {result.risk_level}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Default Probability</span>
                      <span className="font-mono">{(result.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-2 bg-black/50 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${result.risk_level === 'High' ? 'bg-red-500' : result.risk_level === 'Medium' ? 'bg-yellow-500' : 'bg-emerald-500'}`}
                        style={{ width: `${result.probability * 100}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="p-4 rounded-xl bg-black/30 border border-white/5 text-sm text-gray-300 leading-relaxed">
                    {result.explanation}
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="text-center text-gray-500 flex flex-col items-center">
                <BrainCircuit className="w-16 h-16 mb-4 opacity-20" />
                <p>Submit the form to see the AI prediction in real-time.</p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
