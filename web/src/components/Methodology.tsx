"use client";

import { motion } from "framer-motion";
import { Database, Scissors, BarChart3, Binary, Split, Brain, GitMerge, FileCheck } from "lucide-react";

const steps = [
  { icon: Database, label: "Data Collection", desc: "Raw loan data ingestion" },
  { icon: Scissors, label: "Data Cleaning", desc: "Missing values & outliers" },
  { icon: BarChart3, label: "EDA", desc: "Distribution & correlations" },
  { icon: Binary, label: "Feature Engineering", desc: "Encoding & scaling" },
  { icon: Split, label: "Train/Test Split", desc: "80/20 Stratified Split" },
  { icon: Brain, label: "Model Training", desc: "Random Forest & XGBoost" },
  { icon: GitMerge, label: "Hyperparameter Tuning", desc: "RandomizedSearchCV" },
  { icon: FileCheck, label: "Evaluation", desc: "ROC-AUC & Classification Report" },
];

export default function Methodology() {
  return (
    <section id="methodology" className="py-24 relative bg-black/50 border-y border-white/5">
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="text-center mb-16">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl font-bold tracking-tight mb-4"
          >
            Research <span className="text-gradient">Methodology</span>
          </motion.h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            A systematic approach to transforming raw financial data into a highly accurate predictive model.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {steps.map((step, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1 }}
              className="relative p-6 glass-card rounded-2xl flex flex-col items-center text-center group hover:bg-white/10 transition-colors"
            >
              <div className="w-12 h-12 rounded-full bg-primary-500/20 flex items-center justify-center mb-4 text-primary-400 group-hover:scale-110 transition-transform">
                <step.icon className="w-6 h-6" />
              </div>
              <h3 className="font-semibold text-white mb-2">{step.label}</h3>
              <p className="text-sm text-gray-400">{step.desc}</p>
              
              {/* Connector Line (except for last item in row) */}
              {(idx + 1) % 4 !== 0 && (
                <div className="hidden md:block absolute top-1/2 -right-3 w-6 h-px bg-white/20" />
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
