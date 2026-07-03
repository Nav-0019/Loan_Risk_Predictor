"use client";

import { motion } from "framer-motion";
import { CheckCircle2, Trophy, ArrowUpRight } from "lucide-react";

const models = [
  {
    name: "Random Forest (Best)",
    accuracy: "86.5%",
    auc: "0.91",
    desc: "Provides excellent predictive power by building an ensemble of decision trees. Handled non-linear relationships best.",
    isBest: true
  },
  {
    name: "XGBoost",
    accuracy: "85.2%",
    auc: "0.89",
    desc: "Gradient boosting framework that performed well but slightly overfit on the training data compared to RF.",
    isBest: false
  },
  {
    name: "Logistic Regression",
    accuracy: "78.4%",
    auc: "0.82",
    desc: "Baseline model. Good interpretability but struggled with complex feature interactions.",
    isBest: false
  }
];

export default function ModelsPerformance() {
  return (
    <section id="models" className="py-24 relative">
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="text-center mb-16">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl font-bold tracking-tight mb-4"
          >
            Machine Learning <span className="text-gradient">Models</span>
          </motion.h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            We evaluated multiple algorithms. Random Forest emerged as the top performer with the best balance of accuracy and generalization.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {models.map((model, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.2 }}
              className={`relative p-8 rounded-2xl border transition-all hover:-translate-y-2 ${
                model.isBest 
                  ? "bg-primary-900/20 border-primary-500/50 shadow-[0_0_30px_rgba(99,102,241,0.2)]" 
                  : "glass-card hover:bg-white/5"
              }`}
            >
              {model.isBest && (
                <div className="absolute -top-4 -right-4 w-12 h-12 bg-primary-500 rounded-full flex items-center justify-center shadow-lg">
                  <Trophy className="w-6 h-6 text-white" />
                </div>
              )}
              
              <h3 className="text-xl font-bold text-white mb-4">{model.name}</h3>
              <p className="text-sm text-gray-400 mb-6 leading-relaxed h-20">
                {model.desc}
              </p>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-4 border-b border-white/10">
                  <span className="text-gray-400 text-sm">Accuracy</span>
                  <span className="font-mono text-lg font-semibold text-white">{model.accuracy}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">ROC-AUC</span>
                  <span className="font-mono text-lg font-semibold text-primary-400 flex items-center gap-1">
                    {model.auc}
                    <ArrowUpRight className="w-4 h-4" />
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
