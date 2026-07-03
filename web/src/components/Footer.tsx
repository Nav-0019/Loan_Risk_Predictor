"use client";

import { BrainCircuit, Code, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-white/10 bg-black/80 py-12">
      <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center">
        <div className="flex items-center gap-2 mb-4 md:mb-0">
          <BrainCircuit className="w-6 h-6 text-primary-500" />
          <span className="font-semibold text-lg tracking-tight">Nav-0019 Research</span>
        </div>

        <div className="flex gap-6 text-sm text-gray-400">
          <a href="#" className="hover:text-white transition-colors flex items-center gap-1">
            <Code className="w-4 h-4" />
            GitHub
          </a>
          <a href="#" className="hover:text-white transition-colors flex items-center gap-1">
            <ExternalLink className="w-4 h-4" />
            Hugging Face
          </a>
          <a href="#" className="hover:text-white transition-colors">
            Paper
          </a>
        </div>
      </div>
      <div className="container mx-auto px-6 mt-8 pt-8 border-t border-white/5 text-center text-xs text-gray-600">
        <p>&copy; {new Date().getFullYear()} Loan Risk Predictor. All rights reserved. Version 1.0.0</p>
      </div>
    </footer>
  );
}
