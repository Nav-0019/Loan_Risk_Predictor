"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { BrainCircuit } from "lucide-react";

export default function LoadingScreen({ onComplete }: { onComplete: () => void }) {
  const [progress, setProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          setTimeout(() => {
            setIsVisible(false);
            setTimeout(onComplete, 800); // Wait for exit animation
          }, 500);
          return 100;
        }
        return prev + 2;
      });
    }, 30);
    return () => clearInterval(timer);
  }, [onComplete]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ y: "100%" }}
          animate={{ y: 0 }}
          exit={{ y: "-100%", opacity: 0, filter: "blur(10px)" }}
          transition={{ duration: 0.8, ease: [0.76, 0, 0.24, 1] }}
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black text-white"
        >
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
             <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary-600/20 rounded-full blur-[120px]" />
          </div>

          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="flex flex-col items-center gap-6 z-10"
          >
            <div className="relative p-6 glass-card rounded-2xl">
              <BrainCircuit className="w-16 h-16 text-primary-400" />
            </div>
            
            <div className="text-center">
              <h2 className="text-2xl font-semibold tracking-tight text-gradient">Neural Core Initializing</h2>
              <p className="text-gray-400 mt-2 text-sm uppercase tracking-widest">Loading Predictor Model</p>
            </div>

            <div className="w-64 h-1 bg-white/10 rounded-full overflow-hidden mt-8">
              <motion.div 
                className="h-full bg-gradient-to-r from-primary-600 to-primary-400"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-xs text-gray-500 font-mono mt-2">{progress}%</div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
