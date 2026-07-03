"use client";

import { useState, useEffect } from "react";
import { motion, useScroll, useMotionValueEvent } from "framer-motion";
import { BrainCircuit } from "lucide-react";

export default function Navbar() {
  const { scrollY } = useScroll();
  const [scrolled, setScrolled] = useState(false);

  useMotionValueEvent(scrollY, "change", (latest) => {
    setScrolled(latest > 50);
  });

  return (
    <motion.header
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6 }}
      className={`fixed top-0 left-0 right-0 z-40 transition-all duration-300 ${
        scrolled ? "py-4 bg-black/50 backdrop-blur-md border-b border-white/10" : "py-6 bg-transparent"
      }`}
    >
      <div className="container mx-auto px-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-8 h-8 text-primary-500" />
          <span className="font-semibold text-xl tracking-tight">Nav-0019</span>
        </div>
        
        <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-300">
          <a href="#intro" className="hover:text-white transition-colors">Intro</a>
          <a href="#methodology" className="hover:text-white transition-colors">Methodology</a>
          <a href="#models" className="hover:text-white transition-colors">Models</a>
          <a href="#demo" className="px-4 py-2 rounded-lg bg-white text-black hover:bg-gray-200 transition-colors">
            Live Demo
          </a>
        </nav>
      </div>
    </motion.header>
  );
}
