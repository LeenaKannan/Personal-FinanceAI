import React from "react";
import { PiggyBank, TrendingUp, FileText } from "lucide-react";
import { Link } from "react-router-dom";

const Home = () => (
  <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex flex-col items-center justify-center">
    <div className="bg-white rounded-xl shadow-xl p-8 max-w-xl w-full text-center">
      <PiggyBank className="mx-auto text-green-500 mb-4" size={48} />
      <h1 className="text-3xl font-bold text-gray-800 mb-2">Welcome to SmartFinance AI</h1>
      <p className="text-gray-600 mb-6">
        Your AI-powered personal finance assistant for smarter money management, budgeting, and investing.
      </p>
      <div className="flex justify-center gap-4 mb-6">
        <Link
          to="/profile"
          className="bg-green-500 hover:bg-green-600 text-white px-5 py-2 rounded-lg shadow transition"
        >
          My Profile
        </Link>
        <Link
          to="/analytics"
          className="bg-blue-500 hover:bg-blue-600 text-white px-5 py-2 rounded-lg shadow transition"
        >
          Analytics
        </Link>
        <Link
          to="/settings"
          className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-5 py-2 rounded-lg shadow transition"
        >
          Settings
        </Link>
      </div>
      <div className="italic text-gray-500">
        <FileText className="inline mr-2" size={20} />
        "A budget is telling your money where to go instead of wondering where it went." – Dave Ramsey
      </div>
    </div>
    <footer className="mt-8 text-gray-400 text-sm">
      © 2025 SmartFinance AI • Made with <span className="text-red-400">♥</span> for your financial journey
    </footer>
  </div>
);

export default Home;
