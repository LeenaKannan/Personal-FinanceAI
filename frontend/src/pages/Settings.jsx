import React, { useState } from "react";
import { Settings as SettingsIcon, Bell, Shield, Target } from "lucide-react";

const Settings = () => {
  const [notifications, setNotifications] = useState(true);
  const [smsAlerts, setSmsAlerts] = useState(false);
  const [goal, setGoal] = useState("Save ₹1,00,000 this year");

  const handleSave = () => {
    // TODO: Save settings to backend
    alert("Settings saved!");
  };

  return (
    <div className="max-w-2xl mx-auto mt-12 bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center mb-6">
        <SettingsIcon className="text-gray-700 mr-3" size={36} />
        <h2 className="text-2xl font-bold text-gray-800">Settings</h2>
      </div>
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Bell className="text-blue-500" size={24} />
          <label className="flex-1">
            <span className="text-gray-700">Email Notifications</span>
            <input
              type="checkbox"
              checked={notifications}
              onChange={() => setNotifications(!notifications)}
              className="ml-3 accent-green-500"
            />
          </label>
        </div>
        <div className="flex items-center gap-3">
          <Shield className="text-green-500" size={24} />
          <label className="flex-1">
            <span className="text-gray-700">SMS Alerts</span>
            <input
              type="checkbox"
              checked={smsAlerts}
              onChange={() => setSmsAlerts(!smsAlerts)}
              className="ml-3 accent-blue-500"
            />
          </label>
        </div>
        <div className="flex items-center gap-3">
          <Target className="text-purple-500" size={24} />
          <label className="flex-1">
            <span className="text-gray-700">Financial Goal</span>
            <input
              type="text"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              className="ml-3 border rounded px-3 py-2 w-full"
            />
          </label>
        </div>
      </div>
      <button
        onClick={handleSave}
        className="mt-8 bg-blue-500 text-white px-6 py-2 rounded shadow hover:bg-blue-600"
      >
        Save Settings
      </button>
    </div>
  );
};

export default Settings;
