import React, { useState } from "react";
import { User, MapPin, DollarSign } from "lucide-react";

const initialProfile = {
  name: "Amit Sharma",
  email: "amit.sharma@email.com",
  city: "Mumbai",
  gender: "male",
  income: 80000,
  age: 28,
};

const Profile = () => {
  const [profile, setProfile] = useState(initialProfile);
  const [editing, setEditing] = useState(false);

  const handleChange = (e) => {
    setProfile({ ...profile, [e.target.name]: e.target.value });
  };

  const handleSave = () => {
    // TODO: Save profile to backend
    setEditing(false);
  };

  return (
    <div className="max-w-2xl mx-auto mt-12 bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center mb-6">
        <User className="text-blue-500 mr-3" size={36} />
        <h2 className="text-2xl font-bold text-gray-800">My Profile</h2>
      </div>
      <form className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-gray-600 mb-1">Full Name</label>
          <input
            name="name"
            value={profile.name}
            onChange={handleChange}
            disabled={!editing}
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-gray-600 mb-1">Email</label>
          <input
            name="email"
            value={profile.email}
            onChange={handleChange}
            disabled
            className="w-full border rounded px-3 py-2 bg-gray-100"
          />
        </div>
        <div>
          <label className="block text-gray-600 mb-1 flex items-center">
            <MapPin className="mr-1" size={18} /> City
          </label>
          <select
            name="city"
            value={profile.city}
            onChange={handleChange}
            disabled={!editing}
            className="w-full border rounded px-3 py-2"
          >
            <option>Mumbai</option>
            <option>Delhi</option>
            <option>Bangalore</option>
            <option>Chennai</option>
            <option>Pune</option>
            <option>Hyderabad</option>
            <option>Kolkata</option>
            <option>Ahmedabad</option>
            <option>Jaipur</option>
            <option>Lucknow</option>
          </select>
        </div>
        <div>
          <label className="block text-gray-600 mb-1">Gender</label>
          <select
            name="gender"
            value={profile.gender}
            onChange={handleChange}
            disabled={!editing}
            className="w-full border rounded px-3 py-2"
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
        <div>
          <label className="block text-gray-600 mb-1 flex items-center">
            <DollarSign className="mr-1" size={18} /> Monthly Income (₹)
          </label>
          <input
            name="income"
            type="number"
            value={profile.income}
            onChange={handleChange}
            disabled={!editing}
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-gray-600 mb-1">Age</label>
          <input
            name="age"
            type="number"
            value={profile.age}
            onChange={handleChange}
            disabled={!editing}
            className="w-full border rounded px-3 py-2"
          />
        </div>
      </form>
      <div className="mt-6 flex gap-4">
        {editing ? (
          <>
            <button
              onClick={handleSave}
              className="bg-green-500 text-white px-4 py-2 rounded shadow hover:bg-green-600"
            >
              Save
            </button>
            <button
              onClick={() => setEditing(false)}
              className="bg-gray-200 text-gray-700 px-4 py-2 rounded shadow hover:bg-gray-300"
            >
              Cancel
            </button>
          </>
        ) : (
          <button
            onClick={() => setEditing(true)}
            className="bg-blue-500 text-white px-4 py-2 rounded shadow hover:bg-blue-600"
          >
            Edit Profile
          </button>
        )}
      </div>
    </div>
  );
};

export default Profile;
