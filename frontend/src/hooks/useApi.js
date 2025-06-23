import { useState, useEffect } from 'react';
import { fetchUserProfile, fetchTransactions, fetchInsights } from '../utils/api';

export const useUserProfile = () => {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUserProfile().then(data => {
      setProfile(data);
      setLoading(false);
    });
  }, []);

  return { profile, loading };
};

export const useTransactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTransactions().then(data => {
      setTransactions(data);
      setLoading(false);
    });
  }, []);

  return { transactions, loading };
};
