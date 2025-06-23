import axios from 'axios'
import toast from 'react-hot-toast'

// Create axios instance
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
      toast.error('Session expired. Please login again.')
    } else if (error.response?.status === 403) {
      toast.error('Access denied')
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.')
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Request timeout. Please check your connection.')
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authAPI = {
  login: (credentials) => api.post('/auth/login', credentials),
  register: (userData) => api.post('/auth/register', userData),
  getProfile: () => api.get('/users/me'),
  updateProfile: (profileData) => api.put('/users/me', profileData),
  refreshToken: () => api.post('/auth/refresh'),
}

// Transactions API
export const transactionsAPI = {
  getAll: (params = {}) => api.get('/transactions', { params }),
  create: (transactionData) => api.post('/transactions', transactionData),
  update: (id, transactionData) => api.put(`/transactions/${id}`, transactionData),
  delete: (id) => api.delete(`/transactions/${id}`),
  bulkUpload: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/transactions/bulk-upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
  categorize: (description) => api.post('/utils/categorize-transaction', { description }),
}

// Analytics API
export const analyticsAPI = {
  getInsights: () => api.get('/analytics/insights'),
  getSpendingBreakdown: (period = 'month') => api.get('/analytics/spending-breakdown', { params: { period } }),
  getExpenseForecast: (months = 6) => api.get('/predictions/expenses/historical', { params: { months } }),
}

// Predictions API
export const predictionsAPI = {
  predictExpenses: (profileData) => api.post('/predictions/expenses', profileData),
  getHistoricalForecast: (months = 6) => api.get('/predictions/expenses/historical', { params: { months } }),
}

// Investments API
export const investmentsAPI = {
  getAll: () => api.get('/investments'),
  create: (investmentData) => api.post('/investments', investmentData),
  update: (id, investmentData) => api.put(`/investments/${id}`, investmentData),
  delete: (id) => api.delete(`/investments/${id}`),
}

// Market API
export const marketAPI = {
  getStockInfo: (symbol) => api.get(`/market/stock/${symbol}`),
  getStockHistory: (symbol, period = '1mo') => api.get(`/market/stock/${symbol}/history`, { params: { period } }),
  getRecommendations: () => api.get('/market/recommendations'),
}

// Utilities API
export const utilsAPI = {
  getSupportedCities: () => api.get('/utils/cities'),
  seedTestData: () => api.get('/dev/seed-data'),
}

// Health check
export const healthAPI = {
  check: () => api.get('/health'),
}

export default api
