import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  transactionsAPI, 
  analyticsAPI, 
  predictionsAPI, 
  investmentsAPI, 
  marketAPI,
  utilsAPI 
} from '../services/api'
import toast from 'react-hot-toast'

// Transactions hooks
export const useTransactions = (params = {}) => {
  return useQuery({
    queryKey: ['transactions', params],
    queryFn: () => transactionsAPI.getAll(params),
    select: (data) => data.data,
  })
}

export const useCreateTransaction = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: transactionsAPI.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transactions'] })
      queryClient.invalidateQueries({ queryKey: ['analytics'] })
      toast.success('Transaction added successfully')
    },
    onError: (error) => {
      toast.error(error.response?.data?.detail || 'Failed to add transaction')
    },
  })
}

export const useDeleteTransaction = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: transactionsAPI.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transactions'] })
      queryClient.invalidateQueries({ queryKey: ['analytics'] })
      toast.success('Transaction deleted successfully')
    },
    onError: (error) => {
      toast.error(error.response?.data?.detail || 'Failed to delete transaction')
    },
  })
}

export const useBulkUploadTransactions = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: transactionsAPI.bulkUpload,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['transactions'] })
      queryClient.invalidateQueries({ queryKey: ['analytics'] })
      toast.success('Transactions uploaded successfully')
    },
    onError: (error) => {
      toast.error(error.response?.data?.detail || 'Failed to upload transactions')
    },
  })
}

// Analytics hooks
export const useInsights = () => {
  return useQuery({
    queryKey: ['analytics', 'insights'],
    queryFn: analyticsAPI.getInsights,
    select: (data) => data.data.insights,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

export const useSpendingBreakdown = (period = 'month') => {
  return useQuery({
    queryKey: ['analytics', 'spending-breakdown', period],
    queryFn: () => analyticsAPI.getSpendingBreakdown(period),
    select: (data) => data.data,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export const useExpenseForecast = (months = 6) => {
  return useQuery({
    queryKey: ['analytics', 'expense-forecast', months],
    queryFn: () => analyticsAPI.getExpenseForecast(months),
    select: (data) => data.data,
    staleTime: 30 * 60 * 1000, // 30 minutes
  })
}

// Predictions hooks
export const usePredictExpenses = () => {
  return useMutation({
    mutationFn: predictionsAPI.predictExpenses,
    onError: (error) => {
      toast.error(error.response?.data?.detail || 'Failed to predict expenses')
    },
  })
}

// Investments hooks
export const useInvestments = () => {
  return useQuery({
    queryKey: ['investments'],
    queryFn: investmentsAPI.getAll,
    select: (data) => data.data,
  })
}

export const useCreateInvestment = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: investmentsAPI.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['investments'] })
      toast.success('Investment added successfully')
    },
    onError: (error) => {
      toast.error(error.response?.data?.detail || 'Failed to add investment')
    },
  })
}

// Market hooks
export const useStockInfo = (symbol) => {
  return useQuery({
    queryKey: ['market', 'stock-info', symbol],
    queryFn: () => marketAPI.getStockInfo(symbol),
    select: (data) => data.data,
    enabled: !!symbol,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export const useStockHistory = (symbol, period = '1mo') => {
  return useQuery({
    queryKey: ['market', 'stock-history', symbol, period],
    queryFn: () => marketAPI.getStockHistory(symbol, period),
    select: (data) => data.data,
    enabled: !!symbol,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

export const useInvestmentRecommendations = () => {
  return useQuery({
    queryKey: ['market', 'recommendations'],
    queryFn: marketAPI.getRecommendations,
    select: (data) => data.data.recommendations,
    staleTime: 60 * 60 * 1000, // 1 hour
  })
}

// Utilities hooks
export const useSupportedCities = () => {
  return useQuery({
    queryKey: ['utils', 'cities'],
    queryFn: utilsAPI.getSupportedCities,
    select: (data) => data.data.cities,
    staleTime: 24 * 60 * 60 * 1000, // 24 hours
  })
}

// Custom hook for transaction categorization
export const useCategorizeTransaction = () => {
  return useMutation({
    mutationFn: (description) => transactionsAPI.categorize(description),
    onError: (error) => {
      toast.error(error.response?.data?.detail || 'Failed to categorize transaction')
    },
  })
}

// Custom hook for real-time data updates
export const useRealTimeUpdates = () => {
  const queryClient = useQueryClient()
  
  const refreshAllData = () => {
    queryClient.invalidateQueries({ queryKey: ['transactions'] })
    queryClient.invalidateQueries({ queryKey: ['analytics'] })
    queryClient.invalidateQueries({ queryKey: ['investments'] })
  }
  
  return { refreshAllData }
}

// Custom hook for offline support
export const useOfflineSupport = () => {
  const queryClient = useQueryClient()
  
  const getCachedData = (queryKey) => {
    return queryClient.getQueryData(queryKey)
  }
  
  const isOnline = navigator.onLine
  
  return { getCachedData, isOnline }

}

