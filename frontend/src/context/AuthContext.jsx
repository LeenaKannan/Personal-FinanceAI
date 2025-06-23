import React, { createContext, useContext, useReducer, useEffect } from 'react'
import { authAPI } from '../services/api'
import toast from 'react-hot-toast'

const AuthContext = createContext()

const initialState = {
  user: null,
  token: localStorage.getItem('token'),
  isAuthenticated: false,
  isLoading: true,
  error: null,
}

const authReducer = (state, action) => {
  switch (action.type) {
    case 'AUTH_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      }
    case 'AUTH_SUCCESS':
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      }
    case 'AUTH_FAILURE':
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: action.payload,
      }
    case 'LOGOUT':
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      }
    case 'UPDATE_USER':
      return {
        ...state,
        user: { ...state.user, ...action.payload },
      }
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      }
    default:
      return state
  }
}

export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState)

  // Check if user is authenticated on app load
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token')
      if (token) {
        try {
          const response = await authAPI.getProfile()
          dispatch({
            type: 'AUTH_SUCCESS',
            payload: {
              user: response.data,
              token,
            },
          })
        } catch (error) {
          localStorage.removeItem('token')
          dispatch({
            type: 'AUTH_FAILURE',
            payload: 'Session expired. Please login again.',
          })
        }
      } else {
        dispatch({
          type: 'AUTH_FAILURE',
          payload: null,
        })
      }
    }

    checkAuth()
  }, [])

  const login = async (email, password) => {
    try {
      dispatch({ type: 'AUTH_START' })
      
      const response = await authAPI.login({ email, password })
      const { access_token, user } = response.data
      
      localStorage.setItem('token', access_token)
      
      dispatch({
        type: 'AUTH_SUCCESS',
        payload: {
          user,
          token: access_token,
        },
      })
      
      toast.success(`Welcome back, ${user.name}!`)
      return { success: true }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Login failed'
      dispatch({
        type: 'AUTH_FAILURE',
        payload: errorMessage,
      })
      toast.error(errorMessage)
      return { success: false, error: errorMessage }
    }
  }

  const register = async (userData) => {
    try {
      dispatch({ type: 'AUTH_START' })
      
      const response = await authAPI.register(userData)
      const user = response.data
      
      // Auto-login after registration
      const loginResponse = await authAPI.login({
        email: userData.email,
        password: userData.password,
      })
      
      const { access_token } = loginResponse.data
      localStorage.setItem('token', access_token)
      
      dispatch({
        type: 'AUTH_SUCCESS',
        payload: {
          user,
          token: access_token,
        },
      })
      
      toast.success(`Welcome to SmartFinance AI, ${user.name}!`)
      return { success: true }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Registration failed'
      dispatch({
        type: 'AUTH_FAILURE',
        payload: errorMessage,
      })
      toast.error(errorMessage)
      return { success: false, error: errorMessage }
    }
  }

  const logout = () => {
    localStorage.removeItem('token')
    dispatch({ type: 'LOGOUT' })
    toast.success('Logged out successfully')
  }

  const updateProfile = async (profileData) => {
    try {
      const response = await authAPI.updateProfile(profileData)
      dispatch({
        type: 'UPDATE_USER',
        payload: response.data,
      })
      toast.success('Profile updated successfully')
      return { success: true }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || 'Profile update failed'
      toast.error(errorMessage)
      return { success: false, error: errorMessage }
    }
  }

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' })
  }

  const value = {
    ...state,
    login,
    register,
    logout,
    updateProfile,
    clearError,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext