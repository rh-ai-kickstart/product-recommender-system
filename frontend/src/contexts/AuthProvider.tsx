import React, { createContext, useContext, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { authService, storeUserData } from '../services/auth';
import type { User, LoginRequest, SignUpRequest, AuthResponse } from '../types';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
  signup: (userData: SignUpRequest) => Promise<void>;
  logout: () => void;
  refetchUser: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const queryClient = useQueryClient();

  // Query to get current user
  const {
    data: user,
    isLoading,
    error,
    refetch: refetchUser,
  } = useQuery({
    queryKey: ['currentUser'],
    queryFn: authService.getCurrentUser,
    enabled: authService.isAuthenticated(),
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Login mutation
  const loginMutation = useMutation({
    mutationFn: authService.login,
    onSuccess: (authResponse: AuthResponse) => {
      storeUserData(authResponse.user);
      queryClient.setQueryData(['currentUser'], authResponse.user);
      queryClient.invalidateQueries({ queryKey: ['currentUser'] });
    },
    onError: (error: Error) => {
      console.error('Login failed:', error.message);
      throw error;
    },
  });

  // Signup mutation
  const signupMutation = useMutation({
    mutationFn: authService.signup,
    onSuccess: (authResponse: AuthResponse) => {
      storeUserData(authResponse.user);
      queryClient.setQueryData(['currentUser'], authResponse.user);
      queryClient.invalidateQueries({ queryKey: ['currentUser'] });
    },
    onError: (error: Error) => {
      console.error('Signup failed:', error.message);
      throw error;
    },
  });

  // Auth functions
  const login = async (credentials: LoginRequest): Promise<void> => {
    await loginMutation.mutateAsync(credentials);
  };

  const signup = async (userData: SignUpRequest): Promise<void> => {
    await signupMutation.mutateAsync(userData);
  };

  const logout = (): void => {
    authService.logout();
    queryClient.setQueryData(['currentUser'], null);
    queryClient.removeQueries({ queryKey: ['currentUser'] });
    queryClient.clear(); // Clear all cached data on logout
  };

  // Check authentication status on mount and when token changes
  useEffect(() => {
    if (!authService.isAuthenticated() && user) {
      logout();
    }
  }, [user]);

  // Handle query errors (e.g., token expired)
  useEffect(() => {
    if (error && authService.isAuthenticated()) {
      logout();
    }
  }, [error]);

  const contextValue: AuthContextType = {
    user: user || null,
    isLoading: isLoading || loginMutation.isPending || signupMutation.isPending,
    isAuthenticated: !!user && authService.isAuthenticated(),
    login,
    signup,
    logout,
    refetchUser: () => refetchUser(),
  };

  return (
    <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
  );
};
