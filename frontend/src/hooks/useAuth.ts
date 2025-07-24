import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import { useAuth as useAuthContext } from '../contexts/AuthProvider';
// import { apiRequest } from '../services/api';
import type { LoginRequest, SignUpRequest} from '../types';

/**
 * Hook for login with automatic redirect
 */
export const useLogin = () => {
  const { login } = useAuthContext();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: async (credentials: LoginRequest) => {
      await login(credentials);
    },
    onSuccess: () => {
      // Get redirect path from URL params or default to home
      const searchParams = new URLSearchParams(window.location.search);
      const redirectPath = searchParams.get('redirect') || '/';
      navigate({ to: redirectPath });
    },
  });
};

/**
 * Hook for signup with automatic redirect
 */
export const useSignup = () => {
  const { signup } = useAuthContext();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: async (userData: SignUpRequest) => {
      await signup(userData);
    },
    onSuccess: () => {
      navigate({ to: '/' });
    },
  });
};

/**
 * Hook for logout with redirect to login
 */
export const useLogout = () => {
  const { logout } = useAuthContext();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      logout();
    },
    onSuccess: () => {
      // Clear all cached data and redirect to login
      queryClient.clear();
      navigate({ to: '/login' });
    },
  });
};

/**
 * Re-export the main auth context hook
 */
export const useAuth = useAuthContext;
