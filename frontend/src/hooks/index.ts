// Re-export all hooks for easier importing
export {
  usePersonalizedRecommendations,
  useExistingUserRecommendations,
  useNewUserRecommendations,
} from './useRecommendations';
export * from './useProducts';
export * from './useCart';
export * from './useOrders';
export * from './useFeedback';
export * from './usePreferences';
export * from './useInteractions';
export * from './useComposite';
export * from './useOptimistic';
export { useAuth, useLogin, useSignup, useLogout } from './useAuth';
