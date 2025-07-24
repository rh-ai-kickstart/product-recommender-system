import { apiRequest, ServiceLogger } from './api';

export interface PreferencesRequest {
  preferences: string;
}

export const setPreferences = async (
  preferences: PreferencesRequest
): Promise<any> => {
  ServiceLogger.logServiceCall('setPreferences', { preferences });
  return apiRequest<any>('/api/users/preferences', 'setPreferences', {
    method: 'POST',
    body: preferences,
  });
};

export const getPreferences = async (): Promise<string> => {
  ServiceLogger.logServiceCall('getPreferences');
  return apiRequest<string>('/api/users/preferences', 'getPreferences');
};
