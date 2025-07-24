import {
  ApiLogger,
  ServiceLogger,
  type ApiRequestOptions,
} from '../utils/logging/logger';

// Enhanced API fetch utility using the new logger
export async function apiRequest<T>(
  endpoint: string,
  operation: string,
  options: ApiRequestOptions = {}
): Promise<T> {
  const context = ApiLogger.startRequest(operation, endpoint, options);

  try {
    const { method = 'GET', body, headers } = options;
    const requestOptions: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
    };

    if (body && method !== 'GET') {
      requestOptions.body = JSON.stringify(body);
    }

    ApiLogger.logRequest(context, requestOptions);
    const response = await fetch(endpoint, requestOptions);
    ApiLogger.logResponseReceived(context, response);

    if (!response.ok) {
      let errorDetails = '';
      try {
        errorDetails = await response.text();
      } catch (e) {
        errorDetails = 'Could not read error response';
      }

      ApiLogger.logError(
        context,
        new Error(
          `HTTP ${response.status}: ${response.statusText}. ${errorDetails}`
        ),
        response
      );
      throw new Error(
        `API request failed: ${response.status} ${response.statusText}. ${errorDetails}`
      );
    }

    const data = (await response.json()) as T;
    ApiLogger.logDataParsing(context, data);

    // Calculate response size for logging
    let dataSize: string | number = 'unknown';
    if (Array.isArray(data)) {
      dataSize = `${data.length} items`;
    } else if (typeof data === 'object' && data !== null) {
      dataSize = `${Object.keys(data as object).length} properties`;
    }

    ApiLogger.logResponse(context, response, dataSize);
    return data;
  } catch (error) {
    ApiLogger.logError(context, error);
    throw error;
  }
}

export { ServiceLogger };
