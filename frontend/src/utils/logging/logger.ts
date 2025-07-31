import pino from 'pino';

// Browser-compatible pino configuration with visible console output
const logger = pino({
  level: import.meta.env.PROD ? 'info' : 'debug',
  browser: {
    asObject: true,
    write: {
      info: (obj: any) => {
        console.info('🟢 [INFO]', obj);
      },
      warn: (obj: any) => {
        console.warn('🟡 [WARN]', obj);
      },
      error: (obj: any) => {
        console.error('🔴 [ERROR]', obj);
      },
      debug: (obj: any) => {
        console.debug('🔵 [DEBUG]', obj);
      },
    },
  },
  formatters: {
    level: label => {
      return { level: label };
    },
  },
  timestamp: pino.stdTimeFunctions.isoTime,
});

// Enhanced logging context interfaces
export interface LogContext {
  operation: string;
  endpoint: string;
  startTime: number;
  requestId: string;
}

export interface ApiRequestOptions {
  method?: string;
  body?: any;
  headers?: Record<string, string>;
}

// API Logger class with pino backend
export class ApiLogger {
  private static generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  static startRequest(
    operation: string,
    endpoint: string,
    params?: any
  ): LogContext {
    const requestId = this.generateRequestId();
    const startTime = performance.now();

    logger.info(
      {
        phase: 'start',
        operation,
        endpoint,
        requestId,
        params,
        userAgent: navigator.userAgent,
      },
      `🚀 Starting ${operation}`
    );

    return { operation, endpoint, startTime, requestId };
  }

  static logResponse(
    context: LogContext,
    response: Response,
    dataSize?: string | number
  ) {
    const duration = performance.now() - context.startTime;

    logger.info(
      {
        phase: 'success',
        operation: context.operation,
        requestId: context.requestId,
        endpoint: context.endpoint,
        status: response.status,
        statusText: response.statusText,
        duration: parseFloat(duration.toFixed(2)),
        dataSize,
        headers: Object.fromEntries(response.headers.entries()),
      },
      `✅ ${context.operation} completed in ${duration.toFixed(2)}ms`
    );
  }

  static logError(context: LogContext, error: any, response?: Response) {
    const duration = performance.now() - context.startTime;

    logger.error(
      {
        phase: 'error',
        operation: context.operation,
        requestId: context.requestId,
        endpoint: context.endpoint,
        duration: parseFloat(duration.toFixed(2)),
        error: {
          message: error.message,
          name: error.name,
          stack: error.stack,
        },
        response: response
          ? {
              status: response.status,
              statusText: response.statusText,
              headers: Object.fromEntries(response.headers.entries()),
            }
          : null,
      },
      `❌ ${context.operation} failed after ${duration.toFixed(2)}ms`
    );
  }

  static logWarning(context: LogContext, message: string, details?: any) {
    logger.warn(
      {
        phase: 'warning',
        operation: context.operation,
        requestId: context.requestId,
        endpoint: context.endpoint,
        ...details,
      },
      `⚠️ ${message}`
    );
  }

  static logRequest(context: LogContext, options: RequestInit) {
    logger.debug(
      {
        phase: 'request',
        operation: context.operation,
        requestId: context.requestId,
        endpoint: context.endpoint,
        method: options.method || 'GET',
        headers: options.headers,
      },
      `📤 Making ${options.method || 'GET'} request`
    );
  }

  static logResponseReceived(context: LogContext, response: Response) {
    logger.debug(
      {
        phase: 'response_received',
        operation: context.operation,
        requestId: context.requestId,
        status: response.status,
        statusText: response.statusText,
      },
      `📥 Received response: ${response.status} ${response.statusText}`
    );
  }

  static logDataParsing(context: LogContext, data: any) {
    const dataInfo = Array.isArray(data)
      ? { type: 'array', length: data.length }
      : typeof data === 'object' && data !== null
        ? { type: 'object', properties: Object.keys(data).length }
        : { type: typeof data };

    logger.debug(
      {
        phase: 'data_parsed',
        operation: context.operation,
        requestId: context.requestId,
        dataInfo,
      },
      `📋 Parsed ${dataInfo.type} data`
    );
  }
}

// Service-level logging helpers
export class ServiceLogger {
  static logServiceCall(serviceName: string, params?: any) {
    logger.info(
      {
        service: serviceName,
        params,
      },
      `🎯 Service ${serviceName} called`
    );
  }

  static logServiceWarning(
    serviceName: string,
    message: string,
    details?: any
  ) {
    logger.warn(
      {
        service: serviceName,
        ...details,
      },
      `⚠️ ${serviceName}: ${message}`
    );
  }

  static logServiceError(serviceName: string, error: any, details?: any) {
    logger.error(
      {
        service: serviceName,
        error: {
          message: error.message,
          name: error.name,
          stack: error.stack,
        },
        ...details,
      },
      `💥 ${serviceName} error: ${error.message}`
    );
  }
}

// Export the base logger for direct use if needed
export { logger };
export default logger;
