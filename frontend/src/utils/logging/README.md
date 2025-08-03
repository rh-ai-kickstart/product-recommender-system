# Logging System

This project uses **Pino** for structured logging with specialized utilities for API and service logging.

## 📁 Files

- `logger.ts` - Main logging utility with Pino configuration
- `../services/products.ts` - Example implementation using the logging system

## 🚀 Quick Start

### Import the loggers:

```typescript
import { ApiLogger, ServiceLogger, logger } from '../utils/logging/logger';
```

### Service-level logging:

```typescript
export const myService = async (param: string) => {
  ServiceLogger.logServiceCall('myService', { param });

  try {
    // Your service logic here
    const result = await doSomething(param);
    return result;
  } catch (error) {
    ServiceLogger.logServiceError('myService', error, { param });
    throw error;
  }
};
```

### API request logging:

```typescript
const makeApiCall = async () => {
  const context = ApiLogger.startRequest('getUserData', '/users/123');

  try {
    const response = await fetch('/users/123');
    ApiLogger.logResponseReceived(context, response);

    if (!response.ok) {
      ApiLogger.logError(context, new Error('Failed to fetch'), response);
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    ApiLogger.logDataParsing(context, data);
    ApiLogger.logResponse(context, response, `${data.length} users`);

    return data;
  } catch (error) {
    ApiLogger.logError(context, error);
    throw error;
  }
};
```

### Direct logging:

```typescript
import logger from '../utils/logger';

// Info logging
logger.info({ userId: 123, action: 'login' }, 'User logged in');

// Warning logging
logger.warn({ route: '/slow' }, 'Slow API response detected');

// Error logging
logger.error(
  { error: error.message, stack: error.stack },
  'Database connection failed'
);
```

## 📊 Log Structure

All logs include structured data that's easy to search and filter:

### Service Logs:

```json
{
  "level": "info",
  "time": "2024-01-15T10:30:45.123Z",
  "service": "fetchRecommendations",
  "params": { "userId": 123 },
  "msg": "🎯 Service fetchRecommendations called"
}
```

### API Logs:

```json
{
  "level": "info",
  "time": "2024-01-15T10:30:45.456Z",
  "phase": "success",
  "operation": "searchProducts",
  "requestId": "req_1705123456789_abc123def",
  "endpoint": "/products/search?q=laptop",
  "status": 200,
  "duration": 45.23,
  "dataSize": "12 items",
  "msg": "✅ searchProducts completed in 45.23ms"
}
```

## 🔧 Configuration

The logger is configured in `logger.ts`:

```typescript
const logger = pino({
  level: import.meta.env.PROD ? 'info' : 'debug', // Debug logs in dev mode
  browser: { asObject: true }, // Browser compatibility
  timestamp: pino.stdTimeFunctions.isoTime, // ISO timestamps
});
```

## 🎯 Log Levels

- **`debug`** - Detailed technical information (dev only)
- **`info`** - General information about application flow
- **`warn`** - Warning conditions that should be noted
- **`error`** - Error conditions that need attention

## 🔍 Viewing Logs

### Browser Console:

1. Open Developer Tools (F12)
2. Go to Console tab
3. Look for structured log objects with emojis

### Filtering in Console:

```javascript
// Filter by log level
console.filter(log => log.level === 'error');

// Filter by operation
console.filter(log => log.operation === 'searchProducts');

// Filter by service
console.filter(log => log.service === 'fetchRecommendations');
```

## 🏗️ Best Practices

### ✅ Do:

- Log at service entry/exit points
- Include relevant context (user ID, request ID, etc.)
- Use structured data instead of string interpolation
- Log errors with full context
- Use appropriate log levels

### ❌ Don't:

- Log sensitive data (passwords, tokens, PII)
- Use string concatenation in log messages
- Log in tight loops without throttling
- Mix console.log with the structured logger

## 🔗 Integration with External Services

For production, you can extend the logger to send logs to external services:

```typescript
// In logger.ts, add:
if (import.meta.env.PROD) {
  // Send to external logging service
  logger.addDestination({
    dest: {
      write: log => {
        // Send to Datadog, Sentry, CloudWatch, etc.
        externalLoggingService.send(JSON.parse(log));
      },
    },
  });
}
```

## 📈 Monitoring

Key metrics you can extract from logs:

- API response times (`duration` field)
- Error rates by operation (`phase: 'error'`)
- Service call patterns (`service` field)
- Request volume by endpoint (`endpoint` field)
