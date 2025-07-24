import { logger } from './logger';
import type { User, LoginRequest, SignUpRequest } from '../../types';

// Auth-specific logging context
export interface AuthContext {
  operation: string;
  startTime: number;
  sessionId: string;
  userAgent: string;
}

// Auth event types
export type AuthEvent =
  | 'login_attempt'
  | 'login_success'
  | 'login_failure'
  | 'signup_attempt'
  | 'signup_success'
  | 'signup_failure'
  | 'logout'
  | 'token_validation'
  | 'token_expired'
  | 'token_invalid'
  | 'session_check'
  | 'unauthorized_access';

export class AuthLogger {
  private static generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private static getCurrentUserId(): string | null {
    try {
      const userData = localStorage.getItem('user_data');
      if (userData) {
        const user = JSON.parse(userData);
        return user.user_id || user.email || 'unknown';
      }
    } catch {
      // Ignore parsing errors
    }
    return null;
  }

  private static getTokenInfo(): {
    hasToken: boolean;
    isExpired: boolean;
    expiresAt?: number;
  } {
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) return { hasToken: false, isExpired: false };

      const payload = JSON.parse(atob(token.split('.')[1]));
      const expiresAt = payload.exp * 1000;
      const isExpired = expiresAt < Date.now();

      return { hasToken: true, isExpired, expiresAt };
    } catch {
      return { hasToken: true, isExpired: true };
    }
  }

  // Start an auth operation
  static startAuthOperation(operation: string, details?: any): AuthContext {
    const sessionId = this.generateSessionId();
    const startTime = performance.now();
    const userAgent = navigator.userAgent;
    const tokenInfo = this.getTokenInfo();

    logger.info(
      {
        phase: 'start',
        operation,
        sessionId,
        userAgent: userAgent.substring(0, 100), // Truncate for readability
        currentUser: this.getCurrentUserId(),
        tokenInfo,
        details,
      },
      `🔐 Starting ${operation}`
    );

    return { operation, startTime, sessionId, userAgent };
  }

  // Log successful auth events
  static logAuthSuccess(
    context: AuthContext,
    event: AuthEvent,
    user?: User,
    details?: any
  ) {
    const duration = performance.now() - context.startTime;

    logger.info(
      {
        phase: 'success',
        event,
        operation: context.operation,
        sessionId: context.sessionId,
        userId: user?.user_id || this.getCurrentUserId(),
        userEmail: user?.email,
        duration: parseFloat(duration.toFixed(2)),
        tokenInfo: this.getTokenInfo(),
        details,
      },
      `✅ ${event} successful (${duration.toFixed(2)}ms)`
    );
  }

  // Log failed auth events
  static logAuthFailure(
    context: AuthContext,
    event: AuthEvent,
    error: any,
    details?: any
  ) {
    const duration = performance.now() - context.startTime;

    logger.error(
      {
        phase: 'failure',
        event,
        operation: context.operation,
        sessionId: context.sessionId,
        duration: parseFloat(duration.toFixed(2)),
        error: {
          message: error.message,
          name: error.name,
        },
        tokenInfo: this.getTokenInfo(),
        details,
      },
      `❌ ${event} failed: ${error.message} (${duration.toFixed(2)}ms)`
    );
  }

  // Log auth warnings
  static logAuthWarning(event: AuthEvent, message: string, details?: any) {
    logger.warn(
      {
        event,
        currentUser: this.getCurrentUserId(),
        tokenInfo: this.getTokenInfo(),
        userAgent: navigator.userAgent.substring(0, 100),
        details,
      },
      `⚠️ Auth Warning: ${message}`
    );
  }

  // Specific auth event loggers
  static logLoginAttempt(credentials: LoginRequest) {
    logger.info(
      {
        event: 'login_attempt',
        email: credentials.email,
        userAgent: navigator.userAgent.substring(0, 100),
        timestamp: new Date().toISOString(),
      },
      `🔑 Login attempt for ${credentials.email}`
    );
  }

  static logSignupAttempt(userData: SignUpRequest) {
    logger.info(
      {
        event: 'signup_attempt',
        email: userData.email,
        age: userData.age,
        gender: userData.gender,
        userAgent: navigator.userAgent.substring(0, 100),
        timestamp: new Date().toISOString(),
      },
      `👤 Signup attempt for ${userData.email}`
    );
  }

  static logLogout(userId?: string) {
    logger.info(
      {
        event: 'logout',
        userId: userId || this.getCurrentUserId(),
        timestamp: new Date().toISOString(),
      },
      `🚪 User logged out`
    );
  }

  static logTokenExpired() {
    logger.warn(
      {
        event: 'token_expired',
        currentUser: this.getCurrentUserId(),
        timestamp: new Date().toISOString(),
      },
      `⏰ Token expired for user`
    );
  }

  static logTokenInvalid(error?: string) {
    logger.error(
      {
        event: 'token_invalid',
        currentUser: this.getCurrentUserId(),
        error,
        timestamp: new Date().toISOString(),
      },
      `🚫 Invalid token detected`
    );
  }

  static logSessionCheck(isValid: boolean) {
    logger.debug(
      {
        event: 'session_check',
        isValid,
        currentUser: this.getCurrentUserId(),
        tokenInfo: this.getTokenInfo(),
        timestamp: new Date().toISOString(),
      },
      `🔍 Session validation: ${isValid ? 'valid' : 'invalid'}`
    );
  }

  static logUnauthorizedAccess(attemptedRoute?: string) {
    logger.warn(
      {
        event: 'unauthorized_access',
        attemptedRoute,
        currentUser: this.getCurrentUserId(),
        tokenInfo: this.getTokenInfo(),
        userAgent: navigator.userAgent.substring(0, 100),
        timestamp: new Date().toISOString(),
      },
      `🚨 Unauthorized access attempt to ${attemptedRoute || 'protected route'}`
    );
  }

  // Security event logging
  static logSecurityEvent(eventType: string, details: any) {
    logger.warn(
      {
        event: 'security_event',
        eventType,
        currentUser: this.getCurrentUserId(),
        userAgent: navigator.userAgent.substring(0, 100),
        timestamp: new Date().toISOString(),
        ...details,
      },
      `🛡️ Security Event: ${eventType}`
    );
  }

  // Auth metrics logging (useful for monitoring)
  static logAuthMetrics() {
    const tokenInfo = this.getTokenInfo();
    const currentUser = this.getCurrentUserId();

    logger.debug(
      {
        event: 'auth_metrics',
        hasActiveSession: !!currentUser,
        tokenStatus: tokenInfo.hasToken
          ? tokenInfo.isExpired
            ? 'expired'
            : 'valid'
          : 'none',
        sessionAge: tokenInfo.expiresAt
          ? tokenInfo.expiresAt - Date.now()
          : null,
        timestamp: new Date().toISOString(),
      },
      `📊 Auth metrics snapshot`
    );
  }
}

export default AuthLogger;
