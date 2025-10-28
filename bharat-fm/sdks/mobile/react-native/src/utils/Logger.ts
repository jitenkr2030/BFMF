/**
 * Logger utility for the Bharat AI SDK
 */

export class Logger {
  private enabled: boolean;
  private prefix: string;

  constructor(enabled: boolean = false, prefix: string = 'BharatAI') {
    this.enabled = enabled;
    this.prefix = prefix;
  }

  /**
   * Update logging configuration
   */
  public updateConfig(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Log debug message
   */
  public debug(message: string, ...args: any[]): void {
    if (this.enabled) {
      console.debug(`[${this.prefix}] DEBUG: ${message}`, ...args);
    }
  }

  /**
   * Log info message
   */
  public info(message: string, ...args: any[]): void {
    if (this.enabled) {
      console.info(`[${this.prefix}] INFO: ${message}`, ...args);
    }
  }

  /**
   * Log warning message
   */
  public warn(message: string, ...args: any[]): void {
    if (this.enabled) {
      console.warn(`[${this.prefix}] WARN: ${message}`, ...args);
    }
  }

  /**
   * Log error message
   */
  public error(message: string, ...args: any[]): void {
    if (this.enabled) {
      console.error(`[${this.prefix}] ERROR: ${message}`, ...args);
    }
  }

  /**
   * Log with custom level
   */
  public log(level: string, message: string, ...args: any[]): void {
    if (!this.enabled) {
      return;
    }

    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${this.prefix}] ${level.toUpperCase()}: ${message}`;

    switch (level.toLowerCase()) {
      case 'debug':
        console.debug(logMessage, ...args);
        break;
      case 'info':
        console.info(logMessage, ...args);
        break;
      case 'warn':
        console.warn(logMessage, ...args);
        break;
      case 'error':
        console.error(logMessage, ...args);
        break;
      default:
        console.log(logMessage, ...args);
    }
  }

  /**
   * Create child logger with custom prefix
   */
  public child(childPrefix: string): Logger {
    return new Logger(this.enabled, `${this.prefix}:${childPrefix}`);
  }

  /**
   * Enable or disable logging
   */
  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Check if logging is enabled
   */
  public isEnabled(): boolean {
    return this.enabled;
  }
}