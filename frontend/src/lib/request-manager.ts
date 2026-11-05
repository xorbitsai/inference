class RequestManager {
  private errorLocks = new Map<string, symbol>();

  /** Acquire before emitting so concurrent failures produce a single UI action. */
  acquireErrorLock(event: string): (() => void) | null {
    if (this.errorLocks.has(event)) return null;
    const owner = Symbol(event);
    this.errorLocks.set(event, owner);
    return () => {
      // A stale release must never unlock a newer event.
      if (this.errorLocks.get(event) === owner) this.errorLocks.delete(event);
    };
  }

  /**
   * Is 401 being processed?
   */
  private handling401 = false;

  /**
   * Is 403 being processed?
   */
  private handling403 = false;

  canHandle401() {
    if (this.handling401) {
      return false;
    }

    this.handling401 = true;

    return true;
  }

  reset401() {
    this.handling401 = false;
  }

  canHandle403() {
    if (this.handling403) {
      return false;
    }

    this.handling403 = true;

    return true;
  }

  reset403() {
    this.handling403 = false;
  }
}

export const requestManager = new RequestManager();
