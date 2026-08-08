type EventHandler<T = any> = (...args: T[]) => void;

class EventBus {
  private events: Map<string, Set<EventHandler>> =
    new Map();

  on(event: string, handler: EventHandler) {
    if (!this.events.has(event)) {
      this.events.set(event, new Set());
    }

    this.events.get(event)?.add(handler);
  }

  off(event: string, handler: EventHandler) {
    this.events.get(event)?.delete(handler);
  }

  emit(event: string, ...args: any[]) {
    this.events.get(event)?.forEach((handler) => {
      handler(...args);
    });
  }

  clear(event: string) {
    this.events.delete(event);
  }
}

export const eventBus = new EventBus();