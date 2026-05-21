type EventHandler<T = any> = (payload?: T) => void;

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

  emit<T = any>(event: string, payload?: T) {
    this.events.get(event)?.forEach((handler) => {
      handler(payload);
    });
  }

  clear(event: string) {
    this.events.delete(event);
  }
}

export const eventBus = new EventBus();