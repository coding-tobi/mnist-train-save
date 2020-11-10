export interface EventListener<T> {
  (e: CustomEvent<T>): void;
}

export interface DispatchEventFn<T> {
  (detail: T): boolean;
}

export interface AddEventListenerFn<T> {
  (listener: EventListener<T>): void;
}

export default interface ComponentEvent<T> {
  dispatch: DispatchEventFn<T>;
  addListener: AddEventListenerFn<T>;
}
