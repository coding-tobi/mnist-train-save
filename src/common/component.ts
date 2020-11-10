import ComponentEvent, {
  AddEventListenerFn,
  DispatchEventFn,
} from "./component-event";

export default class Component extends HTMLElement {
  protected readonly root: ShadowRoot;

  constructor() {
    super();
    this.root = this.attachShadow({ mode: "closed" });
  }

  public getChildById<T = HTMLElement>(id: string): T {
    const child = <T | null>this.root.getElementById(id);
    console.assert(child, id + " NOT FOUND!");
    return <T>child;
  }

  public cloneTemplate(id: string): Node {
    const template = this.getChildById<HTMLTemplateElement>(id);
    console.assert(
      template instanceof HTMLTemplateElement,
      id + " IS NOT A TEMPLATE!"
    );
    return template.content.cloneNode(true);
  }

  protected createEvent<T = any>(eventName: string): ComponentEvent<T> {
    return {
      dispatch: this.getDispatchEventFn<T>(eventName),
      addListener: this.getAddEventListenerFn<T>(eventName),
    };
  }

  protected getDispatchEventFn<T>(eventName: string): DispatchEventFn<T> {
    return (detail) => {
      return this.dispatchEvent(
        new CustomEvent<T>(eventName, { detail })
      );
    };
  }

  protected getAddEventListenerFn<T>(eventName: string): AddEventListenerFn<T> {
    return (listener) => {
      this.addEventListener(eventName, (e) => {
        listener(<CustomEvent<T>>e);
      });
    };
  }

  protected connected() {}
  protected disconnected() {}
}

// Hidden implementation of custom element callbacks
(<any>Component.prototype).connectedCallback = function () {
  if (!this.isConnected) return;
  this.connected();
};
(<any>Component.prototype).disconnectedCallback = function () {
  if (!this.isConnected) return;
  this.disconnected();
};
