import Component from "./component";

export interface ComponentDecoratorOptions {
  selector: string;
  style?: string;
  template?: string;
}

const ComponentDecorator = (options: ComponentDecoratorOptions) => (
  component: new () => any
) => {
  console.assert(
    component === Component || component.prototype instanceof Component,
    component.name + " IS NOT A COMPONENT!"
  );

  let html = "";
  if (options.style) html += `<style>${options.style}</style>`;
  if (options.template) html += options.template;
  const template = document.createElement("template");
  template.innerHTML = html;

  const originalConnectedCallback =
    component.prototype.connectedCallback || function () {};
  component.prototype.connectedCallback = function () {
    const templateClone = document.importNode(template.content, true);
    this.root.appendChild(templateClone);
    originalConnectedCallback.call(this);
  };

  window.customElements.define(options.selector, component);
};
export default ComponentDecorator;
