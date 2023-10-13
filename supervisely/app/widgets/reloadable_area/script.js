Vue.component("sly-template-renderer", {
  props: ["id", "state", "data", "context", "post", "session"],
  data: function () {
    return {
      templateRenderer: false,
      componentKey: 0,
    };
  },
  methods: {
    renderTemplate(payload) {
      if (!payload?.template) return;

      const { template } = payload;

      if (!template) return;

      this._staticTrees = [];

      const compiled = Vue.compile(`<div>${template}</div>`);
      this.$options.staticRenderFns = compiled.staticRenderFns;

      this.templateRenderer = compiled.render;
      this.componentKey += 1;
    },
  },

  render(createElement) {
    if (this.templateRenderer) {
      return this.templateRenderer.call(this, createElement);
    } else if (this.$slots?.default?.[0]) {
      return this.$slots.default[0];
    } else {
      return "";
    }
  },

  created() {
    this.$eventBus.$on(`rerender-template-${this.id}`, this.renderTemplate);
  },
  beforeDestroy() {
    this.$eventBus.$off(`rerender-template-${this.id}`, this.renderTemplate);
  },
});
