Vue.component("html-video", {
  props: ["timeToSet", "url", "mimeType", "isPlaying", "maskPath"],
  template: `
<div style="position: relative;">
    <video ref="video"
        width="100%"
        height="auto"
        controls
        @timeupdate="$emit('timeupdate', $refs['video'].currentTime)"
        @play="$emit('update:is-playing', true)"
        @pause="$emit('update:is-playing', false)"
    >
        <source ref="video-data" :src="url" :type="mimeType">
    </video>
    <div v-if="maskPath"
        ref="mask"
        style="
            opacity: 0.4;
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            background-size: contain;
        " 
        :style="overlayStyle">
    </div>
</div>
`,
  computed: {
    overlayStyle() {
      return {
        backgroundImage: `url("${this.maskPath}")`,
      };
    },
  },
  watch: {
    timeToSet(time) {
      if (Number.isFinite(time)) {
        this.$refs["video"].currentTime = time;
        this.$emit("update:time-to-set", null);
      }
    },
    url: {
      handler() {
        this.updateVideoSrc();
      },
      immediate: true,
    },
    mimeType: {
      handler() {
        this.updateVideoSrc();
      },
      immediate: true,
    },
    isPlaying: {
      handler(value) {
        this.playPause(value);
      },
    },
  },
  mounted() {
    if (this.isPlaying) {
      this.$emit("update:is-playing", false);
    }
  },
  methods: {
    updateVideoSrc() {
      const video = this.$refs["video"];
      const source = this.$refs["video-data"];

      if (!this.url || !this.mimeType || !video) {
        return;
      }
      video.pause();
      source.setAttribute("src", this.url);
      source.setAttribute("type", this.mimeType);
      video.load();
    },
    playPause() {
      const video = this.$refs["video"];
      if (!video) {
        return;
      }
      this.isPlaying ? video.play() : video.pause();
    },
  },
});
