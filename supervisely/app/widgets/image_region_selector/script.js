const POINT_SIZE = 6;
const VIEW_BOX_OFFSET = 60;
const VIEW_BOX_OFFSET_HALF = VIEW_BOX_OFFSET / 2;

function canvasTintImage(image, color) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  context.canvas.width = image.width;
  context.canvas.height = image.height;

  context.save();
  context.fillStyle = color;
  context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  context.globalCompositeOperation = "destination-atop";
  context.globalAlpha = 1;
  context.drawImage(image, 0, 0);
  context.restore();

  return context.canvas;
}

function getViewBox(viewBox) {
  viewBox.height += VIEW_BOX_OFFSET;
  viewBox.h += VIEW_BOX_OFFSET;
  viewBox.width += VIEW_BOX_OFFSET;
  viewBox.w += VIEW_BOX_OFFSET;
  viewBox.x -= VIEW_BOX_OFFSET_HALF;
  viewBox.x2 += VIEW_BOX_OFFSET_HALF;
  viewBox.y -= VIEW_BOX_OFFSET_HALF;
  viewBox.y2 += VIEW_BOX_OFFSET_HALF;

  return viewBox;
}

function loadImage(urlPath, force = false) {
  let canceled = false;
  let imgPath = urlPath;

  const img = new Image();

  return Object.assign(new Promise(async (res, rej) => {
    try {
      img.onload = () => {
        img.onerror = null;
        img.onload = null;

        URL.revokeObjectURL(imgPath);

        return res(img);
      };

      img.onerror = (err) => {
        img.onload = null;
        img.onerror = null;

        URL.revokeObjectURL(imgPath);

        let curErr;

        if (canceled) {
          curErr = new Error('Image downloading has been canceled');

          curErr.canceled = true;
        } else {
          curErr = new Error('Couldn\'t load the image');
        }

        curErr.event = err;

        rej(curErr);
      };

      img.src = imgPath;
    } catch (err) {
      err.url = imgPath;

      rej(err);
    }
  }), {
    cancel() {
      if (!canceled) {
        img.src = '';
        canceled = true;
      }

      return this;
    },
  });
}

async function base64BitmapToRaw(srcBitmap) {
  const decodedStr = self.atob(srcBitmap); // eslint-disable-line no-restricted-globals
  let result;

  if (srcBitmap.startsWith('eJ')) {
    result = pako.inflate(decodedStr);
  } else {
    result = Uint8Array.from(decodedStr, c => c.charCodeAt(0));
  }

  return result;
}

function getBBoxSize(bbox) {
  return {
    width: bbox[1][0] - bbox[0][0],
    height: bbox[1][1] - bbox[0][1],
  };
}

Vue.component('smarttool-editor', {
  template: `
    <div v-loading="loading" style="position: relative;">
      <div v-if="disabled" style="position: absolute; inset: 0; opacity: 0.5; background-color: #808080;"></div>
      <svg ref="container" xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:xlink="http://www.w3.org/1999/xlink" width="100%" height="100%"></svg>
    </div>
  `,
  props: {
    maskOpacity: 0.5,
    bbox: {
      type: Array,
      required: true,
    },
    imageUrl: {
      type: String,
      required: true,
    },
    mask: {
      type: Object,
    },
    positivePoints: {
      type: Array,
      default: [],
    },
    negativePoints: {
      type: Array,
      default: [],
    },
    disabled: {
      type: Boolean,
      default: false,
    },
    pointsDisabled: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      pt: null,
      container: null,
      loading: true,
      contours: [],
    };
  },
  watch: {
    imageUrl() {
      this.group.clear()
      const viewBox = getViewBox(this.bboxEl.bbox());
      this.sceneEl.viewbox(viewBox)
      this.backgroundEl = this.sceneEl.image(this.imageUrl).loaded(() => {
        this.pointSize = POINT_SIZE * (viewBox.w / this.container.width.baseVal.value);
        this.initPoints();
      });
      this.group.add(
        this.backgroundEl,
        this.bboxEl
      );
    },
    'mask.contour': {
      handler() {
        this.contours.forEach((c) => {
          c.remove();
        });

        if (this.mask?.contour) {
          this.mask?.contour.forEach((c) => {
            const contourEl1 = this.sceneEl.polygon()
              .plot(c || [])
              .fill('none')
              .stroke({
                color: '#ff6600',
                width: 3,
              });
            const contourEl2 = this.sceneEl.polygon()
              .plot(c || [])
              .fill('none')
              .stroke({
                color: 'white',
                width: 3,
                dasharray: '10 10',
              });

            this.contours.push(contourEl1, contourEl2);
          });
        }
      },
      deep: true,
    },
    mask: {
      async handler() {
        if (!this.mask) {
          if (this.maskEl) {
            this.maskEl.load('data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7');
          }
          return;
        };
        const buf = await base64BitmapToRaw(this.mask.data);
        const annImageUrl = URL.createObjectURL(new Blob([buf]));
        let image = await loadImage(annImageUrl);
        const canvasImg = canvasTintImage(image, this.mask.color);

        this.maskEl.load(canvasImg.toDataURL())
          .attr({
            width: image.width,
            height: image.height,
          })
          .move(...this.mask.origin);
      },
      deep: true,
      immediate: true
    },

    bbox() {
      const bboxSize = getBBoxSize(this.bbox);
      this.bboxEl.size(bboxSize.width, bboxSize.height)
        .move(this.bbox[0][0], this.bbox[0][1])

      this.sceneEl.viewbox(getViewBox(this.bboxEl.bbox()))
    },

    positivePoints: {
      handler() {
        this.pointsChanged(this.positivePoints, true);
      },
      deep: true,
    },

    negativePoints: {
      handler() {
        this.pointsChanged(this.negativePoints, false);
      },
      deep: true,
    },

    maskOpacity: {
      handler() {
        if (!this.maskEl) return;
        this.maskEl.node.style.opacity = this.maskOpacity;
      },
      immediate: true,
    },
  },
  methods: {
    pointsChanged(points, isPositive) {
      const pointsSet = new Set();
      points.forEach((point) => {
        pointsSet.add(point.id);

        const pt = this.pointsMap.get(point.id);
        const position = [Math.floor(point.position[0][0] - (this.pointSize)), Math.floor(point.position[0][1] - (this.pointSize))]

        if (pt) {
          pt.point.move(position[0], position[1])
          return;
        }

        this.addPoint({
          id: point.id,
          x: position[0],
          y: position[1],
          isPositive,
        });
      });

      this.pointsMap.forEach((p) => {
        if (p.point.slyData.isPositive !== isPositive) return;

        if (!pointsSet.has(p.point.slyData.id)) {
          p.point.off('contextmenu', this.removePointHandler);
          p.point.remove();
          this.pointsMap.delete(p.point.slyData.id);
        }
      });
    },

    removePointHandler(pEvt) {
      if (!pEvt.ctrlKey) return;

      const curPoint = pEvt.target && pEvt.target.instance;

      if (!curPoint) return;

      let eventKey = 'positive';
      let curPoints = this.positivePoints;

      if (!curPoint.slyData.isPositive) {
        eventKey = 'negative';
        curPoints = this.negativePoints;
      }

      let eventName = `update:${eventKey}-points`;

      this.$emit(eventName, [...curPoints.filter(p => p.id !== curPoint.slyData.id)])

      this.pointsMap.delete(curPoint.slyData.id);
      curPoint.off('contextmenu', this.removePointHandler);
      curPoint.remove();
    },

    addPoint(params) {
      const {
        id,
        x,
        y,
        isPositive,
      } = params;

      const typeKey = isPositive ? 'positive' : 'negative';

      const point = this.sceneEl
        .circle(this.pointSize * 2)
        .move(x, y)
        .draggable()
        .on('contextmenu', this.removePointHandler)
        .on('dragend', () => {
          const pointsArr = isPositive ? this.positivePoints : this.negativePoints;

          const curPoint = pointsArr.find(p => p.id === id);

          if (!curPoint) return;

          curPoint.position[0][0] = Math.floor(point.x() + this.pointSize);
          curPoint.position[0][1] = Math.floor(point.y() + this.pointSize);
          this.$emit(`update:${typeKey}-points`, [...pointsArr]);
        });

      point.slyData = {
        id,
        isPositive,
      };

      point.attr({
        fill: isPositive ? 'green' : 'red',
      });

      point.addClass('sly-smart-tool__point');
      point.addClass(typeKey);

      this.pointsMap.set(id, { point });
    },

    pointHandler(evt) {
      this.pt.x = evt.x;
      this.pt.y = evt.y;

      const transformed = this.pt.matrixTransform(this.container.getScreenCTM().inverse());

      const pointData = {
        position: [[Math.floor(transformed.x), Math.floor(transformed.y)]],
        id: uuidv4(),
      };

      const isPositive = !(evt.shiftKey || evt.type === 'contextmenu');

      this.addPoint({
        id: pointData.id,
        x: pointData.position[0][0],
        y: pointData.position[0][1],
        isPositive,
      });

      let eventKey = 'positive';
      let curPoints = this.positivePoints;

      if (!isPositive) {
        eventKey = 'negative';
        curPoints = this.negativePoints;
      }

      let eventName = `update:${eventKey}-points`;

      this.$emit(eventName, [...curPoints, pointData])
    },

    initPoints() {
      this.pointsChanged(this.positivePoints, true);
      this.pointsChanged(this.negativePoints, false);

      this.bboxEl.node.nextElementSibling.childNodes.forEach((n) => {
        if (!n.r) return;
        n.setAttribute('r', this.pointSize);
      });

      this.loading = false;
    },

    init() {
      this.container.addEventListener('contextmenu', (e) => {
        e.preventDefault();
      });

      this.sceneEl = SVG(this.container)
        .panZoom({
          zoomMin: 0.1,
          zoomMax: 20,
          panButton: 2
        });

      this.group = this.sceneEl.group();

      const bboxSize = getBBoxSize(this.bbox);

      this.maskEl = this.sceneEl.image();
      this.maskEl.addClass('sly-smart-tool__annotation');
      this.maskEl.node.style.opacity = this.maskOpacity;

      if (this.mask?.contour) {
        this.mask?.contour.forEach((c) => {
          const contourEl1 = this.sceneEl.polygon()
            .plot(c || [])
            .fill('none')
            .stroke({
              width: 3,
              color: '#ff6600',
              // dasharray: '10 10',
            });
          const contourEl2 = this.sceneEl.polygon()
            .plot(c || [])
            .fill('none')
            .stroke({
              color: 'white',
              width: 3,
              dasharray: '10 10',
            });

          this.contours.push(contourEl1, contourEl2);
        });
      }

      this.bboxEl = this.sceneEl
        .rect(bboxSize.width, bboxSize.height)
        .move(this.bbox[0][0], this.bbox[0][1])
        .selectize()
        .resize()
        .attr({
          "fill-opacity": 0,
        })
        .on('resizedone', () => {
          let x = this.bboxEl.x();
          let y = this.bboxEl.y();
          let w = this.bboxEl.width();
          let h = this.bboxEl.height();
          let image_width = this.backgroundEl.node.width.baseVal.value;
          let image_height = this.backgroundEl.node.height.baseVal.value;

          if (x < 0) { x = 0 }
          if (y < 0) { y = 0 }
          if ((x + w) > image_width) { w = image_width - x}
          if ((y + h) > image_height) { h = image_height - y}
          this.$emit('update:bbox', [[x, y], [x + w, y + h]]);
        });

      const viewBox = getViewBox(this.bboxEl.bbox());
      this.sceneEl.viewbox(viewBox)

      this.backgroundEl = this.sceneEl.image(this.imageUrl).loaded(() => {
        this.pointSize = POINT_SIZE * (viewBox.w / this.container.width.baseVal.value);
        this.initPoints();
      });
      this.group.add(this.backgroundEl, this.maskEl, this.bboxEl);

      this.pt = this.container.createSVGPoint();
      
      if (!this.pointsDisabled) {
        this.bboxEl.click(this.pointHandler);
      }
      // this.bboxEl.on('contextmenu', this.pointHandler);
    },
  },

  mounted() {
    this.pointsMap = new Map();
    this.container = this.$refs['container'];

    this.init();
  }
});