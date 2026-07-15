import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { topKIndices, topKWeights, type WeightEntry } from './math';

interface LayerConfig {
  name: string;
  size: number;
  grid: { cols: number; rows: number };
  position: THREE.Vector3;
}

export class NetworkVisualization {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private neuronMeshes: THREE.InstancedMesh[] = [];
  private inputPlane: THREE.InstancedMesh | null = null;
  private connectionLines: THREE.LineSegments | null = null;
  private layerConfigs: LayerConfig[] = [];
  private weightMatrices: Float32Array[] = [];
  private inputActivations: Float32Array | null = null;
  private staticTopK: WeightEntry[][] = [];
  private container: HTMLElement;
  private animationFrameId: number = 0;

  constructor(container: HTMLElement) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x2A2A2A);

    const aspect = container.clientWidth / container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    this.camera.position.set(0, 0, 40);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    // Lights so that emissive neuron materials actually glow
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(10, 10, 20);
    this.scene.add(directionalLight);

    this.animate();
  }

  /**
   * Builds the 3D layer geometry and renders the strongest connections.
   *
   * @param layerSizes - Number of neurons in each layer (input + hidden + output).
   * @param weights - Flat weight matrices where `weights[i]` connects layer i to
   *   layer i+1. The flat layout is `index = dst * srcSize + src` (dst is the
   *   destination neuron in layer i+1 and src is the source neuron in layer i).
   */
  buildLayers(layerSizes: number[], weights: Float32Array[]) {
    this.weightMatrices = weights;

    this.neuronMeshes.forEach((m) => {
      this.scene.remove(m);
      this.disposeMesh(m);
    });
    this.neuronMeshes = [];

    if (this.inputPlane) {
      this.scene.remove(this.inputPlane);
      this.disposeMesh(this.inputPlane);
      this.inputPlane = null;
    }

    if (this.connectionLines) {
      this.scene.remove(this.connectionLines);
      this.disposeMesh(this.connectionLines);
      this.connectionLines = null;
    }

    const layerCount = layerSizes.length;
    const spacingX = 12;
    const startX = -((layerCount - 1) * spacingX) / 2;

    this.layerConfigs = layerSizes.map((size, index) => {
      const isOutput = index === layerSizes.length - 1;
      const cols = isOutput ? size : Math.ceil(Math.sqrt(size));
      const rows = isOutput ? 1 : Math.ceil(size / cols);
      return {
        name: `layer-${index}`,
        size,
        grid: { cols, rows },
        position: new THREE.Vector3(startX + index * spacingX, 0, 0),
      };
    });

    // Input layer: 28x28 grid of small cubes
    const inputConfig = this.layerConfigs[0];
    const inputGeometry = new THREE.BoxGeometry(0.25, 0.25, 0.25);
    const inputMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.inputPlane = new THREE.InstancedMesh(inputGeometry, inputMaterial, inputConfig.size);
    this.inputPlane.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    const inputDummy = new THREE.Object3D();
    const inputCols = 28;
    const inputRows = 28;
    const inputSpacing = 0.3;
    const inputStartY = -((inputRows - 1) * inputSpacing) / 2;
    const inputStartZ = -((inputCols - 1) * inputSpacing) / 2;

    for (let i = 0; i < inputConfig.size; i++) {
      const col = i % inputCols;
      const row = Math.floor(i / inputCols);
      inputDummy.position.set(
        inputConfig.position.x - 2,
        inputStartY + (inputRows - 1 - row) * inputSpacing,
        inputStartZ + col * inputSpacing
      );
      inputDummy.updateMatrix();
      this.inputPlane.setMatrixAt(i, inputDummy.matrix);
    }
    this.scene.add(this.inputPlane);

    // Hidden and output layers: spheres
    for (let i = 1; i < this.layerConfigs.length; i++) {
      const config = this.layerConfigs[i];
      const geometry = new THREE.SphereGeometry(0.35, 16, 16);
      // MeshBasicMaterial ignores scene lights so high instance colors look
      // like the neurons are self-illuminating / emitting light.
      const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const mesh = new THREE.InstancedMesh(geometry, material, config.size);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

      const dummy = new THREE.Object3D();
      const { cols, rows } = config.grid;
      const spacing = 1.2;
      const startY = -((rows - 1) * spacing) / 2;
      const startZ = -((cols - 1) * spacing) / 2;

      for (let j = 0; j < config.size; j++) {
        const col = j % cols;
        const row = Math.floor(j / cols);
        dummy.position.set(
          config.position.x,
          startY + row * spacing,
          startZ + col * spacing
        );
        dummy.updateMatrix();
        mesh.setMatrixAt(j, dummy.matrix);
      }

      this.scene.add(mesh);
      this.neuronMeshes.push(mesh);
    }

    this.renderConnections();
  }

  private disposeMesh(mesh: THREE.InstancedMesh | THREE.LineSegments | null) {
    if (!mesh) return;
    mesh.geometry.dispose();
    if (Array.isArray(mesh.material)) {
      mesh.material.forEach((m) => m.dispose());
    } else {
      mesh.material.dispose();
    }
  }

  private renderConnections(k = 100) {
    if (this.connectionLines) {
      this.scene.remove(this.connectionLines);
      this.disposeMesh(this.connectionLines);
      this.connectionLines = null;
    }

    if (this.weightMatrices.length === 0 || this.layerConfigs.length < 2) return;

    this.staticTopK = [];
    const positions: number[] = [];

    for (let layerIndex = 1; layerIndex < this.layerConfigs.length; layerIndex++) {
      const prevConfig = this.layerConfigs[layerIndex - 1];
      const currConfig = this.layerConfigs[layerIndex];
      const weightMatrix = this.weightMatrices[layerIndex - 1];

      const topK = topKWeights(weightMatrix, k);
      this.staticTopK.push(topK);

      for (const entry of topK) {
        const srcIndex = entry.index % prevConfig.size;
        const dstIndex = Math.floor(entry.index / prevConfig.size);

        const srcPos = this.getNeuronPosition(prevConfig, srcIndex);
        const dstPos = this.getNeuronPosition(currConfig, dstIndex);

        positions.push(srcPos.x, srcPos.y, srcPos.z);
        positions.push(dstPos.x, dstPos.y, dstPos.z);
      }
    }

    this.buildConnectionLines(positions, { color: 0x335533, opacity: 0.12 });
  }

  private buildConnectionLines(
    positions: number[],
    options: { color?: number; opacity?: number } = {}
  ) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

    const material = new THREE.LineBasicMaterial({
      color: options.color ?? 0x55aa55,
      transparent: true,
      opacity: options.opacity ?? 0.5,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.connectionLines = new THREE.LineSegments(geometry, material);
    this.scene.add(this.connectionLines);
  }

  /**
   * Recomputes the top-K connection lines from scratch on every prediction.
   * Each weight is scored by |weight| × sourceActivation × destinationActivation,
   * so the lines always reflect the most active pathways for the current input.
   */
  private updateConnections(activations: Float32Array[], k = 100) {
    if (this.connectionLines) {
      this.scene.remove(this.connectionLines);
      this.disposeMesh(this.connectionLines);
      this.connectionLines = null;
    }

    if (this.weightMatrices.length === 0 || this.layerConfigs.length < 2) return;

    const positions: number[] = [];

    for (let layerIndex = 1; layerIndex < this.layerConfigs.length; layerIndex++) {
      const prevConfig = this.layerConfigs[layerIndex - 1];
      const currConfig = this.layerConfigs[layerIndex];
      const weightMatrix = this.weightMatrices[layerIndex - 1];

      const srcActivations =
        layerIndex === 1
          ? this.inputActivations ?? new Float32Array(prevConfig.size)
          : activations[layerIndex - 2];
      const dstActivations = activations[layerIndex - 1];

      const scores = new Float32Array(weightMatrix.length);
      for (let i = 0; i < weightMatrix.length; i++) {
        const srcIndex = i % prevConfig.size;
        const dstIndex = Math.floor(i / prevConfig.size);
        scores[i] =
          Math.abs(weightMatrix[i]) *
          (srcActivations[srcIndex] ?? 0) *
          (dstActivations[dstIndex] ?? 0);
      }

      const topIndices = topKIndices(scores, k);

      for (const index of topIndices) {
        const srcIndex = index % prevConfig.size;
        const dstIndex = Math.floor(index / prevConfig.size);

        const srcPos = this.getNeuronPosition(prevConfig, srcIndex);
        const dstPos = this.getNeuronPosition(currConfig, dstIndex);

        positions.push(srcPos.x, srcPos.y, srcPos.z);
        positions.push(dstPos.x, dstPos.y, dstPos.z);
      }
    }

    this.buildConnectionLines(positions);
  }

  private getNeuronPosition(config: LayerConfig, index: number): THREE.Vector3 {
    if (config.name === 'layer-0') {
      const cols = 28;
      const rows = 28;
      const spacing = 0.3;
      const startY = -((rows - 1) * spacing) / 2;
      const startZ = -((cols - 1) * spacing) / 2;
      const col = index % cols;
      const row = Math.floor(index / cols);
      return new THREE.Vector3(
        config.position.x - 2,
        startY + row * spacing,
        startZ + col * spacing
      );
    }

    const { cols, rows } = config.grid;
    const spacing = 1.2;
    const startY = -((rows - 1) * spacing) / 2;
    const startZ = -((cols - 1) * spacing) / 2;
    const col = index % cols;
    const row = Math.floor(index / cols);

    return new THREE.Vector3(
      config.position.x,
      startY + row * spacing,
      startZ + col * spacing
    );
  }

  updateInputPlane(imageData: Uint8ClampedArray) {
    if (!this.inputPlane) return;

    const color = new THREE.Color();
    this.inputActivations = new Float32Array(this.inputPlane.count);
    for (let i = 0; i < this.inputPlane.count; i++) {
      const pixelIndex = i * 4;
      const gray = imageData[pixelIndex] / 255;
      this.inputActivations[i] = gray;
      color.setRGB(gray, gray, gray);
      this.inputPlane.setColorAt(i, color);
    }
    this.inputPlane.instanceColor.needsUpdate = true;
  }

  /**
   * Updates the neuron colors for each non-input layer from activation values.
   *
   * @param activations - Array where `activations[i]` corresponds to
   *   `neuronMeshes[i]` (i.e., layer i+1, skipping the input layer).
   */
  updateActivations(activations: Float32Array[]) {
    const color = new THREE.Color();

    for (let i = 0; i < this.neuronMeshes.length; i++) {
      const mesh = this.neuronMeshes[i];
      const data = activations[i];
      if (!data) continue;

      if (data.length !== mesh.count) {
        console.warn(
          `updateActivations: data length ${data.length} does not match mesh count ${mesh.count} for layer ${i}. Skipping.`
        );
        continue;
      }

      for (let j = 0; j < mesh.count; j++) {
        // Clamp activations because hidden-layer ReLU outputs can exceed 1.0.
        const activation = Math.min(1, Math.max(0, data[j] ?? 0));
        // Single green gradient: black → dark green → green → light green.
        color.setHSL(0.33, 1.0, activation * 0.75);
        mesh.setColorAt(j, color);
      }

      mesh.instanceColor.needsUpdate = true;
    }

    this.updateConnections(activations);
  }

  /**
   * Resets the visualization to its initial state. Called when the user
   * clears the drawing canvas.
   */
  clear() {
    const black = new THREE.Color(0x000000);

    if (this.inputPlane) {
      for (let i = 0; i < this.inputPlane.count; i++) {
        this.inputPlane.setColorAt(i, black);
      }
      this.inputPlane.instanceColor.needsUpdate = true;
    }

    this.neuronMeshes.forEach((mesh) => {
      for (let j = 0; j < mesh.count; j++) {
        mesh.setColorAt(j, black);
      }
      mesh.instanceColor.needsUpdate = true;
    });

    this.inputActivations = new Float32Array(784);
    this.renderConnections();
  }

  resize() {
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }

  private animate = () => {
    this.animationFrameId = requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  dispose() {
    cancelAnimationFrame(this.animationFrameId);

    this.neuronMeshes.forEach((mesh) => {
      this.scene.remove(mesh);
      this.disposeMesh(mesh);
    });
    this.neuronMeshes = [];

    if (this.inputPlane) {
      this.scene.remove(this.inputPlane);
      this.disposeMesh(this.inputPlane);
      this.inputPlane = null;
    }

    if (this.connectionLines) {
      this.scene.remove(this.connectionLines);
      this.disposeMesh(this.connectionLines);
      this.connectionLines = null;
    }

    this.controls.dispose();
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}
