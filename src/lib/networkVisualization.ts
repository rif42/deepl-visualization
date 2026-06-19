import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { topKWeights } from './math';

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
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a);

    const aspect = container.clientWidth / container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    this.camera.position.set(0, 0, 40);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.animate();
  }

  buildLayers(layerSizes: number[], weights: Float32Array[]) {
    this.weightMatrices = weights;
    this.neuronMeshes.forEach((m) => this.scene.remove(m));
    this.neuronMeshes = [];
    if (this.inputPlane) {
      this.scene.remove(this.inputPlane);
      this.inputPlane = null;
    }
    if (this.connectionLines) {
      this.scene.remove(this.connectionLines);
      this.connectionLines = null;
    }

    const layerCount = layerSizes.length;
    const spacingX = 12;
    const startX = -((layerCount - 1) * spacingX) / 2;

    this.layerConfigs = layerSizes.map((size, index) => {
      const cols = Math.ceil(Math.sqrt(size));
      const rows = Math.ceil(size / cols);
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
    const inputMaterial = new THREE.MeshBasicMaterial({ color: 0x111111 });
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
        inputStartY + row * inputSpacing,
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
      const material = new THREE.MeshBasicMaterial({ color: 0x222222 });
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

  private renderConnections(k = 100) {
    if (this.connectionLines) {
      this.scene.remove(this.connectionLines);
      this.connectionLines = null;
    }

    if (this.weightMatrices.length === 0 || this.layerConfigs.length < 2) return;

    const positions: number[] = [];

    for (let layerIndex = 1; layerIndex < this.layerConfigs.length; layerIndex++) {
      const prevConfig = this.layerConfigs[layerIndex - 1];
      const currConfig = this.layerConfigs[layerIndex];
      const weightMatrix = this.weightMatrices[layerIndex - 1];

      const topK = topKWeights(weightMatrix, k);

      for (const entry of topK) {
        const srcIndex = entry.index % prevConfig.size;
        const dstIndex = Math.floor(entry.index / prevConfig.size);

        const srcPos = this.getNeuronPosition(prevConfig, srcIndex);
        const dstPos = this.getNeuronPosition(currConfig, dstIndex);

        positions.push(srcPos.x, srcPos.y, srcPos.z);
        positions.push(dstPos.x, dstPos.y, dstPos.z);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

    const material = new THREE.LineBasicMaterial({
      color: 0x666666,
      transparent: true,
      opacity: 0.25,
    });

    this.connectionLines = new THREE.LineSegments(geometry, material);
    this.scene.add(this.connectionLines);
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
    for (let i = 0; i < this.inputPlane.count; i++) {
      const pixelIndex = i * 4;
      const gray = imageData[pixelIndex] / 255;
      color.setRGB(gray, gray, gray);
      this.inputPlane.setColorAt(i, color);
    }
    this.inputPlane.instanceColor.needsUpdate = true;
  }

  updateActivations(activations: Float32Array[]) {
    const color = new THREE.Color();

    for (let i = 0; i < this.neuronMeshes.length; i++) {
      const mesh = this.neuronMeshes[i];
      const data = activations[i];
      if (!data) continue;

      for (let j = 0; j < mesh.count; j++) {
        const activation = data[j] ?? 0;
        color.setHSL(0.35 - activation * 0.35, 1.0, 0.2 + activation * 0.5);
        mesh.setColorAt(j, color);
      }

      mesh.instanceColor.needsUpdate = true;
    }
  }

  resize() {
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  dispose() {
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}
