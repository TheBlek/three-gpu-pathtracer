import {
	ACESFilmicToneMapping,
	NoToneMapping,
	DoubleSide,
	Mesh,
	MeshStandardMaterial,
	PlaneGeometry,
	Scene,
	PerspectiveCamera,
	OrthographicCamera,
	WebGPURenderer,
	EquirectangularReflectionMapping,
} from 'three/webgpu';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { generateRadialFloorTexture } from './utils/generateRadialFloorTexture.js';
import { GradientEquirectTexture } from 'three-gpu-pathtracer';
import { WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { getScaledSettings } from './utils/getScaledSettings.js';
import { LoaderElement } from './utils/LoaderElement.js';
import { disposeModel, loadModelToScene, MODELS } from './utils/ModelLibrary.js';
import { ENV_MAPS } from './utils/EnvMaps.js';

const params = {

	multipleImportanceSampling: true,
	acesToneMapping: true,
	renderScale: 1 / window.devicePixelRatio,
	tiles: 2,

	model: '',

	envMap: ENV_MAPS[ 'Aristea Wreck Puresky' ],

	gradientTop: '#bfd8ff',
	gradientBottom: '#ffffff',

	environmentIntensity: 1.0,
	environmentRotation: 0,

	cameraProjection: 'Perspective',

	backgroundType: 'Gradient',
	bgGradientTop: '#111111',
	bgGradientBottom: '#000000',
	backgroundBlur: 0.0,
	transparentBackground: false,
	checkerboardTransparency: true,

	enable: true,
	useMegakernel: false,
	bounces: 5,
	filterGlossyFactor: 0.5,
	pause: false,
	debugBounds: false,
	displayTLAS: true,
	displayBLAS: true,
	stopAtSurface: false,
	saturationCount: 64,

	floorColor: '#111111',
	floorOpacity: 1.0,
	floorRoughness: 0.2,
	floorMetalness: 0.2,

	...getScaledSettings(),

};

let floorPlane, gui, stats;
let pathTracer, renderer, orthoCamera, perspectiveCamera, activeCamera;
let controls, scene, model;
let gradientMap;
let loader;
let models;

const orthoWidth = 2;

init();

async function init() {

	models = MODELS;

	loader = new LoaderElement();
	loader.attach( document.body );

	// renderer
	renderer = new WebGPURenderer( { antialias: true } );
	renderer.init();
	renderer.toneMapping = ACESFilmicToneMapping;
	document.body.appendChild( renderer.domElement );

	// path tracer
	pathTracer = new WebGPUPathTracer( renderer );
	pathTracer.tiles.set( params.tiles, params.tiles );
	pathTracer.multipleImportanceSampling = params.multipleImportanceSampling;
	pathTracer.transmissiveBounces = 10;
	pathTracer.useMegakernel( params.useMegakernel );

	// camera
	const aspect = window.innerWidth / window.innerHeight;
	perspectiveCamera = new PerspectiveCamera( 60, aspect, 0.025, 500 );
	perspectiveCamera.position.set( - 1, 0.25, 1 );

	const orthoHeight = orthoWidth / aspect;
	orthoCamera = new OrthographicCamera( orthoWidth / - 2, orthoWidth / 2, orthoHeight / 2, orthoHeight / - 2, 0, 100 );
	orthoCamera.position.set( - 1, 0.25, 1 );

	// background map
	gradientMap = new GradientEquirectTexture();
	gradientMap.topColor.set( params.bgGradientTop );
	gradientMap.bottomColor.set( params.bgGradientBottom );
	gradientMap.update();

	// controls
	controls = new OrbitControls( perspectiveCamera, renderer.domElement );
	controls.addEventListener( 'change', () => {

		pathTracer.updateCamera();

	} );

	// scene
	scene = new Scene();
	scene.background = gradientMap;

	const floorTex = generateRadialFloorTexture( 2048 );
	floorPlane = new Mesh(
		new PlaneGeometry(),
		new MeshStandardMaterial( {
			map: floorTex,
			transparent: true,
			color: 0x111111,
			roughness: 0.1,
			metalness: 0.0,
			side: DoubleSide,
		} )
	);
	floorPlane.scale.setScalar( 5 );
	floorPlane.rotation.x = - Math.PI / 2;
	scene.add( floorPlane );

	stats = new Stats();
	document.body.appendChild( stats.dom );

	updateCameraProjection( params.cameraProjection );
	onHashChange();
	updateEnvMap();
	onResize();

	animate();

	window.addEventListener( 'resize', onResize );
	window.addEventListener( 'hashchange', onHashChange );

}

function animate() {

	requestAnimationFrame( animate );

	stats.update();

	if ( ! model ) {

		return;

	}

	if ( params.debugBounds ) {

		pathTracer.renderDebugBounds( {
			displayTLAS: params.displayTLAS,
			displayBLAS: params.displayBLAS,
			stopAtSurface: params.stopAtSurface,
			saturationCount: params.saturationCount,
		} );

	} else if ( params.enable ) {

		if ( ! params.pause || pathTracer.samples < 1 ) {

			pathTracer.renderSample();

		}

	} else {

		renderer.render( scene, activeCamera );

	}

	loader.setSamples( pathTracer.samples );

}

function onParamsChange() {

	pathTracer.multipleImportanceSampling = params.multipleImportanceSampling;
	pathTracer.bounces = params.bounces;
	pathTracer.filterGlossyFactor = params.filterGlossyFactor;
	pathTracer.renderScale = params.renderScale;

	floorPlane.material.color.set( params.floorColor );
	floorPlane.material.roughness = params.floorRoughness;
	floorPlane.material.metalness = params.floorMetalness;
	floorPlane.material.opacity = params.floorOpacity;

	scene.environmentIntensity = params.environmentIntensity;
	scene.environmentRotation.y = params.environmentRotation;
	scene.backgroundBlurriness = params.backgroundBlur;

	if ( params.backgroundType === 'Gradient' ) {

		gradientMap.topColor.set( params.bgGradientTop );
		gradientMap.bottomColor.set( params.bgGradientBottom );
		gradientMap.update();

		scene.background = gradientMap;
		scene.backgroundIntensity = 1;
		scene.environmentRotation.y = 0;

	} else {

		scene.background = scene.environment;
		scene.backgroundIntensity = params.environmentIntensity;
		scene.backgroundRotation.y = params.environmentRotation;

	}

	if ( params.transparentBackground ) {

		scene.background = null;
		renderer.setClearAlpha( 0 );

	}

	pathTracer.updateMaterials();
	pathTracer.updateEnvironment();

}

function onHashChange() {

	let hashModel = '';
	if ( window.location.hash ) {

		const modelName = decodeURI( window.location.hash.substring( 1 ) );
		if ( modelName in models ) {

			hashModel = modelName;

		}

	}

	if ( ! ( hashModel in models ) ) {

		hashModel = Object.keys( models )[ 0 ];

	}

	params.model = hashModel;
	updateModel();

}

function onResize() {

	const w = window.innerWidth;
	const h = window.innerHeight;
	const dpr = window.devicePixelRatio;

	renderer.setSize( w, h );
	renderer.setPixelRatio( dpr );

	const aspect = w / h;
	perspectiveCamera.aspect = aspect;
	perspectiveCamera.updateProjectionMatrix();

	const orthoHeight = orthoWidth / aspect;
	orthoCamera.top = orthoHeight / 2;
	orthoCamera.bottom = orthoHeight / - 2;
	orthoCamera.updateProjectionMatrix();

	pathTracer.updateCamera();

}

function buildGui() {

	if ( gui ) {

		gui.destroy();

	}

	gui = new GUI();

	gui.add( params, 'model', Object.keys( models ).sort() ).onChange( v => {

		window.location.hash = v;

	} );

	const pathTracingFolder = gui.addFolder( 'Path Tracer' );
	pathTracingFolder.add( params, 'enable' );
	pathTracingFolder.add( params, 'pause' );
	pathTracingFolder.add( params, 'useMegakernel' ).onChange( () => {

		pathTracer.useMegakernel( params.useMegakernel );
		pathTracer.setScene( scene, activeCamera );
		pathTracer.reset();

	} );
	pathTracingFolder.add( params, 'multipleImportanceSampling' ).onChange( onParamsChange );
	pathTracingFolder.add( params, 'acesToneMapping' ).onChange( v => {

		renderer.toneMapping = v ? ACESFilmicToneMapping : NoToneMapping;

	} );
	pathTracingFolder.add( params, 'bounces', 1, 20, 1 ).onChange( onParamsChange );
	pathTracingFolder.add( params, 'filterGlossyFactor', 0, 1 ).onChange( onParamsChange );
	pathTracingFolder.add( params, 'renderScale', 0.1, 1.0, 0.01 ).onChange( () => {

		onParamsChange();

	} );
	pathTracingFolder.add( params, 'tiles', 1, 10, 1 ).onChange( v => {

		pathTracer.tiles.set( v, v );

	} );
	pathTracingFolder.add( params, 'cameraProjection', [ 'Perspective', 'Orthographic' ] ).onChange( v => {

		updateCameraProjection( v );

	} );
	pathTracingFolder.open();

	const debugFolder = gui.addFolder( 'debug' );
	debugFolder.add( params, 'debugBounds' ).name( 'bvh bounds heatmap' );
	debugFolder.add( params, 'displayTLAS' ).name( 'display TLAS' );
	debugFolder.add( params, 'displayBLAS' ).name( 'display BLAS' );
	debugFolder.add( params, 'stopAtSurface' ).name( 'stop at surface' );
	debugFolder.add( params, 'saturationCount', 1, 512, 1 ).name( 'saturation count' );

	const environmentFolder = gui.addFolder( 'environment' );
	environmentFolder.add( params, 'envMap', ENV_MAPS ).name( 'map' ).onChange( updateEnvMap );
	environmentFolder.add( params, 'environmentIntensity', 0.0, 10.0 ).onChange( onParamsChange ).name( 'intensity' );
	environmentFolder.add( params, 'environmentRotation', 0, 2 * Math.PI ).onChange( onParamsChange );
	environmentFolder.open();

	const backgroundFolder = gui.addFolder( 'background' );
	backgroundFolder.add( params, 'backgroundType', [ 'Environment', 'Gradient' ] ).onChange( onParamsChange );
	backgroundFolder.addColor( params, 'bgGradientTop' ).onChange( onParamsChange );
	backgroundFolder.addColor( params, 'bgGradientBottom' ).onChange( onParamsChange );
	backgroundFolder.add( params, 'backgroundBlur', 0, 1 ).onChange( onParamsChange );
	backgroundFolder.add( params, 'transparentBackground', 0, 1 ).onChange( onParamsChange );
	backgroundFolder.add( params, 'checkerboardTransparency' ).onChange( v => {

		if ( v ) document.body.classList.add( 'checkerboard' );
		else document.body.classList.remove( 'checkerboard' );

	} );

	const floorFolder = gui.addFolder( 'floor' );
	floorFolder.addColor( params, 'floorColor' ).onChange( onParamsChange );
	floorFolder.add( params, 'floorRoughness', 0, 1 ).onChange( onParamsChange );
	floorFolder.add( params, 'floorMetalness', 0, 1 ).onChange( onParamsChange );
	floorFolder.add( params, 'floorOpacity', 0, 1 ).onChange( onParamsChange );
	floorFolder.close();

}

function updateEnvMap() {

	new HDRLoader()
		.load( params.envMap, texture => {

			if ( scene.environment ) {

				scene.environment.dispose();

			}

			texture.mapping = EquirectangularReflectionMapping;
			scene.environment = texture;
			pathTracer.updateEnvironment();
			onParamsChange();

		} );

}

function updateCameraProjection( cameraProjection ) {

	// sync position
	if ( activeCamera ) {

		perspectiveCamera.position.copy( activeCamera.position );
		orthoCamera.position.copy( activeCamera.position );

	}

	// set active camera
	if ( cameraProjection === 'Perspective' ) {

		activeCamera = perspectiveCamera;

	} else {

		activeCamera = orthoCamera;

	}

	controls.object = activeCamera;
	controls.update();

	pathTracer.setCamera( activeCamera );

}

async function updateModel() {

	if ( gui ) {

		document.body.classList.remove( 'checkerboard' );
		gui.destroy();
		gui = null;

	}

	if ( model ) {

		disposeModel( model );

	}

	const modelInfo = models[ params.model ];

	renderer.domElement.style.visibility = 'hidden';

	const onProgress = ( v ) => loader.setPercentage( 0.5 * v );

	const { model: newModel, box, error } = await loadModelToScene( scene, renderer, modelInfo, onProgress );
	model = newModel;


	if ( error ) {

		loader.setCredits( 'Failed to load model:' + error );
		loader.setPercentage( 1 );
		return;

	}

	floorPlane.position.y = box.min.y;

	pathTracer.setScene( scene, activeCamera );

	loader.setPercentage( 1 );
	loader.setCredits( modelInfo.credit || '' );
	params.bounces = modelInfo.bounces || 5;
	params.floorColor = modelInfo.floorColor || '#111111';
	params.floorRoughness = modelInfo.floorRoughness || 0.2;
	params.floorMetalness = modelInfo.floorMetalness || 0.2;
	params.bgGradientTop = modelInfo.gradientTop || '#111111';
	params.bgGradientBottom = modelInfo.gradientBot || '#000000';

	buildGui();
	onParamsChange();

	renderer.domElement.style.visibility = 'visible';
	if ( params.checkerboardTransparency ) {

		document.body.classList.add( 'checkerboard' );

	}

}
