import { ACESFilmicToneMapping, EquirectangularReflectionMapping, PerspectiveCamera, Scene, WebGLRenderer } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { WebGPUPathTracer } from '../src/webgpu';
import { WebGLPathTracer } from '../src';
import GUI from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { disposeModel, loadModelToScene, MODELS } from './utils/ModelLibrary';
import { HDRLoader } from 'three/examples/jsm/Addons.js';
import { ENV_MAPS } from './utils/EnvMaps';

// TODO: make a tool that will go through a matrix of configurations,
// Plot avg sample count / pixel over time for two implementation (median + min/max + line that connects median values)
// Add ability to checkout a repository to run the benchmark on that data

let gui;

const params = {
	// Run settings
	warmupIterations: 3,
	iterations: 5,

	// Rendering configuration
	isWebGPU: false,
	useMegakernel: true,

	bounces: 10,
	tileCount: 3,

	resolution: 1024,
	model: '',

	// Stop condition
	targetTimeSeconds: 10, // TODO: remove?
	targetSampleCount: 64,

	// Button property
	runBenchmark: async function () {

		// TODO: display results
		const res = await runBenchmark();
		console.log( res );

	},

};

params.model = Object.keys( MODELS )[ 0 ];

function areParamsValid() {

	return params.model in MODELS && ( params.targetSampleCount > 0 || params.targetTimeSeconds > 0 );

}

async function createRenderer( params ) {

	if ( params.isWebGPU ) {

		const renderer = new WebGPURenderer();
		await renderer.init();
		renderer.toneMapping = ACESFilmicToneMapping;
		renderer.setDrawingBufferSize( params.resolution, params.resolution, 1.0 );
		document.body.append( renderer.domElement );

		const pathtracer = new WebGPUPathTracer( renderer );
		pathtracer.useMegakernel( params.useMegakernel );
		pathtracer.tiles.set( params.tileCount, params.tileCount );
		pathtracer.bounces = params.bounces;
		pathtracer.setSize( params.resolution, params.resolution );
		pathtracer.dynamicLowRes = false;
		pathtracer.renderDelay = 0;

		return { renderer, pathtracer };

	} else {

		const renderer = new WebGLRenderer();
		renderer.toneMapping = ACESFilmicToneMapping;
		renderer.setDrawingBufferSize( params.resolution, params.resolution, 1.0 );
		document.body.append( renderer.domElement );

		const pathtracer = new WebGLPathTracer( renderer );
		pathtracer.tiles.set( params.tileCount, params.tileCount );
		pathtracer.bounces = params.bounces;
		pathtracer.rasterizeScene = false;

		return { renderer, pathtracer };

	}

}

function cleanup( renderer, pathtracer ) {

	pathtracer.dispose();
	renderer.domElement.remove();
	renderer.dispose();

}

async function runIteration( renderer, pathtracer, params ) {

	// Schedule N renderFrames in pathtracer; What's N for wavefront?
	// Fence
	// Get actual information
	// Calcuate resulting throughput
	// Return

	const waitGpuIdle = async () => {

		if ( params.isWebGPU ) {

			await renderer.backend.device.queue.onSubmittedWorkDone();

		} else {

			const gl = renderer.getContext();
			const sync = gl.fenceSync( gl.SYNC_GPU_COMMANDS_COMPLETE, 0 );
			gl.flush();
			// poll with 0 timeout so you don't hard-block the main thread forever
			while ( gl.clientWaitSync( sync, 0, 0 ) === gl.TIMEOUT_EXPIRED ) {

				await new Promise( r => setTimeout( r, 0 ) );

			}

			gl.deleteSync( sync );

		}

	};

	let iterationCount = params.tileCount * params.tileCount * params.targetSampleCount;
	if ( params.isWebGPU && ! params.useMegakernel ) {

		iterationCount = ( params.bounces / 2 ) * params.targetSampleCount * params.resolution * params.resolution / 250000;

	}

	await waitGpuIdle();

	const start = performance.now();
	for ( let i = 0; i < iterationCount; i ++ ) {

		pathtracer.renderSample();

	}

	const cpuEnd = performance.now();

	await waitGpuIdle();

	const end = performance.now();

	const elapsedMs = end - start;
	const samples = await pathtracer.getDetailedSampleCount();
	const samplesPerSecond = ( samples.total * 1000 ) / elapsedMs;

	console.log( `cpu time: ${ cpuEnd - start }; gpu time: ${ end - start }` );

	return { totalSamples: samples.total, elapsedMs, samplesPerSecond };

}

async function runBenchmark() {

	if ( ! areParamsValid() ) {

		return { error: 'Invalid params' };

	}

	const { renderer, pathtracer } = await createRenderer( params );

	const scene = new Scene();

	const envMapPromise = new HDRLoader().loadAsync( ENV_MAPS[ 'Measuring Lab' ] );

	const { model, box, error } = await loadModelToScene( scene, renderer, MODELS[ params.model ], () => {} );

	if ( error ) {

		return { error };

	}

	const aspect = 1; // Benchmarking on squares for now
	const perspectiveCamera = new PerspectiveCamera( 60, aspect, 0.025, 500 );
	perspectiveCamera.position.set( - 1, 0.25, 1 );
	perspectiveCamera.lookAt( 0, 0, 0 );
	perspectiveCamera.updateMatrixWorld();

	const envMap = await envMapPromise;
	envMap.mapping = EquirectangularReflectionMapping;
	scene.environment = envMap;

	pathtracer.setScene( scene, perspectiveCamera );

	for ( let wIter = 0; wIter < params.warmupIterations; wIter ++ ) {

		pathtracer.reset();
		await runIteration( renderer, pathtracer, params );

	}

	const results = [];
	for ( let iter = 0; iter < params.iterations; iter ++ ) {

		pathtracer.reset();
		const res = await runIteration( renderer, pathtracer, params );
		results.push( res );

	}

	disposeModel( model );

	cleanup( renderer, pathtracer );

	return results;

}

function buildGUI() {

	if ( gui ) {

		gui.destroy();

	}

	gui = new GUI();

	const runSettings = gui.addFolder( 'Run Settings' );
	runSettings.add( params, 'warmupIterations' );
	runSettings.add( params, 'iterations' );

	const stopCondition = gui.addFolder( 'Stop Condition' );
	stopCondition.add( params, 'targetTimeSeconds' );
	stopCondition.add( params, 'targetSampleCount' );

	const renderingSettings = gui.addFolder( 'Rendering Settings' );
	renderingSettings.add( params, 'isWebGPU' );
	renderingSettings.add( params, 'useMegakernel' );

	renderingSettings.add( params, 'tileCount' );

	renderingSettings.add( params, 'resolution' );
	renderingSettings.add( params, 'bounces', 1, 30 );

	const sceneSettings = gui.addFolder( 'Scene Settings' );
	sceneSettings.add( params, 'model', Object.keys( MODELS ).sort() ).onChange( v => {

		window.location.hash = v;

	} );

	// TODO: physical camera settings?

	gui.add( params, 'runBenchmark' );

}

buildGUI();
