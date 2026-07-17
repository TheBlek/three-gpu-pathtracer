import { ACESFilmicToneMapping, Vector2, WebGLRenderer } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { WebGPUPathTracer } from '../src/webgpu';
import { WebGLPathTracer } from '../src';
import GUI from 'three/examples/jsm/libs/lil-gui.module.min.js';

const modelLibrary = {

};

let gui;

const params = {
	// Run settings
	warmupIterations: 3,
	iterations: 5,

	// Rendering configuration
	isWebGPU: false,
	useMegakernel: true,

	bounce: 10,
	tileCount: 3,
	iterationsPerFrame: 1,

	resolution: 1024,
	model: '',

	// Stop condition
	targetTimeSeconds: 10,
	targetSampleCount: 64,

	// Button property
	runBenchmark: async function () {

		// TODO: display results
		const res = await runBenchmark();
		console.log( res );

	},

};

function areParamsValid() {

	return params.model in modelLibrary && ( params.targetSampleCount > 0 || params.targetTimeSeconds > 0 );

}

function createRenderer( params ) {

	if ( params.isWebGPU ) {

		const renderer = new WebGPURenderer();
		renderer.init();
		renderer.toneMapping = ACESFilmicToneMapping;
		renderer.setDrawingBufferSize( params.resolution, params.resolution, 1.0 );
		document.body.append( renderer.domElement );

		const pathtracer = new WebGPUPathTracer( renderer );
		pathtracer.tiles.set( params.tileCount, params.tileCount );
		pathtracer.bounces = params.bounces;
		pathtracer.useMegakernel( params.useMegakernel );
		pathtracer.setSize( params.resolution, params.resolution );

		return { renderer, pathtracer };

	} else {

		const renderer = new WebGLRenderer();
		renderer.toneMapping = ACESFilmicToneMapping;
		renderer.setDrawingBufferSize( params.resolution, params.resolution, 1.0 );
		document.body.append( renderer.domElement );

		const pathtracer = new WebGLPathTracer( renderer );
		pathtracer.tiles.set( params.tileCount, params.tileCount );
		pathtracer.bounces = params.bounces;

		return { renderer, pathtracer };

	}

}

function runIteration( pathtracer, params ) {

	return new Promise( ( resolve ) => {

		const startTime = performance.now();

		const shouldFinish = () => {

			if ( params.targetSampleCount > 0 && pathtracer.samples > params.targetSampleCount ) {

				return true;

			}

			const elapsedSeconds = ( performance.now() - startTime ) / 1000;
			if ( params.targetTimeSeconds > 0 && elapsedSeconds > params.targetTimeSeconds ) {

				return true;

			}

			return false;

		};

		const frame = () => {

			if ( shouldFinish() ) {

				// TODO: wait for gpu commands to finish
				const endTime = performance.now();
				const elapsedMs = endTime - startTime;
				resolve( { elapsedMs } );

			}

			requestAnimationFrame( frame );

			for ( let i = 0; i < params.iterationsPerFrame; i ++ ) {

				pathtracer.renderSample();

			}

		};

		frame();


	} );

}

async function runBenchmark() {

	if ( ! areParamsValid() ) {

		return { error: 'Invalid params' };

	}

	const { renderer, pathtracer } = createRenderer( params );

	for ( let wIter = 0; wIter < params.warmupIterations; wIter ++ ) {

		await runIteration( pathtracer, params );

	}

	const results = [];
	for ( let iter = 0; iter < params.iterations; iter ++ ) {

		const res = await runIteration( pathtracer, params );
		results.push( res.elapsedMs );

	}

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

	renderingSettings.add( params, 'model', Object.keys( modelLibrary ).sort() ).onChange( v => {

		window.location.hash = v;

	} );
	renderingSettings.add( params, 'isWebGPU' );
	renderingSettings.add( params, 'useMegakernel' );

	renderingSettings.add( params, 'tileCount' );
	renderingSettings.add( params, 'iterationsPerFrame' );

	renderingSettings.add( params, 'resolution' );

	// TODO: physical camera settings?

	gui.add( params, 'runBenchmark' );

}

buildGUI();
