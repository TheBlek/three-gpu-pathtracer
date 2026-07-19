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
let isBenchmarkRunning = false;

const resultsEl = document.getElementById( 'results' );
const savedRuns = [];

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
	model: Object.keys( MODELS )[ 0 ],

	// Stop condition
	targetTimeSeconds: 10, // TODO: remove?
	targetSampleCount: 64,

	// Buttons
	runBenchmark: onRunBenchmark,
	downloadCSV: downloadCSV,
};

async function onRunBenchmark() {

	if ( isBenchmarkRunning ) {

		return;

	}

	isBenchmarkRunning = true;

	const card = document.createElement( 'div' );
	card.className = 'card';
	card.textContent = 'Benchmarking...';
	resultsEl.prepend( card );

	const res = await runBenchmark();
	fillCard( card, res );

	if ( res.results ) {

		savedRuns.push( {
			...params,
			results: res.results,
		} );

	}

	isBenchmarkRunning = false;

}

function downloadCSV() {

	if ( savedRuns.length === 0 ) {

		return;

	}

	let csv = 'backend,useMegakernel,model,resolution,bounces,tileCount,targetSampleCount,warmupIterations,iterations,runIndex,totalSamples,elapsedMs,samplesPerSecond\n';

	for ( let i = 0; i < savedRuns.length; i ++ ) {

		const run = savedRuns[ i ];
		const backend = run.isWebGPU ? 'WebGPU' : 'WebGL';

		for ( let j = 0; j < run.results.length; j ++ ) {

			const totalSamples = run.results[ j ].totalSamples;
			const elapsedMs = run.results[ j ].elapsedMs;
			const samplesPerSecond = ( totalSamples * 1000 ) / elapsedMs;

			csv += backend + ',';
			csv += run.useMegakernel + ',';
			csv += '"' + run.model + '",';
			csv += run.resolution + ',';
			csv += run.bounces + ',';
			csv += run.tileCount + ',';
			csv += run.targetSampleCount + ',';
			csv += run.warmupIterations + ',';
			csv += run.iterations + ',';
			csv += ( j + 1 ) + ',';
			csv += totalSamples + ',';
			csv += elapsedMs + ',';
			csv += samplesPerSecond + '\n';

		}

	}

	const blob = new Blob( [ csv ], { type: 'text/csv' } );
	const url = URL.createObjectURL( blob );

	const anchor = document.createElement( 'a' );
	anchor.href = url;
	anchor.download = 'benchmark.csv';
	document.body.appendChild( anchor );
	anchor.click();
	anchor.remove();

	URL.revokeObjectURL( url );

}

function formatSamplesPerSecond( samplesPerSecond ) {

	if ( samplesPerSecond > 1000000 ) {

		return ( samplesPerSecond / 1000000 ).toFixed( 3 ) + ' Msamples/s';

	} else if ( samplesPerSecond > 1000 ) {

		return ( samplesPerSecond / 1000 ).toFixed( 3 ) + ' Ksamples/s';

	}

	return samplesPerSecond.toFixed( 3 ) + ' samples/s';

}

function fillCard( card, res ) {

	if ( res.error || ! res.results ) {

		card.classList.add( 'error' );
		card.textContent = ( res && res.error ) ? res.error : 'Benchmark failed';
		return;

	}

	const results = res.results;

	// Header: backend, kernel mode (WebGPU only), model, resolution, run count
	let header = params.isWebGPU ? 'WebGPU' : 'WebGL';
	if ( params.isWebGPU ) {

		header += params.useMegakernel ? ' · megakernel' : ' · wavefront';

	}

	header += ' · ' + params.model;
	header += ' · ' + params.resolution + 'px';
	header += ' · ' + params.targetSampleCount + ' samples';
	header += ' · ' + params.iterations + ' runs';

	let dataHtml = '<div class="header">' + header + '</div>';

	const throughputs = [];
	for ( let i = 0; i < results.length; i ++ ) {

		const totalSamples = results[ i ].totalSamples;
		const elapsedMs = results[ i ].elapsedMs;
		const samplesPerSecond = ( totalSamples * 1000 ) / elapsedMs;
		throughputs.push( samplesPerSecond );

		dataHtml += '<div class="run">';
		dataHtml += '#' + ( i + 1 ) + ': ';
		dataHtml += ( elapsedMs / 1000 ).toFixed( 3 ) + 's with ';
		dataHtml += formatSamplesPerSecond( samplesPerSecond );
		dataHtml += '</div>';

	}

	let avgThroughput = 0;
	for ( let i = 0; i < throughputs.length; i ++ ) {

		avgThroughput += throughputs[ i ];

	}

	avgThroughput /= throughputs.length;

	throughputs.sort( ( a, b ) => a - b );
	const medianThroughput = throughputs[ Math.floor( throughputs.length / 2 ) ];

	dataHtml += '<div class="summary">';
	dataHtml += 'avg ' + formatSamplesPerSecond( avgThroughput );
	dataHtml += ' · median ' + formatSamplesPerSecond( medianThroughput );
	dataHtml += '</div>';

	let html = '<div class="card-body">';
	html += '<div class="data">' + dataHtml + '</div>';

	if ( res.imageUrl ) {

		html += '<a class="preview-link" href="' + res.imageUrl + '" target="_blank" rel="noopener">';
		html += '<img class="preview" src="' + res.imageUrl + '" alt="benchmark preview" />';
		html += '</a>';

	}

	html += '</div>';

	card.innerHTML = html;

}

function captureImage( renderer ) {

	// Blob URL (not data:) so the preview link can open in a new tab.
	return new Promise( ( resolve ) => {

		renderer.domElement.toBlob( ( blob ) => {

			if ( ! blob ) {

				resolve( null );
				return;

			}

			resolve( URL.createObjectURL( blob ) );

		}, 'image/png' );

	} );

}

function areParamsValid() {

	return params.model in MODELS && ( params.targetSampleCount > 0 || params.targetTimeSeconds > 0 );

}

async function createRenderer( params ) {

	if ( params.isWebGPU ) {

		const renderer = new WebGPURenderer( { preserveDrawingBuffer: true } );
		await renderer.init();
		renderer.toneMapping = ACESFilmicToneMapping;
		renderer.setDrawingBufferSize( params.resolution, params.resolution, 1.0 );

		const pathtracer = new WebGPUPathTracer( renderer );
		pathtracer.useMegakernel( params.useMegakernel );
		pathtracer.tiles.set( params.tileCount, params.tileCount );
		pathtracer.bounces = params.bounces;
		pathtracer.setSize( params.resolution, params.resolution );
		pathtracer.dynamicLowRes = false;
		pathtracer.renderDelay = 0;

		return { renderer, pathtracer };

	} else {

		const renderer = new WebGLRenderer( { preserveDrawingBuffer: true } );
		renderer.toneMapping = ACESFilmicToneMapping;
		renderer.setDrawingBufferSize( params.resolution, params.resolution, 1.0 );

		const pathtracer = new WebGLPathTracer( renderer );
		pathtracer.tiles.set( params.tileCount, params.tileCount );
		pathtracer.bounces = params.bounces;
		pathtracer.rasterizeScene = false;

		return { renderer, pathtracer };

	}

}

function cleanup( renderer, pathtracer ) {

	pathtracer.dispose();
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

			// TODO: this does not seem to work?
			// FIX
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

	await waitGpuIdle();

	const end = performance.now();

	const elapsedMs = end - start;
	const samples = await pathtracer.getDetailedSampleCount();

	return { totalSamples: samples.total, elapsedMs };

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

		cleanup( renderer, pathtracer );
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

	const imageUrl = await captureImage( renderer );

	disposeModel( model );

	cleanup( renderer, pathtracer );

	return { results, imageUrl };

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
	gui.add( params, 'downloadCSV' );

}

buildGUI();
