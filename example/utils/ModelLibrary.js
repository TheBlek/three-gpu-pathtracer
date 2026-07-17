import { MeshPhysicalMaterial, Color, DoubleSide, Mesh, CylinderGeometry, Box3, LoadingManager, MeshStandardMaterial } from 'three';
import { FogVolumeMaterial } from 'three-gpu-pathtracer';
import { ColladaLoader, GLTFLoader, LDrawConditionalLineMaterial, LDrawLoader, LDrawUtils } from 'three/examples/jsm/Addons.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

export const MODELS = {
	'M2020 Rover': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/nasa-m2020/Perseverance.glb',
		credit: 'Model credit NASA / JPL-Caltech',
	},
	'M2020 Helicopter': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/nasa-m2020/Ingenuity.glb',
		credit: 'Model credit NASA / JPL-Caltech',
	},
	'Stalenhag Winter': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/colourdrafts/scene.glb',
		credit: 'Model by "ganzhugav" on Sketchfab',
		bounces: 3,
		postProcess( model ) {

			const box = new Box3();
			box.setFromObject( model );

			const fog = new Mesh( new CylinderGeometry( 0.5, 0.5, 1, 20 ), new FogVolumeMaterial( {

				color: 0xaaaaaa,
				density: 1,

			} ) );

			fog.scale.subVectors( box.max, box.min );
			fog.scale.x += 0.1;
			fog.scale.y += 0.01;
			fog.scale.z += 0.1;

			fog.scale.x = fog.scale.z = Math.max( fog.scale.x, fog.scale.z );

			box.getCenter( fog.position );
			fog.position.y += 0.02;

			model.traverse( c => {

				if ( c.material && c.material.emissive.r < 0.1 ) {

					c.material.emissive.set( 0 );

				}

			} );

			model.add( fog );

		}
	},
	'Gelatinous Cube': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/gelatinous-cube/scene.gltf',
		credit: 'Model by "glenatron" on Sketchfab.',
		rotation: [ 0, - Math.PI / 8, 0.0 ],
		opacityToTransmission: true,
		bounces: 8,
		postProcess( model ) {

			const toRemove = [];
			model.traverse( c => {

				if ( c.material ) {

					if ( c.material instanceof MeshPhysicalMaterial ) {

						const material = c.material;
						material.metalness = 0.0;
						material.ior = 1.2;
						material.map = null;

						c.geometry.computeVertexNormals();

					} else if ( c.material.opacity < 1.0 ) {

						toRemove.push( c );

					}

				}

			} );

			toRemove.forEach( c => {

				c.parent.remove( c );

			} );

		}
	},
	'Octopus Tea': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/octopus-tea/scene.gltf',
		credit: 'Model by "AzTiZ" on Sketchfab.',
		opacityToTransmission: true,
		bounces: 8,
		postProcess( model ) {

			const toRemove = [];

			model.updateMatrixWorld();

			model.traverse( c => {

				if ( c.material ) {

					c.material.emissiveIntensity = 0;
					if ( c.material instanceof MeshPhysicalMaterial ) {

						const material = c.material;
						material.metalness = 0.0;
						if ( material.transmission === 1.0 ) {

							material.roughness = 0.0;
							material.metalness = 0.0;

							// 29 === glass
							// 27 === liquid top
							// 23 === liquid
							if ( c.name.includes( '29' ) ) {

								material.ior = 1.52;
								material.color.set( 0xffffff );

							} else {

								material.ior = 1.2;

							}

						}

					} else if ( c.material.opacity < 1.0 ) {

						toRemove.push( c );

					}

				}

			} );

			toRemove.forEach( c => {

				c.parent.remove( c );

			} );

		}
	},
	'Scifi Toad': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/scifi-toad/scene.gltf',
		credit: 'Model by "YuryTheCreator" on Sketchfab.',
		opacityToTransmission: true,
		bounces: 8,
		postProcess( model ) {

			model.traverse( c => {

				if ( c.material && c.material instanceof MeshPhysicalMaterial ) {

					const material = c.material;
					material.metalness = 0.0;
					material.ior = 1.645;
					material.color.lerp( new Color( 0xffffff ), 0.65 );

				}

			} );

		}
	},
	'Halo Twist Ring': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/ring-twist-halo/scene.glb',
		credit: 'Model credit NASA / JPL-Caltech',
		opacityToTransmission: true,
		bounces: 15,
		postProcess( model ) {

			model.traverse( c => {

				if ( c.material ) {

					if ( c.material instanceof MeshPhysicalMaterial ) {

						if ( c.material.transmission === 1.0 ) {

							const material = c.material;
							material.metalness = 0.0;
							material.ior = 1.8;
							material.color.set( 0xffffff );

						}

					}

				}

			} );

		}
	},
	// 'Vino Bike': {
	// 	url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/vino-bike/scene.gltf',
	// 	credit: 'glTF Sample Model.',
	// 	postProcess( model ) {

	// 		model.traverse( c => {
	// 			console.log(c.name);
	// 			if ( c.name === 'mesh_0') {

	// 				// TODO: remove this
	// 				c.material.clearcoatRoughness = 0;
	// 				c.material.clearcoatMap = null;
	// 				c.material.clearcoatNormalMap = null;
	// 				c.material.clearcoatNormalScale.setScalar( 1. );

	// 			}
	// 		})

	// 	}
	// },
	'Damaged Helmet': {
		url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf',
		credit: 'glTF Sample Model.',
	},
	'Flight Helmet': {
		url: 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/FlightHelmet/glTF/FlightHelmet.gltf',
		credit: 'glTF Sample Model.',
	},
	'Statue': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/threedscans/Le_Transi_De_Rene_De_Chalon.glb',
		credit: 'Model courtesy of threedscans.com.',
	},
	'Crab Sculpture': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/threedscans/Crab.glb',
		rotation: [ - 2 * Math.PI / 4, 0, 0 ],
		credit: 'Model courtesy of threedscans.com.',

		bounces: 15,
		floorColor: '#eeeeee',
		floorRoughness: 1.0,
		floorMetalness: 0.0,
		gradientTop: '#eeeeee',
		gradientBot: '#eeeeee',

		postProcess( model ) {

			const mat = new MeshPhysicalMaterial( {
				roughness: 0.05,
				transmission: 1,
				ior: 1.2,
				attenuationDistance: 0.06,
				attenuationColor: 0x46dfea
			} );

			model.traverse( c => {

				if ( c.material ) c.material = mat;

			} );

		}
	},
	'Elbow Crab Sculpture': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/threedscans/Elbow_Crab.glb',
		rotation: [ 2.5 * Math.PI / 4, Math.PI, 0 ],
		credit: 'Model courtesy of threedscans.com.',

		bounces: 15,
		floorColor: '#eeeeee',
		floorRoughness: 1.0,
		floorMetalness: 0.0,
		gradientTop: '#eeeeee',
		gradientBot: '#eeeeee',

		postProcess( model ) {

			const mat = new MeshPhysicalMaterial( {
				color: 0xcc8888,
				roughness: 0.25,
				transmission: 1,
				ior: 1.5,
				side: DoubleSide,
			} );

			model.traverse( c => {

				if ( c.material ) c.material = mat;

			} );

		}
	},
	'Japanese Bridge Garden': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/japanese-bridge-garden/scene.glb',
		credit: 'Model by "kristenlee" on Sketchfab.',
	},
	'Imaginary Friend Room': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/imaginary-friend-room/scene.glb',
		credit: 'Model by "Iman Aliakbar" on Sketchfab.',
	},
	'Botanists Study': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/botanists-study/scene.gltf',
		credit: 'Model by "riikkakilpelainen" on Sketchfab.',
	},
	'Botanists Greenhouse': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/botanists-greenhouse/scene.gltf',
		credit: 'Model by "riikkakilpelainen" on Sketchfab.',
	},
	'Low Poly Rocket': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/lowpoly-space/space_exploration.glb',
		credit: 'Model by "The Sinking Sun" on Sketchfab',
		rotation: [ 0, - Math.PI / 3, 0.0 ],
	},
	'Astraia': {
		url: 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/astraia/scene.gltf',
		credit: 'Model by "Quentin Otani" on Sketchfab',
		removeEmission: true,
		postProcess( model ) {

			const toRemove = [];
			model.traverse( c => {

				if ( c.name.includes( 'ROND' ) ) {

					toRemove.push( c );

				}

			} );

			toRemove.forEach( c => {

				c.parent.remove( c );

			} );

		}
	},

};

function disposeModel( model ) {

	const parent = model.parent;
	parent.remove( model );

	model.traverse( ( obj ) => {

		obj.geometry?.dispose();

		const materials = obj.material ? ( Array.isArray( obj.material ) ? obj.material : [ obj.material ] ) : [];

		for ( const material of materials ) {

			for ( const key in material ) {

				if ( material[ key ].isTexture ) {

					material[ key ].dispose();

				}

			}

			material.dispose();

		}

	} );

}

async function loadModelToScene( scene, renderer, modelInfo, onProgress ) {

	onProgress( 0 );

	let model;

	try {

		model = await loadModel( modelInfo.url, onProgress );

	} catch ( err ) {

		return { error: 'Failed to load model:' + err.message };

	}

	// update after model load
	// TODO: clean up
	if ( modelInfo.removeEmission ) {

		model.traverse( c => {

			if ( c.material ) {

				c.material.emissiveMap = null;
				c.material.emissiveIntensity = 0;

			}

		} );

	}

	if ( modelInfo.opacityToTransmission ) {

		convertOpacityToTransmission( model, modelInfo.ior || 1.5 );

	}

	model.traverse( c => {

		if ( c.material ) {

			// set the thickness so we render the material as a volumetric object
			c.material.thickness = 1.0;

		}

	} );

	if ( modelInfo.postProcess ) {

		modelInfo.postProcess( model );

	}

	// rotate model after so it doesn't affect the bounding sphere scale
	if ( modelInfo.rotation ) {

		model.rotation.set( ...modelInfo.rotation );

	}

	// center the model
	const box = new Box3();
	box.setFromObject( model );
	model.position
		.addScaledVector( box.min, - 0.5 )
		.addScaledVector( box.max, - 0.5 );

	const sphere = new Sphere();
	box.getBoundingSphere( sphere );

	model.scale.setScalar( 1 / sphere.radius );
	model.position.multiplyScalar( 1 / sphere.radius );
	box.setFromObject( model );
	floorPlane.position.y = box.min.y;

	scene.add( model );

}

async function loadModel( url, onProgress ) {

	// TODO: clean up
	const manager = new LoadingManager();
	if ( /dae$/i.test( url ) ) {

		const complete = new Promise( resolve => manager.onLoad = resolve );
		const res = await new ColladaLoader( manager ).loadAsync( url, progress => {

			if ( progress.total !== 0 && progress.total >= progress.loaded ) {

				onProgress( progress.loaded / progress.total );

			}

		} );
		await complete;

		res.scene.scale.setScalar( 1 );
		res.scene.traverse( c => {

			const { material } = c;
			if ( material && material.isMeshPhongMaterial ) {

				c.material = new MeshStandardMaterial( {

					color: material.color,
					roughness: material.roughness || 0,
					metalness: material.metalness || 0,
					map: material.map || null,

				} );

			}

		} );

		return res.scene;

	} else if ( /(gltf|glb)$/i.test( url ) ) {

		const complete = new Promise( resolve => manager.onLoad = resolve );
		const gltf = await new GLTFLoader( manager ).setMeshoptDecoder( MeshoptDecoder ).loadAsync( url, progress => {

			if ( progress.total !== 0 && progress.total >= progress.loaded ) {

				onProgress( progress.loaded / progress.total );

			}

		} );
		await complete;

		return gltf.scene;

	} else if ( /mpd$/i.test( url ) ) {

		manager.onProgress = ( url, loaded, total ) => {

			onProgress( loaded / total );

		};

		const complete = new Promise( resolve => manager.onLoad = resolve );
		const ldrawLoader = new LDrawLoader( manager );
		ldrawLoader.setConditionalLineMaterial( LDrawConditionalLineMaterial );
		await ldrawLoader.preloadMaterials( 'https://raw.githubusercontent.com/gkjohnson/ldraw-parts-library/master/colors/ldcfgalt.ldr' );
		const result = await ldrawLoader
			.setPartsLibraryPath( 'https://raw.githubusercontent.com/gkjohnson/ldraw-parts-library/master/complete/ldraw/' )
			.loadAsync( url );
		await complete;

		const model = LDrawUtils.mergeObject( result );
		model.rotation.set( Math.PI, 0, 0 );

		const toRemove = [];
		model.traverse( c => {

			if ( c.isLineSegments ) {

				toRemove.push( c );

			}

			if ( c.isMesh ) {

				c.material.roughness *= 0.25;

			}

		} );

		toRemove.forEach( c => {

			c.parent.remove( c );

		} );

		return model;

	}

}

function convertOpacityToTransmission( model, ior ) {

	model.traverse( c => {

		if ( c.material ) {

			const material = c.material;
			if ( material.opacity < 0.65 && material.opacity > 0.2 ) {

				const newMaterial = new MeshPhysicalMaterial();
				for ( const key in material ) {

					if ( key in material ) {

						if ( material[ key ] === null ) {

							continue;

						}

						if ( material[ key ].isTexture ) {

							newMaterial[ key ] = material[ key ];

						} else if ( material[ key ].copy && material[ key ].constructor === newMaterial[ key ].constructor ) {

							newMaterial[ key ].copy( material[ key ] );

						} else if ( ( typeof material[ key ] ) === 'number' ) {

							newMaterial[ key ] = material[ key ];

						}

					}

				}

				newMaterial.opacity = 1.0;
				newMaterial.transmission = 1.0;
				newMaterial.ior = ior;

				const hsl = {};
				newMaterial.color.getHSL( hsl );
				hsl.l = Math.max( hsl.l, 0.35 );
				newMaterial.color.setHSL( hsl.h, hsl.s, hsl.l );

				c.material = newMaterial;

			}

		}

	} );

}
