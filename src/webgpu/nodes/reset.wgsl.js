import { wgslFn } from 'three/tsl';

export const resetFn = wgslFn( /* wgsl */ `

	fn reset(
		resultBuffer: ptr<storage, array<vec4f>, read_write>,
		sample_count_buffer: ptr<storage, array<u32>, read_write>,
		dimensions: vec2u,

		globalId: vec2u,
	) -> void {

		let offset = globalId.x + globalId.y * dimensions.x;
		sample_count_buffer[offset] = 0;
		resultBuffer[offset] = vec4f(0.0);

	}

` );

export default resetFn;
