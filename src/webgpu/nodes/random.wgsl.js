import { wgsl, wgslFn } from 'three/tsl';

export const pcgStateStruct = wgsl( /* wgsl */`
	struct PcgState {
		s0: vec4u,
		s1: vec4u,
		pixel: vec2i,
	};
` );

export const pcgInit = wgslFn( /* wgsl */`
	fn pcg_initialize(state: ptr<function, PcgState>, p: vec2u, frame: u32) -> void {
		state.pixel = vec2i( p );

		//white noise seed
		state.s0 = vec4u(p, frame, u32(p.x) + u32(p.y));

		//blue noise seed
		state.s1 = vec4u(frame, frame*15843, frame*31 + 4566, frame*2345 + 58585);
	}
`, [ pcgStateStruct ] );

export const pcg4d = wgslFn( /* wgsl */ `
	fn pcg4d(v: ptr<function, vec4u>) -> void {
		*v = *v * 1664525u + 1013904223u;
		v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
		*v = *v ^ (*v >> vec4u(16u));
		v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	}
` );

// TODO: test if abs there is necessary
export const pcgRand3 = wgslFn( /*wgsl*/`
	fn pcgRand3(state: ptr<function, PcgState>) -> vec3f {
		pcg4d(&state.s0);
		return abs( vec3f(state.s0.xyz) / f32(0xffffffffu) );
	}
`, [ pcg4d, pcgStateStruct ] );

export const pcgRand2 = wgslFn( /*wgsl*/`
	fn pcgRand2(state: ptr<function, PcgState>) -> vec2f {
		pcg4d(&state.s0);
		return abs( vec2f(state.s0.xy) / f32(0xffffffffu) );
	}
`, [ pcg4d, pcgStateStruct ] );
