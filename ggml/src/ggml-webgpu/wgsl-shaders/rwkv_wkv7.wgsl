struct Params {
    n_heads: u32,
    n_tokens: u32,
    n_seq_tokens: u32,
    offset_r: u32,
    offset_w: u32,
    offset_k: u32,
    offset_v: u32,
    offset_a: u32,
    offset_b: u32,
    offset_state: u32,
    offset_dst: u32,
};

// Graph dispatches share a compute pass, so their storage bindings use the same access mode.
@group(0) @binding(0) var<storage, read_write> r: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> a: array<f32>;
@group(0) @binding(5) var<storage, read_write> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> state_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const HEAD_VECS = u32(HEAD_SIZE) / 4u;
var<workgroup> shared_r: array<vec4<f32>, HEAD_VECS>;
var<workgroup> shared_w: array<vec4<f32>, HEAD_VECS>;
var<workgroup> shared_k: array<vec4<f32>, HEAD_VECS>;
var<workgroup> shared_a: array<vec4<f32>, HEAD_VECS>;
var<workgroup> shared_b: array<vec4<f32>, HEAD_VECS>;

@compute @workgroup_size(HEAD_SIZE)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let token_stride = params.n_heads * u32(HEAD_SIZE);
    let head_offset = wg.x * u32(HEAD_SIZE);
    // Each lane owns one state row for this sequence and head.
    let row = (wg.y * token_stride + head_offset + lane) * u32(HEAD_SIZE);
    var state: array<vec4<f32>, HEAD_VECS>;
    for (var i = 0u; i < HEAD_VECS; i++) {
        let j = params.offset_state + row + i * 4u;
        state[i] = vec4<f32>(state_in[j], state_in[j + 1u], state_in[j + 2u], state_in[j + 3u]);
    }

    for (var t = 0u; t < params.n_seq_tokens; t++) {
        let token = (wg.y * params.n_seq_tokens + t) * token_stride + head_offset;
        if (lane < HEAD_VECS) {
            let j = token + lane * 4u;
            let ri = params.offset_r + j;
            let wi = params.offset_w + j;
            let ki = params.offset_k + j;
            let ai = params.offset_a + j;
            let bi = params.offset_b + j;
            shared_r[lane] = vec4<f32>(r[ri], r[ri + 1u], r[ri + 2u], r[ri + 3u]);
            shared_w[lane] = vec4<f32>(w[wi], w[wi + 1u], w[wi + 2u], w[wi + 3u]);
            shared_k[lane] = vec4<f32>(k[ki], k[ki + 1u], k[ki + 2u], k[ki + 3u]);
            shared_a[lane] = vec4<f32>(a[ai], a[ai + 1u], a[ai + 2u], a[ai + 3u]);
            shared_b[lane] = vec4<f32>(b[bi], b[bi + 1u], b[bi + 2u], b[bi + 3u]);
        }
        workgroupBarrier();

        var sa = 0.0;
        for (var i = 0u; i < HEAD_VECS; i++) {
            sa += dot(state[i], shared_a[i]);
        }

        let value = v[params.offset_v + token + lane];
        var out = 0.0;
        for (var i = 0u; i < HEAD_VECS; i++) {
            state[i] = state[i] * shared_w[i] + value * shared_k[i] + sa * shared_b[i];
            out += dot(state[i], shared_r[i]);
        }
        dst[params.offset_dst + token + lane] = out;
        workgroupBarrier();
    }

    // The final state follows all token outputs.
    let state_dst = params.offset_dst + params.n_tokens * token_stride + row;
    for (var i = 0u; i < HEAD_VECS; i++) {
        let j = state_dst + i * 4u;
        let value = state[i];
        dst[j] = value.x;
        dst[j + 1u] = value.y;
        dst[j + 2u] = value.z;
        dst[j + 3u] = value.w;
    }
}
