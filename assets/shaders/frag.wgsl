@group(0) @binding(0) var screen_sampler: sampler;
@group(0) @binding(1) var color_buffer: texture_2d<f32>;

@fragment
fn fs_main(@location(0) tex_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let coord = tex_coord / 2. + 0.5; // normalize between 0...1
    return textureSample(color_buffer, screen_sampler, coord);
}
