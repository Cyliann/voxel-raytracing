@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct CameraUniform {
    view_pos: vec4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>
};

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screen_pos = vec2<i32>(GlobalInvocationID.xy);
    let screen_size = textureDimensions(color_buffer);
    var pixel_color = vec3<f32>(.1, .2, .3);
    let pixel_coord = (vec2<f32>(screen_pos) / vec2<f32>(screen_size)) * 2. - 1.;

    let targetPoint = camera.proj * vec4<f32>(pixel_coord, -1., 1.);
    var origin = camera.view_pos.xyz;
    var direction = (camera.view * vec4<f32>(normalize(targetPoint.xyz / targetPoint.w), 0.)).xyz;

    var ray = Ray(origin, direction);

    let pixel = raytrace(ray);
    if pixel.w == 1. { pixel_color = pixel.rgb; }

    textureStore(color_buffer, screen_pos, vec4<f32>(pixel_color, 1.0));
}

fn raytrace(ray: Ray) -> vec4<f32> {
    var coord = vec4<f32>(ray.origin, 0.0);
    var scale = 64;

    for (var i = 0; i < 3; i++) {
        coord = dda(Ray(coord.xyz, ray.direction), scale);
        scale /= 8;
    }
    return vec4<f32>(vec3<f32>(coord.xyz) / 100., coord.w);
}

fn dda(r: Ray, scale: i32) -> vec4<f32> {
    var direction = normalize(r.direction);
    if direction.x == 0. { direction.x = 0.001; }
    if direction.y == 0. { direction.y = 0.001; }
    if direction.z == 0. { direction.z = 0.001; }

    let raySign = vec3<i32>(sign(direction));
    let rayPositivity = (1 + raySign) / 2;
    let rayInverse = 1. / direction;

    var gridCoords = vec3<i32>(floor(r.origin / f32(scale)));
    var withinVoxelCoords = r.origin / f32(scale) - vec3<f32>(gridCoords);
    let entryCoords = (gridCoords / scale) * scale; // get beginning of the chunk

    var i = 0;
    while inChunk(gridCoords, entryCoords, scale) {
        let t = (vec3f(rayPositivity) - withinVoxelCoords) * rayInverse;
        if getVoxel(gridCoords, scale) {
            return vec4<f32>((vec3<f32>(gridCoords) + withinVoxelCoords) * f32(scale), 1.0);
        }

        var minIdx: i32;
        if t.x < t.y {
            if t.x < t.z {
                minIdx = 0;
            } else { minIdx = 2;}
        } else {
            if t.y < t.z {
                minIdx = 1;
            } else {
                minIdx = 2;
            }
        }

        gridCoords[minIdx] += raySign[minIdx];
        withinVoxelCoords += direction * t[minIdx];
        withinVoxelCoords[minIdx] = 1. - f32(rayPositivity[minIdx]);
        i++;
    }
    return vec4<f32>((vec3<f32>(gridCoords) + withinVoxelCoords) * f32(scale), 0.0);
}

fn inChunk(coords: vec3<i32>, reference: vec3<i32>, scale: i32) -> bool {
    let diff = abs(coords - reference);

    if diff.x > scale || diff.y > scale || diff.z > scale {return false; }
    return true;
}

fn getVoxel(c: vec3<i32>, scale: i32) -> bool {
    //let s = 50 / scale;
    //let c = c - s * vec3<i32>(round(vec3<f32>(c) / f32(s)));
    //return df_sphere(c, scale) <= 0.;
    return f32(c.y) < sin(f32(c.x) / 5.) * sin(f32(c.z) / 5.) * 5.;
}

fn df_sphere(c: vec3<i32>, scale: i32) -> f32 {
    let p = vec3<f32>(0.);
    let r = 8. / f32(scale);
    return distance(vec3<f32>(c), p) - r;
}
