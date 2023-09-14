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

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screen_pos = vec2<i32>(GlobalInvocationID.xy);
    let screen_size = textureDimensions(color_buffer);
    var pixel_color = vec3<f32>(.1, .2, .3);
    let pixel_coord = (vec2<f32>(screen_pos) / vec2<f32>(screen_size)) * 2. - 1.;

    let targetPoint = camera.proj * vec4<f32>(pixel_coord, -1., 1.);
    var origin = camera.view_pos.xyz;
    var direction = (camera.view * vec4<f32>(normalize(targetPoint.xyz / targetPoint.w), 0.)).xyz;

    var ray = Ray(origin, direction);
    var lightDir = vec3<f32>(1.);
    let dda = dda(ray);
    let sphere = intersectSphere(ray, lightDir);
    //if sphere.w > 0. { pixel_color = sphere.xyz; }

    if dda.w == 1. { pixel_color = dda.xyz; }

    textureStore(color_buffer, screen_pos, vec4<f32>(pixel_color, 1.0));
}

fn dda(r: Ray) -> vec4<f32> {
    var direction = normalize(r.direction);
    if direction.x == 0. { direction.x = 0.001; }
    if direction.y == 0. { direction.y = 0.001; }
    if direction.z == 0. { direction.z = 0.001; }

    let raySign = vec3<i32>(sign(direction));
    let rayPositivity = (1 + raySign) / 2;
    let rayInverse = 1. / direction;

    var gridCoords = vec3<i32>(floor(r.origin));
    var withinVoxelCoords = r.origin - vec3<f32>(gridCoords);

    var i = 0;
    while i < 32 {
        if getVoxel(gridCoords) {
            return vec4<f32>(vec3f(abs(gridCoords)) / 100., 1.0);
        }

        let t = (vec3f(rayPositivity) - withinVoxelCoords) * rayInverse;
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
    return vec4<i32>(0.);
}

fn getVoxel(c: vec3<i32>) -> bool {
    //return abs(c.x) == abs(c.y) && abs(c.y) == abs(c.z);
    //return c.x <= 10 && c.x >= 0 && c.y <= 10 && c.y >= 0 && c.z <= 10 && c.z >= 0;
    return c.z == c.y * c.x;
}

fn intersectSphere(ray: Ray, lightDir: vec3<f32>) -> vec4<f32> {
    // (bx^2 + by^2)t^2 + (2(axbx + ayby))t + (ax^2 + ay^2 - r^2) = 0;
    // where
    // a -> ray origin
    // b -> ray direction
    // t -> ray distance
    let radius = 1.5;
    let origin = ray.origin - vec3<f32>(2., 3., 2.);

    let a = dot(ray.direction, ray.direction);
    let b = 2. * dot(origin, ray.direction);
    let c = dot(origin, origin) - radius * radius;

    // Delta
    let delta = b * b - 4. * a * c;

    if delta < 0. {
        return vec4<f32>(0., 0., 0., 0.);
    }

    let closestT = (-b - sqrt(delta)) / (2. * a);
    if closestT <= 0. {return vec4<f32>(0.);}

    let hitPoint = origin + ray.direction * closestT;
    let normal = normalize(hitPoint);
    let d = max(dot(normal, lightDir), 0.);

    var sphereColor = vec3<f32>(1., 1., 1.);
    sphereColor *= d;

    return vec4<f32>(sphereColor, closestT);
}

