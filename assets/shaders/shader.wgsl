struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coord: vec2<f32>,
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct CameraUniform {
    view_pos: vec4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {

    var vertices = array<vec2<f32>, 3>(
        vec2<f32>(-1., 3.),
        vec2<f32>(-1., -1.),
        vec2<f32>(3., -1.),
    );

    var out: VertexOutput;
    out.coord = vertices[vertex_index];
    out.position = vec4<f32>(out.coord, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let targetPoint = camera.proj * vec4<f32>(in.coord, -1., 1.);
    var origin = camera.view_pos.xyz;
    var direction = (camera.view * vec4<f32>(normalize(targetPoint.xyz / targetPoint.w), 0.)).xyz;

    var ray = Ray(origin, direction);
    let lightDir = normalize(vec3<f32>(1., 2., -0.7));

    //let box = intersectBox(ray, lightDir);
    //let sphere = intersectSphere(ray, lightDir);
    //let plane = intersectPlane(ray);

    //var closest = vec4<f32>(.1, .2, .3, 2137.);

    //if plane.w != 0. { closest = plane; }

    //if box.w != 0. && box.w < closest.w { closest = box; }

    //if sphere.w != 0. && sphere.w < closest.w { closest = sphere; }

    //return vec4<f32>(closest.xyz, 0.0);
    let dda = dda(ray);
    if dda.w == 1.0 { return dda(ray); }
    let plane = intersectPlane(ray);
    if plane.w != 0. { return vec4f(plane.xyz, 1.0); }
    return vec4<f32>(.1, .2, .3, 1.);
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

fn intersectBox(r: Ray, lightDir: vec3<f32>) -> vec4<f32> {
    let a = vec3<f32>(-1.);
    let b = vec3<f32>(2.);

    let tx1 = (a.x - r.origin.x) / r.direction.x;
    let tx2 = (b.x - r.origin.x) / r.direction.x;

    var tmin = min(tx1, tx2);
    var tmax = max(tx1, tx2);

    let ty1 = (a.y - r.origin.y) / r.direction.y;
    let ty2 = (b.y - r.origin.y) / r.direction.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    let tz1 = (a.z - r.origin.z) / r.direction.z;
    let tz2 = (b.z - r.origin.z) / r.direction.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    if tmax >= tmin && tmin > 0. {
        let hitPoint = r.origin + r.direction * tmin;

        let c = (a + b) * 0.5;
        let p = hitPoint - c;
        let d = (a - b) * 0.5;

        let bias = 1.00001;
        let normal = normalize(vec3<f32>(vec3<i32>(p / abs(d) * bias)));

        let mul = max(dot(normal, lightDir), 0.);

        var color = vec3<f32>(1.);
        color *= mul;

        return vec4<f32>(color, tmin);
    }

    return vec4<f32>(0.);
}

fn intersectPlane(r: Ray) -> vec4<f32> {
    let n = vec3<f32>(0., 1., 0.);
    let p0 = vec3<f32>(0., 0., 0.);

    let denom = dot(n, r.direction);

    if abs(denom) > 0.00001 {
        let t = dot(p0 + r.origin, n) / denom;
        if t <= 0. {
            return vec4<f32>(0., 0.6, 0.1, -t);
        };
    }

    return vec4<f32>(0.);
}

fn dda(r: Ray) -> vec4<f32> {

    var mapPos = vec3i(floor(r.origin + 0.));
    var deltaDist = 1. / abs(r.direction);
    var rayStep = vec3i(sign(r.direction));
    var sideDist = vec3f(sign(r.direction) * (vec3f(mapPos) - r.origin) + (sign(r.direction) * 0.5) + 0.5) * deltaDist;

    var mask: vec3<bool>;
    var hit = false;

    for (var i: i32 = 0; i < 64; i++) {
        if getVoxel(mapPos) {
            hit = true;
            break;
        }

        if sideDist.x < sideDist.y {
            if sideDist.x < sideDist.z {
                sideDist.x += deltaDist.x;
                mapPos.x += rayStep.x;
                mask = vec3<bool>(true, false, false);
            } else {
                sideDist.z += deltaDist.z;
                mapPos.z += rayStep.z;
                mask = vec3<bool>(false, false, true);
            }
        } else {
            if sideDist.y < sideDist.z {
                sideDist.y += deltaDist.y;
                mapPos.y += rayStep.y;
                mask = vec3<bool>(false, true, false);
            } else {
                sideDist.z += deltaDist.z;
                mapPos.z += rayStep.z;
                mask = vec3<bool>(false, false, true);
            }
        }
    }

    if hit {
        if mask.x {return vec4f(0.5, 0.5, 0.5, 1.0);}
        if mask.y {return vec4f(0.7, 0.7, 0.7, 1.0);}
        if mask.z {return vec4f(1.0, 1.0, 1.0, 1.0);}
    }
    return vec4<f32>(0., 0., 0., 0.0);
}

fn getVoxel(c: vec3<i32>) -> bool {
    let p = vec3f(c) + vec3f(0.5);
    let d = min(max(-sdSphere(p, 7.5), sdBox(p, vec3(6.0))), -sdSphere(p, 25.0));
    return d < 0.0;
    //return abs(c.x) == abs(c.y) && abs(c.y) == abs(c.z);
    //return c.x <= 10 && c.x >= 0 && c.y <= 10 && c.y >= 0 && c.z <= 10 && c.z >= 0;
    //return c.z == c.y * c.x;
}

fn sdSphere(p: vec3f, d: f32) -> f32 { return length(p) - d; }

fn sdBox(p: vec3f, b: vec3f) -> f32 {
    let d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3f(0.0)));
}
