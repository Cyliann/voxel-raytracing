use instant::Duration;
use nalgebra::*;
use winit::event::*;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: nalgebra::Matrix4<f32> = nalgebra::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 1.0,
    0.0, 0.0, 0.0, 1.0,
);

#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    pub direction: Vector3<f32>,
    pub fov: f32,
    pub near_clip: f32,
    pub far_clip: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, F: Into<f32>, N: Into<f32>, M: Into<f32>>(
        position: V,
        fov: F,
        near_clip: N,
        far_clip: M,
    ) -> Self {
        Self {
            position: position.into(),
            direction: Vector3::new(0., 0., 1.),
            fov: fov.into(),
            near_clip: near_clip.into(),
            far_clip: far_clip.into(),
            yaw: 0.,
            pitch: 0.,
        }
    }

    pub fn calc_view(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_lh(
            &self.position,
            &(self.position + self.direction),
            &Vector3::new(0., 1., 0.),
        );

        Matrix4::try_inverse(view).expect("Could not inverse view matrix") * OPENGL_TO_WGPU_MATRIX
    }

    pub fn calc_proj(&self, width: u32, height: u32) -> Matrix4<f32> {
        let aspect = width as f32 / height as f32;
        let proj = Matrix4::new_perspective(aspect, self.fov, self.near_clip, self.far_clip);

        return Matrix4::try_inverse(proj).expect("Could not inverse projection matrix");
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: [f32; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view: nalgebra::Matrix4::identity().into(),
            proj: nalgebra::Matrix4::identity().into(),
        }
    }

    fn update_view(&mut self, camera: &Camera) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view = camera.calc_view().into();
    }

    pub fn update_proj(&mut self, camera: &Camera, width: u32, height: u32) {
        self.proj = camera.calc_proj(width, height).into();
    }
}

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    speed: f32,
    sensitivity: f32,
    last_mouse_pos: (f64, f64),
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            speed,
            sensitivity,
            last_mouse_pos: (0., 0.),
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_pos: (f64, f64)) {
        self.rotate_horizontal = (mouse_pos.0 - self.last_mouse_pos.0) as f32;
        self.rotate_vertical = (mouse_pos.1 - self.last_mouse_pos.1) as f32;
    }

    pub fn update_camera(
        &mut self,
        camera: &mut Camera,
        dt: Duration,
        camera_unifrom: &mut CameraUniform,
    ) {
        let dt = dt.as_secs_f32();

        let up = Vector3::new(0., 1., 0.);
        let forward = Vector3::new(camera.direction.x, 0., camera.direction.z);
        let right = Matrix::cross(&up, &forward);

        // Move forward/backward and left/right
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera.yaw = self.rotate_horizontal * self.sensitivity * dt;
        camera.pitch = self.rotate_vertical * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera.pitch < -90. {
            camera.pitch = -90.;
        } else if camera.pitch > 180. {
            camera.pitch = 180.;
        }

        camera.direction =
            Rotation::from_axis_angle(&Unit::new_normalize(right), camera.pitch) * camera.direction;
        camera.direction =
            Rotation::from_axis_angle(&Unit::new_normalize(up), camera.yaw) * camera.direction;

        camera_unifrom.update_view(camera);
    }
}

pub struct CameraPipeline {
    pub camera: Camera,
    pub controller: CameraController,
    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CameraPipeline {
    pub fn new(device: &wgpu::Device) -> CameraPipeline {
        let camera = Camera::new(Vector3::new(0.0, 2.0, -12.0), 45., 1., 100.);
        let controller = CameraController::new(10.0, 1.0);

        let uniform = CameraUniform::new();

        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        return CameraPipeline {
            camera,
            controller,
            uniform,
            buffer,
            bind_group,
            bind_group_layout,
        };
    }
}
