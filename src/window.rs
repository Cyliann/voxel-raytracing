use std::iter;

use winit::{
    event::{ElementState, KeyboardInput, MouseButton, WindowEvent},
    window::Window,
};

use crate::{camera, raytracing, render};
pub struct State {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub window: Window,
    pub render: render::RenderPipeline,
    pub camera: camera::CameraPipeline,
    pub raytracing: raytracing::RaytracingPipeline,
    pub mouse_pressed: bool,
}

impl State {
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let vert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vert.wgsl").into()),
        });

        let frag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frag.wgsl").into()),
        });

        let camera = camera::CameraPipeline::new(&device);

        let raytracing =
            raytracing::RaytracingPipeline::new(&device, &size, &camera.bind_group_layout);

        let render = render::RenderPipeline::new(
            &device,
            vert_shader,
            frag_shader,
            &config,
            &raytracing.sampler,
            &raytracing.texture,
        );

        Self {
            surface,
            device,
            queue,
            size,
            config,
            window,
            render,
            camera,
            raytracing,
            mouse_pressed: false,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.camera
                .uniform
                .update_proj(&self.camera.camera, new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(unused_variables)]
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => self.camera.controller.process_keyboard(*key, *state),
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                if *state == ElementState::Pressed {
                    self.mouse_pressed = !self.mouse_pressed;
                    if self.mouse_pressed {
                        self.window()
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .unwrap();
                    } else {
                        self.window()
                            .set_cursor_grab(winit::window::CursorGrabMode::None)
                            .unwrap();
                    }
                    self.window().set_cursor_visible(!self.mouse_pressed);
                }
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self, dt: instant::Duration) {
        self.camera
            .controller
            .update_camera(&mut self.camera.camera, dt, &mut self.camera.uniform);
        self.queue.write_buffer(
            &self.camera.buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut ray_tracing_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray tracing pass"),
            });

            ray_tracing_pass.set_pipeline(&self.raytracing.pipeline);
            ray_tracing_pass.set_bind_group(0, &self.raytracing.bind_group, &[]);
            ray_tracing_pass.set_bind_group(1, &self.camera.bind_group, &[]);
            ray_tracing_pass.dispatch_workgroups(self.size.width / 16, self.size.height / 16, 1);
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            // Pipeline
            render_pass.set_pipeline(&self.render.pipeline);
            // Camera
            render_pass.set_bind_group(0, &self.render.bind_group, &[]);
            // Draw
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
