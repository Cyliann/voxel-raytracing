use wgpu::BindGroupLayout;
use winit::dpi::PhysicalSize;

pub struct RaytracingPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
    pub sampler: wgpu::Sampler,
    pub texture: wgpu::TextureView,
}

impl RaytracingPipeline {
    pub fn new(
        device: &wgpu::Device,
        size: &PhysicalSize<u32>,
        camera_bind_group_layout: &BindGroupLayout,
    ) -> RaytracingPipeline {
        let raytrace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray tracing shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ray-tracing.wgsl").into()),
        });

        let color_buffer = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("Color buffer texture"),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            view_formats: &[],
        });

        let color_buffer_view = color_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        let color_buffer_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
            label: Some("color buffer bind group layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray tracing bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_buffer_view),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray tracing Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray tracing pipeline"),
            layout: Some(&pipeline_layout),
            module: &raytrace_shader,
            entry_point: "main",
        });

        RaytracingPipeline {
            pipeline,
            bind_group,
            sampler: color_buffer_sampler,
            texture: color_buffer_view,
        }
    }
}
