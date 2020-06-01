#[macro_use]
extern crate glium;
extern crate nalgebra as na;
use glium::Surface;

mod octree;
use octree::build_octree;

const FPS: u64 = 60;
const DELTA: f32 = 1.0 / FPS as f32;
const SPEED: f32 = 8.0;

#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
}
implement_vertex!(Vertex, position);

fn main() {
    use glium::glutin;

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let vertices = vec![
        Vertex {
            position: [0.0, 0.0, 0.0],
        },
        Vertex {
            position: [128.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.0, 128.0, 0.0],
        },
        Vertex {
            position: [128.0, 128.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.0, 128.0],
        },
        Vertex {
            position: [128.0, 0.0, 128.0],
        },
        Vertex {
            position: [0.0, 128.0, 128.0],
        },
        Vertex {
            position: [128.0, 128.0, 128.0],
        },
    ];

    let indices: Vec<u32> = vec![
        0, 1, 2, 1, 3, 2, // FRONT
        0, 4, 1, 4, 5, 1, // BOTTOM
        0, 2, 4, 2, 6, 4, // LEFT
        4, 5, 6, 5, 7, 6, // BACK
        2, 6, 3, 6, 7, 3, // TOP
        1, 3, 5, 3, 7, 5, // RIGHT
    ];

    let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();
    let index_buffer = glium::index::IndexBuffer::new(
        &display,
        glium::index::PrimitiveType::TrianglesList,
        &indices,
    )
    .unwrap();

    let program = glium::Program::from_source(
        &display,
        &std::fs::read_to_string("shaders/default.vert").unwrap(),
        &std::fs::read_to_string("shaders/raytrace.frag").unwrap(),
        None,
    )
    .unwrap();

    let mut octree_buffer = glium::buffer::Buffer::<[u32]>::empty_unsized(
        &display,
        glium::buffer::BufferType::ShaderStorageBuffer,
        128 * 128 * 128 * 4,
        glium::buffer::BufferMode::Immutable,
    )
    .unwrap();

    let mut voxel_buffer = glium::buffer::Buffer::<[(u32, u32)]>::empty_unsized(
        &display,
        glium::buffer::BufferType::ShaderStorageBuffer,
        16 * 16 * 32 * 16 * 8,
        glium::buffer::BufferMode::Persistent,
    )
    .unwrap();

    #[derive(Clone, Copy)]
    struct VoxelCounter {
        voxel_counter: u32,
    };
    implement_uniform_block!(VoxelCounter, voxel_counter);

    let mut voxel_counter = glium::buffer::Buffer::new(
        &display,
        &VoxelCounter {
            voxel_counter: 0u32,
        },
        glium::buffer::BufferType::ShaderStorageBuffer,
        glium::buffer::BufferMode::Persistent,
    )
    .unwrap();

    let gen_paths_shader = glium::program::ComputeShader::from_source(
        &display,
        &std::fs::read_to_string("shaders/gen_paths.compute").unwrap(),
    )
    .unwrap();

    gen_paths_shader.execute(
        uniform! {
            voxel_buffer: &voxel_buffer,
            voxel_counter_buffer: &voxel_counter,
        },
        1,
        1,
        1,
    );

    let count = voxel_counter.map_read().voxel_counter;
    println!("{}", count);

    build_octree(
        octree_buffer.map(),
        voxel_buffer.map_read(),
        count as usize,
        7,
    );

    let mut cam_position = na::Vector3::new(0.0, 0.0, 0.0);
    let mut cam_speed = na::Vector3::new(0.0, 0.0, 0.0);

    let (width, height) = display.get_framebuffer_dimensions();
    let projection = na::Perspective3::new(
        width as f32 / height as f32,
        std::f32::consts::PI / 3.0,
        0.01,
        1000.0,
    )
    .to_homogeneous();

    event_loop.run(move |ev, _, control_flow| {
        match ev {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                }
                glutin::event::WindowEvent::KeyboardInput { input, .. } => {
                    use glutin::event::VirtualKeyCode;
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => {
                            cam_speed[2] = if input.state == glutin::event::ElementState::Pressed {
                                SPEED
                            } else {
                                0.0
                            };
                        }
                        Some(VirtualKeyCode::S) => {
                            cam_speed[2] = if input.state == glutin::event::ElementState::Pressed {
                                -SPEED
                            } else {
                                0.0
                            };
                        }
                        Some(VirtualKeyCode::A) => {
                            cam_speed[0] = if input.state == glutin::event::ElementState::Pressed {
                                -SPEED
                            } else {
                                0.0
                            };
                        }
                        Some(VirtualKeyCode::D) => {
                            cam_speed[0] = if input.state == glutin::event::ElementState::Pressed {
                                SPEED
                            } else {
                                0.0
                            };
                        }
                        Some(VirtualKeyCode::Space) => {
                            cam_speed[1] = if input.state == glutin::event::ElementState::Pressed {
                                SPEED
                            } else {
                                0.0
                            };
                        }
                        Some(VirtualKeyCode::LShift) => {
                            cam_speed[1] = if input.state == glutin::event::ElementState::Pressed {
                                -SPEED
                            } else {
                                0.0
                            };
                        }
                        _ => (),
                    }
                }
                _ => (),
            },
            _ => (),
        };

        cam_position += DELTA * cam_speed;

        let view = na::Translation3::from(-cam_position);
        let mvp = projection
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(1.0, 1.0, -1.0))
            * view.to_homogeneous();

        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        target
            .draw(
                &vertex_buffer,
                &index_buffer,
                &program,
                &uniform! {
                    cam_position: *cam_position.as_ref(),
                    octree_buffer: &octree_buffer,
                    MVP: *mvp.as_ref(),
                },
                &Default::default(),
            )
            .unwrap();

        target.finish().unwrap();
    });
}
