#[macro_use]
extern crate glium;
extern crate rand;
use glium::Surface;

const FPS: u64 = 60;
const DELTA: f32 = 1.0 / 60 as f32;
const FRAME_NANOS: u64 = 1_000_000_000 / FPS;
const SPEED: f32 = 5.0;

#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
}
implement_vertex!(Vertex, position);

const VERTEX_SHADER_SRC: &str = r#"
#version 140

in vec3 position;
out vec3 world_position;

void main() {
    world_position = position;
    gl_Position = vec4(position, 1.0);
}
"#;

const RAYTRACING_SHADER_SRC: &str = r#"
#version 430

out vec4 color;
in vec3 world_position;

const vec3 SVO_SIZE = vec3(1.0, 1.0, 1.0);
const vec3 SVO_POSITION = vec3(-0.75, -0.75, 1.0);
const uint MAX_ITERATIONS = 100;
const uint MAX_DEPTH = 1;
const float EPS = 0.0001;

uniform vec3 cam_position;

layout(std430, binding = 0) buffer octree_buffer {
    uint octree[];
};

void main() {
    vec3 dir = normalize(world_position - cam_position);
    vec3 dt = 1.0 / dir;
    float t = 0.0;
    vec3 t1 = (SVO_POSITION - cam_position) * dt;
    vec3 t2 = ((SVO_POSITION + SVO_SIZE) - cam_position) * dt;
    vec3 t_min0 = min(t1, t2);
    vec3 t_max0 = max(t1, t2);
    float t_min = max(t_min0.x, max(t_min0.y, t_min0.z));
    float t_max = min(t_max0.x, min(t_max0.y, t_max0.z));
    if (t_max < t_min || t_max < 0.0) {
        color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    t = max(0.0, t_min);

    for (uint i = 0; i < MAX_ITERATIONS; i++) {
        if (t >= t_max) {
            color = vec4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        vec3 pos = t * dir + cam_position;
        vec3 cell_min = SVO_POSITION;
        vec3 cell_max = SVO_POSITION + SVO_SIZE;
        uint index = 0;
        for (uint j = 0; j < MAX_DEPTH; j++) {
            vec3 cell_mid = (cell_min + cell_max) / 2;
            uvec3 child = uvec3(step(cell_mid, pos));
            uint child_index = child.x + 2 * child.y + 4 * child.z;
            cell_min = child * cell_mid + (1 - child) * cell_min;
            cell_max = child * cell_max + (1 - child) * cell_mid;
            uint child_mask = octree[index];
            if (((child_mask >> child_index) & 1u) == 0) {
                index = 0;
                break;
            } else {
                index = octree[index + bitCount(child_mask << (31 - child_index))];
            }
        }
        if (index != 0) {
            uint data = octree[index];
            color = vec4(
                bitfieldExtract(data, 16, 8) / 255.0,
                bitfieldExtract(data, 8, 8) / 255.0,
                bitfieldExtract(data, 0, 8) / 255.0,
                1.0
            );
            return;
        }
        t_max0 = max((cell_max - cam_position) * dt, (cell_min - cam_position) * dt);
        t = min(t_max0.x, min(t_max0.y, t_max0.z)) + EPS;
    }

    color = vec4(1.0, 0.0, 0.0, 1.0);
}
"#;

fn main() {
    use glium::glutin;

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let vertices = vec![
        Vertex {
            position: [-1.0, -1.0, 1.0],
        },
        Vertex {
            position: [-1.0, 1.0, 1.0],
        },
        Vertex {
            position: [1.0, -1.0, 1.0],
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
    ];

    let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip);
    let program =
        glium::Program::from_source(&display, VERTEX_SHADER_SRC, RAYTRACING_SHADER_SRC, None)
            .unwrap();

    let mut octree_buffer = glium::buffer::Buffer::<[u32]>::empty_unsized(
        &display,
        glium::buffer::BufferType::ShaderStorageBuffer,
        5 * 4,
        glium::buffer::BufferMode::Persistent,
    )
    .unwrap();

    octree_buffer.map()[0] = 0b00010001;
    octree_buffer.map()[1] = 3;
    octree_buffer.map()[2] = 4;
    octree_buffer.map()[3] = 0xffff00;
    octree_buffer.map()[4] = 0x00ff00;

    let mut cam_position: [f32; 3] = [0.0, 0.0, 0.0];
    let mut cam_speed: [f32; 3] = [0.0, 0.0, 0.0];

    event_loop.run(move |ev, _, control_flow| {
        let next_frame = std::time::Instant::now() + std::time::Duration::from_nanos(FRAME_NANOS);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame);
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

        cam_position[0] += DELTA * cam_speed[0];
        cam_position[1] += DELTA * cam_speed[1];
        cam_position[2] += DELTA * cam_speed[2];

        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        target
            .draw(
                &vertex_buffer,
                &indices,
                &program,
                &uniform! {
                    cam_position: cam_position,
                    octree_buffer: &octree_buffer
                },
                &Default::default(),
            )
            .unwrap();

        target.finish().unwrap();
    });
}
