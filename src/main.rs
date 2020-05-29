#[macro_use]
extern crate glium;
extern crate rand;
use glium::Surface;
use rand::Rng;

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
uniform vec3 cam_position;

void main() {
    mat4 matrix;
    matrix[0] = vec4(1.0, 0.0, 0.0, 0.0);
    matrix[1] = vec4(0.0, 1.0, 0.0, 0.0);
    matrix[2] = vec4(0.0, 0.0, 1000.1/999.9, 1.0);
    matrix[3] = vec4(0.0, 0.0, -200.0/999.9, 0.0);
    gl_Position = matrix * vec4(position - cam_position, 1.0);
}
"#;

const RAYTRACING_SHADER_SRC: &str = r#"
#version 140

out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
"#;

fn main() {
    use glium::glutin;

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(1241);
    let mut data = [[[([0u8, 0u8, 0u8], false, false); 16]; 32]; 16];
    let mut vertices: Vec<Vertex> = Vec::new();

    for z in 0..16 {
        for y in 0..32 {
            for x in 0..16 {
                data[x][y][z] = (
                    [
                        (8 * x + rng.gen_range(0, 8)) as u8,
                        (4 * y + rng.gen_range(0, 4)) as u8,
                        (8 * z + rng.gen_range(0, 8)) as u8,
                    ],
                    rng.gen_bool(0.5),
                    rng.gen_bool(0.5),
                );
            }
        }
    }

    for _ in 0..2 {
        for z in 0..16 {
            for y in 0..32 {
                for x in 0..16 {
                    let (_, ax, az) = data[x][y][z];
                    if ax && z < 15 {
                        data[x][y][z].0[0] = data[x][y][z + 1].0[0];
                    }
                    if az && x < 15 {
                        data[x][y][z].0[2] = data[x + 1][y][z].0[2];
                    }
                }
            }
        }
    }
    for z in 0..16 {
        for y in 0..32 {
            for x in 0..16 {
                let ([px, py, pz], _, _) = data[x][y][z];
                if z < 15 {
                    let ([qx, qy, qz], _, _) = data[x][y][z + 1];
                    if px == qx {
                        vertices.push(Vertex {
                            position: [px as f32, py as f32, pz as f32],
                        });
                        vertices.push(Vertex {
                            position: [qx as f32, qy as f32, qz as f32],
                        });
                    }
                }
                if x < 15 {
                    let ([qx, qy, qz], _, _) = data[x + 1][y][z];
                    if pz == qz {
                        vertices.push(Vertex {
                            position: [px as f32, py as f32, pz as f32],
                        });
                        vertices.push(Vertex {
                            position: [qx as f32, qy as f32, qz as f32],
                        });
                    }
                }
            }
        }
    }

    let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LinesList);
    let program =
        glium::Program::from_source(&display, VERTEX_SHADER_SRC, RAYTRACING_SHADER_SRC, None)
            .unwrap();
    let mut cam_position: [f32; 3] = [64.0, 64.0, -10.0];
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
                    cam_position: cam_position
                },
                &Default::default(),
            )
            .unwrap();

        target.finish().unwrap();
    });
}
