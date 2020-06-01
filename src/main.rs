#[macro_use]
extern crate glium;
extern crate nalgebra as na;
use glium::Surface;

const VERTEX_SHADER_SRC: &str = r#"
#version 140

in vec3 position;
out vec3 world_position;

uniform mat4 MVP;

void main() {
    world_position = position;
    gl_Position = MVP * vec4(position, 1.0);
}
"#;

const RAYTRACING_SHADER_SRC: &str = r#"
#version 430

out vec4 color;
in vec3 world_position;

const vec3 SVO_SIZE = vec3(128.0, 128.0, 128.0);
const vec3 SVO_POSITION = vec3(0.0, 0.0, 0.0);
const uint MAX_DEPTH = 7;
const uint MAX_ITERATIONS = MAX_DEPTH * 100;
const float EPS = 0.0001;

uniform vec3 cam_position;

layout(std430, binding = 0) buffer octree_buffer {
    uint octree[];
};

vec4 raytrace(vec3 pos, vec3 dir) {
    vec3 not_zero = step(0.0001, abs(dir));
    dir = mix(vec3(0.0001), dir, not_zero);
    vec3 dt = 1.0 / dir;
    float t = 0.0;
    vec3 t1 = (SVO_POSITION - pos) * dt;
    vec3 t2 = ((SVO_POSITION + SVO_SIZE) - pos) * dt;
    vec3 t_min0 = min(t1, t2);
    vec3 t_max0 = max(t1, t2);
    float t_min = max(t_min0.x, max(t_min0.y, t_min0.z));
    float t_max = min(t_max0.x, min(t_max0.y, t_max0.z));
    if (t_max < t_min || t_max < 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    t = max(0.0, t_min);

    vec3 pos1 = t * dir + pos;
    vec3 cell_min = SVO_POSITION;
    vec3 cell_max = SVO_POSITION + SVO_SIZE;
    uint index = 0;
    uint depth = 0;
    for (uint i = 0; i < MAX_ITERATIONS; i++) {
        vec3 cell_mid = (cell_min + cell_max) / 2;
        vec3 child = step(cell_mid, pos1);
        uint child_index = uint(dot(child, vec3(1.0, 2.0, 4.0)));
        cell_min = child * cell_mid + (1 - child) * cell_min;
        cell_max = child * cell_max + (1 - child) * cell_mid;
        uint child_mask = octree[index];
        child_mask = child_mask << (31 - child_index);
        if ((child_mask & (1u << 31)) == 0) {
            index = 0;
        } else {
            index = octree[index + bitCount(child_mask)];
        }
        depth++;
        if (index == 0) {
            t_max0 = max((cell_max - pos) * dt, (cell_min - pos) * dt);
            t = min(t_max0.x, min(t_max0.y, t_max0.z)) + EPS;
            if (t >= t_max) {
                return vec4(0.0, 0.0, 0.0, 0.0);
            }
            pos1 = t * dir + pos;
            cell_min = SVO_POSITION;
            cell_max = SVO_POSITION + SVO_SIZE;
            depth = 0;
        } else if (depth == MAX_DEPTH) {
            return vec4(
                bitfieldExtract(index, 16, 8),
                bitfieldExtract(index, 8, 8),
                bitfieldExtract(index, 0, 8),
                255.0
            ) / 255.0;
        }
    }

    return vec4(1.0, 0.0, 0.0, 1.0);
}

void main() {
    vec3 dir = normalize(world_position - cam_position);
    color = raytrace(cam_position, dir);
}
"#;

const GEN_PATHS_SRC: &str = r#"
#version 430

const uvec3 CELL_SIZE = uvec3(8, 4, 8);
const uvec3 OUTPUT_SIZE = uvec3(128, 128, 128);
const uvec3 OUTPUT_SIZE_GROUP = uvec3(8, 8, 8);
const uvec3 NUM_GROUPS = OUTPUT_SIZE / CELL_SIZE;

layout (local_size_x = 10, local_size_y = 10, local_size_z = 10) in;

shared uint path_offset[10][10][10];

layout (std430, binding = 0) buffer voxel_buffer {
    uvec2 voxels[];
};

// Can't use atomic counter because glium doesn't support it
// Can't put counter in voxel_buffer because glium doesn't support it
layout (binding = 1) buffer voxel_counter_buffer {
    uint voxel_counter;
};

uvec3 get_path_offset(uvec3 v) {
    return (path_offset[v.x][v.y][v.z] / uvec3(1, CELL_SIZE.x, CELL_SIZE.x * CELL_SIZE.y)) % CELL_SIZE;
}

void set_path_offset(uvec3 data) {
    data.yz *= uvec2(CELL_SIZE.x, CELL_SIZE.x * CELL_SIZE.y);
    path_offset[gl_LocalInvocationID.x][gl_LocalInvocationID.y][gl_LocalInvocationID.z] = data.x + data.y + data.z;
}

float rand(vec2 v) {
    return fract(sin(dot(v, vec2(12.9898,78.233))) * 43758.5453);
}

void main() {
    uvec3 pos = gl_LocalInvocationID + 8 * gl_WorkGroupID;
    uvec3 pos1 = pos * uvec3(1, NUM_GROUPS.x, NUM_GROUPS.x * NUM_GROUPS.y);
    float r = rand(vec2(pos1.x + pos1.y + pos1.z, 0.2));
    path_offset[gl_LocalInvocationID.x][gl_LocalInvocationID.y][gl_LocalInvocationID.z] = uint(rand(vec2(r, 0.0)) * float(CELL_SIZE.x * CELL_SIZE.y * CELL_SIZE.z));
    uvec3 offset = get_path_offset(gl_LocalInvocationID);
    uint align = uint(rand(vec2(r, 1.0)) * 4.0);
    for (uint i = 0; i < 2; i++) {
        barrier();
        if ((align & 1u) != 0 && gl_LocalInvocationID.x < gl_WorkGroupSize.x - 1) {
            offset.z = get_path_offset(gl_LocalInvocationID + uvec3(1, 0, 0)).z;
        }
        if ((align & 2u) != 0 && gl_LocalInvocationID.z < gl_WorkGroupSize.z - 1) {
            offset.x = get_path_offset(gl_LocalInvocationID + uvec3(0, 0, 1)).x;
        }
        barrier();
        set_path_offset(offset);
    }
    barrier();

    align = 0;
    if (all(lessThan(gl_LocalInvocationID, OUTPUT_SIZE_GROUP))) {
        if (get_path_offset(gl_LocalInvocationID + uvec3(1, 0, 0)).z == offset.z) {
            align |= 1u;
            ivec2 dist = ivec2(get_path_offset(gl_LocalInvocationID + uvec3(1, 0, 0)).xy) + ivec2(CELL_SIZE.x, 0) - ivec2(offset.xy);
            for (int i = 0; i < dist.x; i++) {
                uvec3 offset1 = uvec3(ivec3(offset) + ivec3(i, i * dist.y / dist.x, 0));
                offset1 += pos * CELL_SIZE;
                offset1.yz *= uvec2(OUTPUT_SIZE.x, OUTPUT_SIZE.x * OUTPUT_SIZE.y);
                uint index = atomicAdd(voxel_counter, 1);
                voxels[index] = uvec2(offset1.x + offset1.y + offset1.z, 0x00ffff);
            }
        }
        if (get_path_offset(gl_LocalInvocationID + uvec3(0, 0, 1)).x == offset.x) {
            align |= 2u;
            ivec2 dist = ivec2(get_path_offset(gl_LocalInvocationID + uvec3(1, 0, 0)).zy) + ivec2(CELL_SIZE.z, 0) - ivec2(offset.zy);
            for (int i = 0; i < dist.x; i++) {
                uvec3 offset1 = uvec3(ivec3(offset) + ivec3(0, i * dist.y / dist.x, i));
                offset1 += pos * CELL_SIZE;
                offset1.yz *= uvec2(OUTPUT_SIZE.x, OUTPUT_SIZE.x * OUTPUT_SIZE.y);
                uint index = atomicAdd(voxel_counter, 1);
                voxels[index] = uvec2(offset1.x + offset1.y + offset1.z, 0x00ff00);
            }
        }

        offset += pos * CELL_SIZE;
        uint color = 0xff0000;
        if ((align & 1u) != 0) {
            color |= 0xff00;
        }
        if ((align & 2u) != 0) {
            color |= 0xff;
        }
        offset.yz *= uvec2(OUTPUT_SIZE.x, OUTPUT_SIZE.x * OUTPUT_SIZE.y);
        uint index = atomicAdd(voxel_counter, 1);
        voxels[index] = uvec2(offset.x + offset.y + offset.z, color);
    }
}

"#;

const FPS: u64 = 60;
const DELTA: f32 = 1.0 / FPS as f32;
const FRAME_NANOS: u64 = 1_000_000_000 / FPS;
const SPEED: f32 = 8.0;
const DEPTH: usize = 7;

#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
}
implement_vertex!(Vertex, position);

fn build_octree(
    mut octree_buffer: glium::buffer::Mapping<[u32]>,
    voxels: glium::buffer::ReadMapping<[(u32, u32)]>,
    length: usize,
) -> usize {
    use std::collections::{BTreeMap, HashMap};
    let mut child_map: HashMap<u32, [u32; 8]> = HashMap::new();
    let size: u32 = 1 << DEPTH;
    for i in 0..length {
        let (pos, data) = voxels[i];
        let x = pos % size;
        let y = (pos / size) % size;
        let z = pos / size / size;
        let child_index = x % 2 + 2 * (y % 2 + 2 * (z % 2));
        let parent = (x / 2) + (size / 2) * ((y / 2) + (size / 2) * (z / 2));
        (*child_map.entry(parent).or_insert([0; 8]))[child_index as usize] = data;
    }

    let mut octree_buffer_index = 9; // TODO
    for depth in (1..DEPTH).rev() {
        let size: u32 = 1 << depth;
        let mut voxel_map: BTreeMap<[u32; 8], Vec<u32>> = BTreeMap::new();
        for (parent, children) in child_map.drain() {
            voxel_map
                .entry(children)
                .or_insert_with(|| Vec::new())
                .push(parent);
        }
        for (children, parents) in voxel_map {
            let mut child_mask = 0xff00u32;
            let child_mask_index = octree_buffer_index;
            octree_buffer_index += 1;
            for i in 0..8 {
                if children[i] != 0 {
                    child_mask |= 1 << i;
                    octree_buffer[octree_buffer_index] = children[i];
                    octree_buffer_index += 1;
                }
            }
            octree_buffer[child_mask_index] = child_mask;
            for parent in parents {
                let parent_x = parent % size;
                let parent_y = (parent / size) % size;
                let parent_z = parent / size / size;
                let child_index = parent_x % 2 + 2 * (parent_y % 2 + 2 * (parent_z % 2));
                let parent_parent =
                    (parent_x / 2) + (size / 2) * ((parent_y / 2) + (size / 2) * (parent_z / 2));
                (*child_map.entry(parent_parent).or_insert([0; 8]))[child_index as usize] =
                    child_mask_index as u32;
            }
        }
    }
    let octree_length = octree_buffer_index;

    let children = child_map[&0];
    let mut child_mask = 0u32;
    octree_buffer_index = 1;
    for i in 0..8 {
        if children[i] != 0 {
            child_mask |= 1 << i;
            octree_buffer[octree_buffer_index] = children[i];
            octree_buffer_index += 1;
        }
    }
    octree_buffer[0] = child_mask;

    // for i in 0..octree_length {
    //     println!("{:06x}", octree_buffer[i]);
    // }
    println!("{}", octree_length);
    octree_length
}

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

    let program =
        glium::Program::from_source(&display, VERTEX_SHADER_SRC, RAYTRACING_SHADER_SRC, None)
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

    let gen_paths_shader =
        glium::program::ComputeShader::from_source(&display, GEN_PATHS_SRC).unwrap();

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

    build_octree(octree_buffer.map(), voxel_buffer.map_read(), count as usize);

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
