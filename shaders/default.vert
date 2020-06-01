#version 140

in vec3 position;
out vec3 world_position;

uniform mat4 MVP;

void main() {
    world_position = position;
    gl_Position = MVP * vec4(position, 1.0);
}
