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

uint raytrace(vec3 pos, vec3 dir, out vec3 normal) {
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
    normal = -step(t_min, t_min0) * sign(dir);
    if (t_max < t_min || t_max < 0.0) {
        return 0;
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
            normal = -step(t_max0, vec3(t)) * sign(dir);
            if (t >= t_max) {
                return 0;
            }
            pos1 = t * dir + pos;
            cell_min = SVO_POSITION;
            cell_max = SVO_POSITION + SVO_SIZE;
            depth = 0;
        } else if (depth == MAX_DEPTH) {
            // 0xfb007d
            // 251 0 125
            return index | 0xff000000;
        }
    }

    return 0xffff0000;
}

void main() {
    vec3 dir = normalize(world_position - cam_position);
    vec3 normal;
    uint index = raytrace(cam_position, dir, normal);
    color = vec4(
        bitfieldExtract(index, 16, 8),
        bitfieldExtract(index, 8, 8),
        bitfieldExtract(index, 0, 8),
        bitfieldExtract(index, 24, 8)
    ) / 255.0;
    color.rgb *= min(max(dot(normal, normalize(vec3(-1.0, 2.0, -3.0))), 0.0) + 0.05, 1.0);
}

