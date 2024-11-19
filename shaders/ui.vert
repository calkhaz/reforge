#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(push_constant) uniform Matrices {
    mat4 ortho_proj;
} matrices;

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec2 oUV;

const float GAMMA = 2.2;

vec3 srgb_to_linear(vec3 color) {
    return pow(color, vec3(GAMMA));
}
vec4 srgb_to_linear(vec4 color) {
    return vec4(srgb_to_linear(color.rgb), color.a);
}

void main() {
    oColor = srgb_to_linear(color);
    oUV = uv;

    gl_Position = matrices.ortho_proj * vec4(pos.x, pos.y, 0.0, 1.0);
}
