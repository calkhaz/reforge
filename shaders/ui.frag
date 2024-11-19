#version 450

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout(binding = 0, set = 0) uniform sampler2D font;

layout(location = 0) out vec4 finalColor;

//layout(constant_id = 0) const bool SRGB_FRAMEBUFFER = true;

const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;

/*
vec3 linear_to_srgb(vec3 color) {
    return pow(color, vec3(INV_GAMMA));
}
vec4 linear_to_srgb(vec4 color) {
    return vec4(linear_to_srgb(color.rgb), color.a);
}
*/
void main() {
 //   if (SRGB_FRAMEBUFFER) {
    finalColor = color * texture(font, uv);
//    } else {
//        finalColor = linear_to_srgb(color * texture(font, uv));
//    }
}
