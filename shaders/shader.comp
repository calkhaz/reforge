#version 450

layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, rgba8) uniform writeonly image2D outputImage;

void main()
{	
    vec4 res = vec4(1.0);

    imageStore(outputImage, ivec2(gl_GlobalInvocationID.xy), res);
}
