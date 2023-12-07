#version 450

layout(location = 0) out vec4 color;
layout(set = 0, binding = 1) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler _tex_sampler;

void access(texture2D tex1, sampler samp) {
    color = texture(sampler2D(tex1, samp), vec2(0.0));
}

void main() {
    access(tex, _tex_sampler);
}

