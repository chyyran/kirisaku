#version 450

layout(location = 0) out vec4 color;
layout(set = 0, binding = 1) uniform sampler2D[2] tex;

void access2(sampler2D[2] tex1) {
    color = texture(tex1[0], vec2(0.0));
}

void access(sampler2D[2] tex1) {
    access2(tex1);
}

void main() {
    access(tex);
}

