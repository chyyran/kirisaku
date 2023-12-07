#version 450

layout(location = 0) out vec4 color;
layout(set = 0, binding = 1) uniform sampler2D tex;

void access(sampler2D tex1) {
    color = texture(tex1, vec2(0.0));
}

void main() {
    access(tex);
}

