#version 450

#define float2 vec2
#define float3 vec3
#define float4 vec4
layout(set = 0, binding = 2) uniform sampler2D Source;

float3 tex2Dblur3x3resize(const sampler2D tex, const float2 tex_uv, const float2 dxdy, const float sigma)
{
    //  Requires:   Global requirements must be met (see file description).
    //  Returns:    A 3x3 Gaussian blurred mipmapped texture lookup of the
    //              resized input.
    //  Description:
    //  This is the only arbitrarily resizable one-pass blur; tex2Dblur5x5resize
    //  would perform like tex2Dblur9x9, MUCH slower than tex2Dblur5resize.
    const float denom_inv = 0.5/(sigma*sigma);
    //  Load each sample.  We need all 3x3 samples.  Quad-pixel communication
    //  won't help either: This should perform like tex2Dblur5x5, but sharing a
    //  4x4 sample field would perform more like tex2Dblur8x8shared (worse).
//    const float2 sample4_uv = tex_uv;
//    const float2 dx = float2(dxdy.x, 0.0);
//    const float2 dy = float2(0.0, dxdy.y);
//    const float2 sample1_uv = sample4_uv - dy;
//    const float2 sample7_uv = sample4_uv + dy;
//    const float3 sample0 = tex2D_linearize(tex, sample1_uv - dx).rgb;
//    const float3 sample1 = tex2D_linearize(tex, sample1_uv).rgb;
//    const float3 sample2 = tex2D_linearize(tex, sample1_uv + dx).rgb;
//    const float3 sample3 = tex2D_linearize(tex, sample4_uv - dx).rgb;
//    const float3 sample4 = tex2D_linearize(tex, sample4_uv).rgb;
//    const float3 sample5 = tex2D_linearize(tex, sample4_uv + dx).rgb;
//    const float3 sample6 = tex2D_linearize(tex, sample7_uv - dx).rgb;
//    const float3 sample7 = tex2D_linearize(tex, sample7_uv).rgb;
//    const float3 sample8 = tex2D_linearize(tex, sample7_uv + dx).rgb;
//    //  Statically compute Gaussian sample weights:
//    const float w4 = 1.0;
//    const float w1_3_5_7 = exp(-LENGTH_SQ(float2(1.0, 0.0)) * denom_inv);
//    const float w0_2_6_8 = exp(-LENGTH_SQ(float2(1.0, 1.0)) * denom_inv);
//    const float weight_sum_inv = 1.0/(w4 + 4.0 * (w1_3_5_7 + w0_2_6_8));
//    //  Weight and sum the samples:
//    const float3 sum = w4 * sample4 +
//    w1_3_5_7 * (sample1 + sample3 + sample5 + sample7) +
//    w0_2_6_8 * (sample0 + sample2 + sample6 + sample8);
//    return sum * denom_inv;
    return float3(1.0, 1.0, 1.0);
}

float3 tex2Dblur3x3resize(const sampler2D tex, const float2 tex_uv,
const float2 dxdy)
{
    return tex2Dblur3x3resize(tex, tex_uv, dxdy, 0.84931640625);
}

void main()
{
    tex2Dblur3x3resize(Source, float2(1.0, 1.0), float2(1.0, 1.0));
}