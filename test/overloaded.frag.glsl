#version 450

#define float2 vec2
#define float3 vec3
#define float4 vec4
layout(set = 0, binding = 2) uniform sampler2D Source;

const bool linearize_input = true;
const bool assume_opaque_alpha = false;

float4 decode_input(const float4 color)
{
    if(linearize_input)
    {
        if(assume_opaque_alpha)
        {
            return float4(pow(color.rgb, float3(1.0, 1.0, 1.0)), 1.0);
        }
        else
        {
            return float4(pow(color.rgb, float3(1.0, 1.0, 1.0)), color.a);
        }
    }
    else
    {
        return color;
    }
}

float4 tex2D_linearize(const sampler2D tex, float2 tex_coords)
{   return decode_input(texture(tex, tex_coords));   }

float4 tex2D_linearize(const sampler2D tex, float3 tex_coords)
{   return decode_input(texture(tex, tex_coords.xy));   }

float4 tex2D_linearize(const sampler2D tex, float2 tex_coords, int texel_off)
{   return decode_input(textureLod(tex, tex_coords, texel_off));    }

float4 tex2D_linearize(const sampler2D tex, float3 tex_coords, int texel_off)
{   return decode_input(textureLod(tex, tex_coords.xy, texel_off));    }


void main()
{
    tex2D_linearize(Source, float3(1.0, 1.0, 1.0));
}