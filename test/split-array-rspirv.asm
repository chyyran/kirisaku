; SPIR-V
; Version: 1.0
; Generator: rspirv
; Bound: 32
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %4 "main" %color
OpExecutionMode %4 OriginUpperLeft
OpSource GLSL 450
OpName %4 "main"
OpName %color "color"
OpName %tex "tex"
OpName %_tex_sampler "_tex_sampler"
OpDecorate %color Location 0
OpDecorate %tex DescriptorSet 0
OpDecorate %tex Binding 1
OpDecorate %_tex_sampler DescriptorSet 0
OpDecorate %_tex_sampler Binding 1
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeFloat 32
%7 = OpTypeVector %6 4
%8 = OpTypePointer Output %7
%color = OpVariable  %8  Output
%10 = OpTypeImage %6 2D 0 0 0 1 Unknown
%11 = OpTypeSampledImage %10
%12 = OpTypeInt 32 0
%13 = OpConstant  %12  2
%14 = OpTypeArray %11 %13
%15 = OpTypePointer UniformConstant %14
%36 = OpTypeArray %10 %13
%37 = OpTypePointer UniformConstant %36
%tex = OpVariable  %37  UniformConstant
%17 = OpTypeInt 32 1
%18 = OpConstant  %17  0
%19 = OpTypePointer UniformConstant %11
%22 = OpTypeVector %6 2
%23 = OpConstant  %6  0.0
%24 = OpConstantComposite  %22  %23 %23
%27 = OpConstant  %17  1
%32 = OpTypeSampler
%33 = OpTypeArray %32 %13
%34 = OpTypePointer UniformConstant %33
%_tex_sampler = OpVariable  %34  UniformConstant
%_ptr_type_tex2d = OpTypePointer UniformConstant %10
%39 = OpTypePointer UniformConstant %32
%4 = OpFunction  %2  None %3
%5 = OpLabel
%20 = OpAccessChain  %_ptr_type_tex2d  %tex %18
%40 = OpLoad  %10  %20
%42 = OpAccessChain  %39  %_tex_sampler %18
%41 = OpLoad  %32  %42
%21 = OpSampledImage  %14  %40 %41
%25 = OpImageSampleImplicitLod  %7  %21 %24
OpStore %color %25
%26 = OpLoad  %7  %color
%28 = OpAccessChain  %_ptr_type_tex2d  %tex %27
%43 = OpLoad  %10  %28
%45 = OpAccessChain  %39  %_tex_sampler %27
%44 = OpLoad  %32  %45
%29 = OpSampledImage  %14  %43 %44
%30 = OpImageSampleImplicitLod  %7  %29 %24
%31 = OpFMul  %7  %26 %30
OpStore %color %31
OpReturn
OpFunctionEnd