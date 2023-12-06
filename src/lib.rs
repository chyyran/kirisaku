use rspirv::dr::{Builder, InsertPoint, Instruction, Module, Operand};
use rustc_hash::FxHashMap;
use spirv::{Decoration, Op, StorageClass, Word};

pub struct Pass<'a> {
    pub builder: &'a mut Builder,
}

#[derive(Debug)]
pub struct CombinedImageSampler{
    sampled_image_handle: spirv::Word,
    created_sampler: spirv::Word,
    pointer_type_id: spirv::Word,
    sampled_image_type: Instruction,
    base_type: Instruction,
}

impl<'a> Pass<'a> {
    pub fn new(builder: &'a mut Builder) -> Self {
        let mut val = Self {
            builder,
        };

        val
    }

    fn do_pass(&mut self) {
        let combined_image_samplers = self.collect_global_sampled_images();
        self.retype_combined_image_samplers(&combined_image_samplers);
        self.rewrite_loads(&combined_image_samplers);
    }

    fn ensure_op_type_sampler(&mut self) {
        self.builder.type_sampler();
    }

    fn find_global_instruction(&self, word: Word) -> Option<&Instruction> {
        self.builder
            .module_ref()
            .global_inst_iter()
            .find(|i| i.result_id == Some(word))
    }

    fn find_global_instruction_mut(&mut self, word: Word) -> Option<&mut Instruction> {
        self.builder
            .module_mut()
            .global_inst_iter_mut()
            .find(|i| i.result_id == Some(word))
    }

    fn create_sampler_name(&mut self, word: Word) -> Option<String> {
        self.builder
            .module_ref()
            .debug_names
            .iter()
            .find_map(|i| {
                if i.class.opcode != spirv::Op::Name {
                    return None;
                }

                let Some(&Operand::IdRef(target)) = &i.operands.get(0) else {
                    return None
                };

                if target != word {
                    return None;
                }

                let Some(Operand::LiteralString(string)) = &i.operands.get(1) else {
                    return None
                };

                return Some(format!("_{string}_sampler"));
            })
    }

    fn get_base_type_for_sampled_image(&self, inst: &Instruction) -> Option<&Instruction> {
        if inst.class.opcode != spirv::Op::TypeSampledImage {
            return None;
        }

        let Some(&Operand::IdRef(referand)) = inst.operands.get(0) else {
            return None;
        };

        self.find_global_instruction(referand)
    }

    fn collect_global_sampled_images(&mut self) -> FxHashMap<spirv::Word, CombinedImageSampler> {
        let mut image_sampler_cadidates = Vec::new();
        let mut image_sampler_types = FxHashMap::default();

        for global in self.builder.module_ref().types_global_values.iter() {
            if global.class.opcode == spirv::Op::Variable
                && global.operands[0] == Operand::StorageClass(StorageClass::UniformConstant)
            {
                let pointer_type = global.result_type;
                let Some(pointer_type) = pointer_type else {
                    continue;
                };
                image_sampler_cadidates.push((pointer_type, global.result_id))
            }
        }

        for (pointer_type_id, combined_image_sampler) in image_sampler_cadidates {
            let Some(pointer_type) = self.find_global_instruction(pointer_type_id).cloned() else {
                continue
            };
            if pointer_type.class.opcode == spirv::Op::TypePointer
                && pointer_type.operands[0] ==
                Operand::StorageClass(StorageClass::UniformConstant) {
                let Some(&Operand::IdRef(sampled_image_type)) = pointer_type.operands.get(1) else {
                    continue;
                };

                let Some(sampled_image_type) = self.find_global_instruction(sampled_image_type).cloned() else {
                    continue;
                };
                let Some(base_type) = self.get_base_type_for_sampled_image(&sampled_image_type).cloned() else {
                    continue;
                };

                let Some(combined_image_sampler) = combined_image_sampler else {
                    continue;
                };

                // insert the sampler
                // todo: type binding array
                if base_type.class.opcode == spirv::Op::TypeImage {
                    let sampler_type = self.builder.type_sampler();
                    let sampler_pointer_type = self.builder
                        .type_pointer(None, StorageClass::UniformConstant, sampler_type);

                    let sampler_uniform = self.builder
                        .variable(sampler_pointer_type, None, StorageClass::UniformConstant, None);

                    let decorations: Vec<Instruction> = self.builder.module_ref()
                        .annotations.iter().filter_map(|f| {

                        if f.class.opcode == spirv::Op::Decorate && f.operands[0] ==
                            Operand::IdRef(combined_image_sampler)
                        {
                            Some(f.clone())
                        } else {
                            None
                        }
                    }).collect();

                    if let Some(name) = self.create_sampler_name(combined_image_sampler) {
                        self.builder.name(sampler_uniform, name);
                    }

                    // Clone decorations to the created sampler
                    for decoration in decorations {
                        let Operand::Decoration(decoration_type) = decoration.operands[1] else {
                            continue;
                        };

                        self.builder
                            .decorate(
                                sampler_uniform,
                                decoration_type,
                                decoration.operands[2..].iter()
                                    .map(|f| f.clone())
                            )
                    }

                    image_sampler_types.insert(combined_image_sampler, CombinedImageSampler {
                        sampled_image_handle: combined_image_sampler,
                        created_sampler: sampler_uniform,
                        sampled_image_type,
                        pointer_type_id,
                        base_type,
                    });
                }


            }
        }

        image_sampler_types
    }


    fn retype_combined_image_samplers(&mut self, combined_image_samplers: &FxHashMap<spirv::Word, CombinedImageSampler>) {
        for sampled_image in combined_image_samplers.values() {
            let Some(Some(base_type_id)) = self.get_base_type_for_sampled_image(&sampled_image.sampled_image_type).map(|f| f.result_id) else {
                continue;
            };

            let Some(image_pointer) = self.find_global_instruction_mut(sampled_image.pointer_type_id) else {
                continue;
            };

            if image_pointer.class.opcode == spirv::Op::TypePointer {
                image_pointer.operands[1] = Operand::IdRef(base_type_id);
            }
        }
    }

    fn rewrite_loads(&mut self, combined_image_samplers: &FxHashMap<spirv::Word, CombinedImageSampler>) {
        let op_load_texture_id = self.builder.id();
        let op_load_sampler_id = self.builder.id();

        for function in self.builder.module_mut().functions.iter_mut() {
            for block in function.blocks.iter_mut() {
                let mut instructions = Vec::new();
                for instr in block.instructions.drain(..) {
                    if instr.class.opcode != Op::Load {
                        instructions.push(instr);
                        continue;
                    }

                    let Some(Operand::IdRef(load_ptr)) = &instr.operands.get(0) else {
                        instructions.push(instr);
                        continue;
                    };

                    let Some(combined_image_sampler) = combined_image_samplers.get(load_ptr) else {
                        instructions.push(instr);
                        continue;
                    };


                    // todo: rewrite opaccess as well?


                    let mut load_instr = instr.clone();

                    load_instr.result_type = combined_image_sampler.base_type.result_id;
                    load_instr.result_id = Some(op_load_texture_id);
                    instructions.push(load_instr);

                    // load the sampler
                    instructions.push(Instruction {
                        class: rspirv::grammar::CoreInstructionTable::get(spirv::Op::Load),
                        result_type: combined_image_sampler.sampled_image_type.result_id,
                        result_id: Some(op_load_sampler_id),
                        operands: vec![
                            Operand::IdRef(combined_image_sampler.created_sampler)
                        ],
                    });


                    // reuse the old id for the OpSampleImage
                    instructions.push(Instruction {
                        class: rspirv::grammar::CoreInstructionTable::get(spirv::Op::SampledImage),
                        result_type: combined_image_sampler.sampled_image_type.result_id,
                        result_id: instr.result_id,

                        operands: vec![
                            Operand::IdRef(op_load_texture_id),
                            Operand::IdRef(op_load_sampler_id),
                        ],
                    });

                }

                block.instructions = instructions;
            }
        }


    }
}

fn find_op_type_sampled_image(module: &Module) {
    for op in module.types_global_values.iter() {
        eprintln!("{:?}", op);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;
    use std::rc::Rc;
    use naga::back::wgsl::WriterFlags;
    use naga::front::spv::Options;
    use naga::valid::{Capabilities, ValidationFlags};
    use rspirv::binary::{Assemble, Disassemble};

    fn check_wgsl(path: impl AsRef<Path>) {
        let mut bin = Vec::new();

        File::open(path)
            .unwrap()
            .read_to_end(&mut bin)
            .unwrap();


        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_bytes(bin, &mut loader).unwrap();
        let module = loader.module();
        let mut builder = Builder::new_from_module(module);

        let mut pass = Pass {
            builder: &mut builder,
        };

        pass.ensure_op_type_sampler();
        pass.do_pass();
        // find_op_type_sampled_image(&builder.builder.module_ref());

        println!("{}", pass.builder.module_ref().disassemble());

        let spirv = builder.module()
            .assemble();

        let mut module = naga::front::spv::parse_u8_slice(bytemuck::cast_slice(&spirv), &Options {
            adjust_coordinate_space: false,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        }).unwrap();

        let images = module
            .global_variables
            .iter()
            .filter(|&(_, gv)| {
                let ty = &module.types[gv.ty];
                match ty.inner {
                    naga::TypeInner::Image { .. } => true,
                    naga::TypeInner::BindingArray { base, .. } => {
                        let ty = &module.types[base];
                        matches!(ty.inner, naga::TypeInner::Image { .. })
                    }
                    _ => false,
                }
            })
            .map(|(_, gv)| (gv.binding.clone(), gv.space))
            .collect::<naga::FastHashSet<_>>();

        module
            .global_variables
            .iter_mut()
            .filter(|(_, gv)| {
                let ty = &module.types[gv.ty];
                match ty.inner {
                    naga::TypeInner::Sampler { .. } => true,
                    naga::TypeInner::BindingArray { base, .. } => {
                        let ty = &module.types[base];
                        matches!(ty.inner, naga::TypeInner::Sampler { .. })
                    }
                    _ => false,
                }
            })
            .for_each(|(_, gv)| {
                if images.contains(&(gv.binding.clone(), gv.space)) {
                    if let Some(binding) = &mut gv.binding {
                        binding.group = 1;
                    }
                }
            });

        let mut valid = naga::valid::Validator::new(ValidationFlags::all(), Capabilities::empty());
        let info = valid.validate(&module).unwrap();

        let wgsl = naga::back::wgsl::write_string(&module, &info, WriterFlags::EXPLICIT_TYPES)
            .unwrap();

        println!("{}", wgsl);

    }
    #[test]
    fn it_works() {

        // check_wgsl("./test/combined-image-sampler.spv");

        check_wgsl("./test/combined-image-sampler-array.spv");

    }
}
