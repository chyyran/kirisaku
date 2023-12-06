use rspirv::dr::{Builder, InsertPoint, Instruction, Module, Operand};
use spirv::{Decoration, StorageClass, Word};

pub struct Pass<'a> {
    pub builder: &'a mut Builder,
    pub combined_image_samplers: Vec<Instruction>,
}

pub struct CombinedImageSampler{
    handle: spirv::Word,
    created_sampler: spirv::Word
}

impl<'a> Pass<'a> {
    fn ensure_op_type_sampler(&mut self) {
        self.builder.type_sampler();
    }

    fn find_global_instruction(&self, word: Word) -> Option<&Instruction> {
        self.builder
            .module_ref()
            .global_inst_iter()
            .find(|i| i.result_id == Some(word))
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

    fn collect_global_sampled_images(&mut self) -> Vec<CombinedImageSampler> {
        let mut image_sampler_cadidates = Vec::new();
        let mut image_sampler_types = Vec::new();

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

        for (pointer_type, combined_image_sampler) in image_sampler_cadidates {
            let Some(pointee_type) = self.find_global_instruction(pointer_type) else {
                continue
            };
            if pointee_type.class.opcode == spirv::Op::TypePointer
                && pointee_type.operands[0] ==
                Operand::StorageClass(StorageClass::UniformConstant) {
                let Some(&Operand::IdRef(sampled_image_type)) = pointee_type.operands.get(1) else {
                    continue;
                };

                let Some(sampled_image_type) = self.find_global_instruction(sampled_image_type) else {
                    continue;
                };
                let Some(base_type) = self.get_base_type_for_sampled_image(sampled_image_type) else {
                    continue;
                };

                let Some(combined_image_sampler) = combined_image_sampler else {
                    continue;
                };

                // insert the sampler
                // todo: type binding array
                if base_type.class.opcode == spirv::Op::TypeImage {
                    let sampler_type = self.builder.type_sampler();
                    let pointer_type = self.builder
                        .type_pointer(None, StorageClass::UniformConstant, sampler_type);

                    let sampler_uniform = self.builder
                        .variable(pointer_type, None, StorageClass::UniformConstant, None);

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

                    image_sampler_types.push(CombinedImageSampler {
                        handle: combined_image_sampler,
                        created_sampler: sampler_uniform,
                    });
                }
            }
        }

        image_sampler_types
    }
}

fn find_op_loads(module: &mut Builder) {}

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
    use std::rc::Rc;
    use rspirv::binary::Disassemble;

    #[test]
    fn it_works() {
        let mut bin = Vec::new();

        File::open("./test/combined-image-sampler.spv")
            .unwrap()
            .read_to_end(&mut bin)
            .unwrap();

        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_bytes(bin, &mut loader).unwrap();
        let module = loader.module();
        let mut builder = Builder::new_from_module(module);

        let mut builder = Pass {
            builder: &mut builder,
            combined_image_samplers: vec![],
        };

        builder.ensure_op_type_sampler();
        builder.collect_global_sampled_images();
        find_op_type_sampled_image(&builder.builder.module_ref());

        println!("{}", builder.builder.module_ref().disassemble());
    }
}
