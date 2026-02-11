use std::any::{Any, TypeId};

pub(crate) struct VBuffer {
    pub(crate) inner: Box<dyn Any>,
    pub(crate) typeid: TypeId,

    pub(crate) stride: usize,
    pub(crate) length: usize,
}
