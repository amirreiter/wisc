pub(crate) struct SlicePointerWriter {
    head: *mut u8,
    end: *mut u8,
}

impl SlicePointerWriter {
    pub(crate) fn from_slice(slice: &mut [u8]) -> SlicePointerWriter {
        let begin = slice.as_mut_ptr();
        let end = unsafe { begin.add(slice.len()) };

        Self {
            head: begin,
            end,
        }
    }

    pub(crate) fn write(&mut self, slice: &[u8]) -> Result<(), ()> {
        unsafe {
            if self.head.add(slice.len()) > self.end {
                Err(())
            } else {
                self.head
                    .copy_from_nonoverlapping(slice.as_ptr(), slice.len());
                self.head = self.head.add(slice.len());
                Ok(())
            }
        }
    }
}
